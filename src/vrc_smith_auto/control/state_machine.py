from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from vrc_smith_auto.config import AppConfig, TriggerConfig
from vrc_smith_auto.vision.models import DetectionResult

_MIN_DT_S = 1e-4
_MAX_DT_S = 0.25
_COOLDOWN_EPS_MS = 1e-9


class TriggerState(str, Enum):
    """Lifecycle states for the blacksmith timing trigger controller."""

    IDLE = "IDLE"
    ARMED = "ARMED"
    LATCHED = "LATCHED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True, slots=True)
class StateTransition:
    """Single deterministic state change emitted during TriggerStateMachine.update()."""

    from_state: TriggerState
    to_state: TriggerState
    reason: str


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Output of TriggerStateMachine.update() for a single detection frame."""

    should_click: bool
    state: TriggerState
    previous_state: TriggerState
    transitions: tuple[StateTransition, ...]
    raw_norm_pos: float | None
    predicted_norm_pos: float | None
    velocity_norm_per_s: float
    cooldown_remaining_ms: float
    hold_ms: float
    dry_run: bool
    reason: str
    timestamp_s: float
    # Non-zero when the button should be released after this delay (seconds)
    # instead of immediately, so the bar is centred when the release-judged
    # game processes the pointer-up.  Derived from the predicted position.
    release_delay_s: float = 0.0

    @property
    def effective_norm_pos(self) -> float | None:
        return self.predicted_norm_pos


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _is_gold_jump(
    prev_span: tuple[float, float] | None,
    curr_span: tuple[float, float] | None,
) -> bool:
    """Detect a strike-to-strike Gold-zone change from the Gold length alone.

    ``gold_span`` is a 1D projection onto the *current* frame's gauge axis, in
    absolute screen pixels.  The gauge is world-anchored, so it translates and
    rotates on screen whenever the camera or avatar moves; a small change in the
    estimated axis tilt alone shifts that projection by tens of pixels even when
    the physical gauge is stationary.  Comparing the absolute projected centre
    therefore fires on ordinary camera sway and wipes the velocity estimate
    exactly in the fast, narrow-Gold phase where lead compensation is needed.

    A new strike re-rolls the Gold length, so the length delta (invariant to
    screen translation and rotation) is used as the jump signal instead.
    """
    if prev_span is None or curr_span is None:
        return False
    prev_len = abs(prev_span[1] - prev_span[0])
    curr_len = abs(curr_span[1] - curr_span[0])
    return abs(curr_len - prev_len) > max(3.0, 0.15 * prev_len)


class TriggerStateMachine:
    """Hysteresis + cooldown + exit-gold debounce controller for timing strikes.

    Enforces a dual-guard against double-clicks across instant H_k -> H_{k+1}
    strike transitions: after clicking, the controller stays LATCHED until the
    red bar physically exits the Gold zone (|norm_pos| > exit_margin) AND waits
    until cooldown_ms has elapsed before returning to ARMED.
    """

    def __init__(self, config: TriggerConfig | AppConfig | None = None) -> None:
        if config is None:
            self._config = TriggerConfig()
        elif isinstance(config, AppConfig):
            self._config = config.trigger
        elif isinstance(config, TriggerConfig):
            self._config = config
        else:
            raise TypeError(
                f"config must be TriggerConfig, AppConfig, or None; got {type(config).__name__}"
            )

        self._state: TriggerState = TriggerState.IDLE
        self._lost_frames: int = 0
        self._last_trigger_s: float | None = None
        self._trigger_count: int = 0
        self._prev_norm_pos: float | None = None
        self._prev_timestamp_s: float | None = None
        self._prev_gold_span: tuple[float, float] | None = None
        self._velocity_norm: float = 0.0
        self._has_velocity: bool = False

    @property
    def config(self) -> TriggerConfig:
        return self._config

    @property
    def state(self) -> TriggerState:
        return self._state

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def last_trigger_s(self) -> float | None:
        return self._last_trigger_s

    def reset(self) -> None:
        self._state = TriggerState.IDLE
        self._lost_frames = 0
        self._last_trigger_s = None
        self._trigger_count = 0
        self._clear_velocity()

    def _clear_velocity(self) -> None:
        self._prev_norm_pos = None
        self._prev_timestamp_s = None
        self._prev_gold_span = None
        self._velocity_norm = 0.0
        self._has_velocity = False

    def _cooldown_remaining_ms(self, now_s: float) -> float:
        if self._last_trigger_s is None or self._config.cooldown_ms <= 0.0:
            return 0.0
        elapsed_ms = max(0.0, (now_s - self._last_trigger_s) * 1000.0)
        rem_ms = max(0.0, self._config.cooldown_ms - elapsed_ms)
        ulp_eps_ms = 1.5 * math.ulp(max(abs(now_s), abs(self._last_trigger_s))) * 1000.0
        eps_ms = max(_COOLDOWN_EPS_MS, ulp_eps_ms)
        return 0.0 if rem_ms <= eps_ms else rem_ms

    def _update_velocity(
        self,
        raw_pos: float,
        gold_span: tuple[float, float] | None,
        now_s: float,
    ) -> float:
        if _is_gold_jump(self._prev_gold_span, gold_span):
            self._velocity_norm = 0.0
            self._has_velocity = False
        elif self._prev_norm_pos is not None and self._prev_timestamp_s is not None:
            dt_s = now_s - self._prev_timestamp_s
            if dt_s <= _MIN_DT_S or dt_s > _MAX_DT_S:
                self._velocity_norm = 0.0
                self._has_velocity = False
            else:
                inst_v = (raw_pos - self._prev_norm_pos) / dt_s
                if self._has_velocity and inst_v * self._velocity_norm < 0.0:
                    self._velocity_norm = inst_v
                elif self._has_velocity:
                    self._velocity_norm = 0.65 * inst_v + 0.35 * self._velocity_norm
                else:
                    self._velocity_norm = inst_v
                    self._has_velocity = True
        else:
            self._velocity_norm = 0.0
            self._has_velocity = False

        self._prev_norm_pos = raw_pos
        self._prev_timestamp_s = now_s
        self._prev_gold_span = gold_span
        return self._velocity_norm

    def update(
        self,
        detection: DetectionResult,
        timestamp_s: float,
        *,
        allow_trigger: bool = True,
    ) -> TriggerDecision:
        if not _is_finite_number(timestamp_s):
            raise ValueError(f"timestamp_s must be a finite number, got {timestamp_s!r}")
        now_s = float(timestamp_s)

        prev_state = self._state
        transitions: list[StateTransition] = []
        should_click = False
        reason = ""

        # 1. Missing gauge handling (with rearm_lost_frames debounce)
        if not detection.gauge_present:
            self._lost_frames += 1
            self._clear_velocity()
            if self._state == TriggerState.IDLE:
                reason = "gauge_absent"
            elif self._lost_frames >= self._config.rearm_lost_frames:
                transitions.append(
                    StateTransition(
                        from_state=self._state,
                        to_state=TriggerState.IDLE,
                        reason="gauge_lost_rearm",
                    )
                )
                self._state = TriggerState.IDLE
                self._last_trigger_s = None
                reason = "gauge_lost_rearm"
            else:
                reason = "gauge_absent_preserved"

            return TriggerDecision(
                should_click=False,
                state=self._state,
                previous_state=prev_state,
                transitions=tuple(transitions),
                raw_norm_pos=None,
                predicted_norm_pos=None,
                velocity_norm_per_s=0.0,
                cooldown_remaining_ms=self._cooldown_remaining_ms(now_s),
                hold_ms=self._config.hold_ms,
                dry_run=self._config.dry_run,
                reason=reason,
                timestamp_s=now_s,
            )

        # Gauge is present
        self._lost_frames = 0
        rem_ms = self._cooldown_remaining_ms(now_s)

        # 2. Missing / non-finite norm_pos handling
        if not _is_finite_number(detection.norm_pos):
            self._clear_velocity()
            if self._state == TriggerState.COOLDOWN and rem_ms == 0.0:
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.COOLDOWN,
                        to_state=TriggerState.ARMED,
                        reason="cooldown_elapsed",
                    )
                )
                self._state = TriggerState.ARMED
            reason = "red_missing"
            return TriggerDecision(
                should_click=False,
                state=self._state,
                previous_state=prev_state,
                transitions=tuple(transitions),
                raw_norm_pos=None,
                predicted_norm_pos=None,
                velocity_norm_per_s=0.0,
                cooldown_remaining_ms=rem_ms,
                hold_ms=self._config.hold_ms,
                dry_run=self._config.dry_run,
                reason=reason,
                timestamp_s=now_s,
            )

        # 3. Finite norm_pos present: update velocity & predicted position
        raw_pos = float(detection.norm_pos)  # type: ignore[arg-type]
        vel = self._update_velocity(raw_pos, detection.gold_span, now_s)
        if self._config.lead_ms > 0.0 and self._has_velocity:
            lead_s = self._config.lead_ms / 1000.0
            lead_offset = _clamp(
                vel * lead_s,
                -self._config.max_lead_offset,
                self._config.max_lead_offset,
            )
            pred_pos = raw_pos + lead_offset
        else:
            pred_pos = raw_pos

        # 4. State machine transitions
        if self._state == TriggerState.IDLE:
            if abs(raw_pos) > self._config.entry_margin:
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.IDLE,
                        to_state=TriggerState.ARMED,
                        reason="gauge_acquired",
                    )
                )
                self._state = TriggerState.ARMED
                reason = "gauge_acquired"
            else:
                reason = "idle_waiting_outside_gold"

        elif self._state == TriggerState.LATCHED:
            if abs(raw_pos) <= self._config.exit_margin:
                reason = "latched_in_gold"
            elif rem_ms > 0.0:
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.LATCHED,
                        to_state=TriggerState.COOLDOWN,
                        reason="exited_gold_waiting_cooldown",
                    )
                )
                self._state = TriggerState.COOLDOWN
                reason = "exited_gold_waiting_cooldown"
            else:
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.LATCHED,
                        to_state=TriggerState.ARMED,
                        reason="exited_gold_cooldown_elapsed",
                    )
                )
                self._state = TriggerState.ARMED
                reason = "exited_gold_cooldown_elapsed"
                # Do NOT trigger on the exit frame itself

        elif self._state == TriggerState.COOLDOWN:
            if rem_ms > 0.0:
                reason = "cooldown_active"
            else:
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.COOLDOWN,
                        to_state=TriggerState.ARMED,
                        reason="cooldown_elapsed",
                    )
                )
                self._state = TriggerState.ARMED
                reason = "cooldown_elapsed"
                # Fall through immediately to ARMED check

        else:
            reason = "armed_waiting"

        # Evaluate ARMED trigger condition (initial ARMED or fall-through from COOLDOWN)
        if self._state == TriggerState.ARMED and reason not in (
            "gauge_acquired",
            "exited_gold_cooldown_elapsed",
        ):
            raw_in = abs(raw_pos) <= self._config.entry_margin
            pred_in = (
                self._config.lead_ms > 0.0
                and abs(pred_pos) <= self._config.entry_margin
            )
            if allow_trigger and (raw_in or pred_in) and rem_ms == 0.0:
                trig_reason = "entered_gold" if raw_in else "predicted_gold_entry"
                transitions.append(
                    StateTransition(
                        from_state=TriggerState.ARMED,
                        to_state=TriggerState.LATCHED,
                        reason=trig_reason,
                    )
                )
                self._state = TriggerState.LATCHED
                self._last_trigger_s = now_s
                self._trigger_count += 1
                should_click = True
                reason = trig_reason
                rem_ms = float(self._config.cooldown_ms)
            elif not transitions:
                reason = (
                    "mouse_held_waiting"
                    if (not allow_trigger and (raw_in or pred_in) and rem_ms == 0.0)
                    else "armed_waiting"
                )

        release_delay_s = 0.0
        if should_click and self._config.lead_ms > 0.0 and abs(vel) > 1e-6:
            # -pred_pos / vel is the time from "now" until the bar reaches the
            # centre at now + lead_ms.  Releasing then puts the pointer-up at
            # the centre for a game that judges at release + lead_ms, instead of
            # at whatever position the 60 FPS trigger frame happened to catch.
            release_delay_s = _clamp(-pred_pos / vel, 0.0, 0.030)

        return TriggerDecision(
            should_click=should_click,
            state=self._state,
            previous_state=prev_state,
            transitions=tuple(transitions),
            raw_norm_pos=raw_pos,
            predicted_norm_pos=pred_pos,
            velocity_norm_per_s=vel,
            cooldown_remaining_ms=rem_ms,
            hold_ms=self._config.hold_ms,
            dry_run=self._config.dry_run,
            reason=reason,
            timestamp_s=now_s,
            release_delay_s=release_delay_s,
        )
