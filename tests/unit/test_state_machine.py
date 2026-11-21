from __future__ import annotations

import math

import pytest

from vrc_smith_auto.config import AppConfig, TriggerConfig
from vrc_smith_auto.control import (
    StateTransition,
    TriggerDecision,
    TriggerState,
    TriggerStateMachine,
)
from vrc_smith_auto.vision.models import DetectionResult


def _det(
    *,
    present: bool = True,
    norm_pos: float | None = 0.0,
    gold_span: tuple[float, float] | None = (400.0, 440.0),
    confidence: float = 0.0,
) -> DetectionResult:
    return DetectionResult(
        gauge_present=present,
        norm_pos=norm_pos if present else None,
        gold_span=gold_span if present else None,
        confidence=confidence,
    )


def test_constructor_variants_and_reset() -> None:
    sm_default = TriggerStateMachine()
    assert sm_default.config == TriggerConfig()
    assert sm_default.state == TriggerState.IDLE
    assert sm_default.trigger_count == 0
    assert sm_default.lost_frames == 0
    assert sm_default.last_trigger_s is None

    app_cfg = AppConfig().with_overrides(cooldown_ms=150.0, hold_ms=35.0, dry_run=True)
    sm_app = TriggerStateMachine(app_cfg)
    assert sm_app.config.cooldown_ms == 150.0
    assert sm_app.config.hold_ms == 35.0
    assert sm_app.config.dry_run is True

    with pytest.raises(TypeError):
        TriggerStateMachine("invalid")  # type: ignore[arg-type]

    # Direct norm_pos=0.0 from IDLE waits outside Gold without clicking
    dec_idle = sm_app.update(_det(norm_pos=0.0), 0.9)
    assert dec_idle.should_click is False
    assert dec_idle.state == TriggerState.IDLE
    assert dec_idle.reason == "idle_waiting_outside_gold"

    # Arm outside Gold, then enter Gold -> clicks
    dec_arm = sm_app.update(_det(norm_pos=-1.5), 0.95)
    assert dec_arm.should_click is False
    assert dec_arm.state == TriggerState.ARMED

    dec = sm_app.update(_det(norm_pos=0.0), 1.0)
    assert dec.should_click is True
    assert dec.dry_run is True
    assert dec.hold_ms == 35.0
    assert sm_app.trigger_count == 1
    assert sm_app.last_trigger_s == 1.0

    sm_app.reset()
    assert sm_app.state == TriggerState.IDLE
    assert sm_app.trigger_count == 0
    assert sm_app.lost_frames == 0
    assert sm_app.last_trigger_s is None


def test_startup_and_basic_hysteresis_lifecycle() -> None:
    # Lead prediction is disabled here so this test isolates the zero-latency
    # hysteresis/cooldown transitions; predicted entry is covered separately.
    sm = TriggerStateMachine(
        TriggerConfig(entry_margin=0.95, exit_margin=1.25, cooldown_ms=200.0, lead_ms=0.0)
    )

    # IDLE stays IDLE on absent gauge
    d0 = sm.update(_det(present=False), 0.00)
    assert isinstance(d0, TriggerDecision)
    assert d0.state == TriggerState.IDLE
    assert d0.should_click is False

    # First valid detection outside entry_margin -> ARMED
    d1 = sm.update(_det(norm_pos=2.0), 0.01)
    assert d1.previous_state == TriggerState.IDLE
    assert d1.state == TriggerState.ARMED
    assert d1.transitions == (StateTransition(TriggerState.IDLE, TriggerState.ARMED, "gauge_acquired"),)
    assert d1.should_click is False

    # 0.96 is strictly outside entry_margin=0.95 -> stays ARMED
    d2 = sm.update(_det(norm_pos=0.96), 0.02)
    assert d2.state == TriggerState.ARMED
    assert d2.should_click is False

    # Exact entry_margin=0.95 -> triggers and latches
    d3 = sm.update(_det(norm_pos=0.95), 0.03)
    assert d3.previous_state == TriggerState.ARMED
    assert d3.state == TriggerState.LATCHED
    assert d3.should_click is True
    assert d3.reason == "entered_gold"
    assert d3.cooldown_remaining_ms == pytest.approx(200.0)
    assert sm.trigger_count == 1

    # Between entry_margin and exit_margin (including 1.25 inclusive) -> stays LATCHED
    for idx, pos in enumerate((0.96, 1.25), start=1):
        d_latch = sm.update(_det(norm_pos=pos), 0.03 + idx * 0.01)
        assert d_latch.state == TriggerState.LATCHED
        assert d_latch.should_click is False

    # Strictly beyond exit_margin (1.26) while cooldown remaining -> LATCHED -> COOLDOWN
    d_exit = sm.update(_det(norm_pos=1.26), 0.08)
    assert d_exit.previous_state == TriggerState.LATCHED
    assert d_exit.state == TriggerState.COOLDOWN
    assert d_exit.should_click is False
    assert d_exit.cooldown_remaining_ms == pytest.approx(150.0)

    # Just before cooldown expires -> stays COOLDOWN
    d_wait = sm.update(_det(norm_pos=1.26), 0.229)
    assert d_wait.state == TriggerState.COOLDOWN
    assert d_wait.cooldown_remaining_ms == pytest.approx(1.0)
    assert d_wait.should_click is False

    # At t=0.23s (elapsed == 200.0 ms) -> COOLDOWN -> ARMED
    d_rearm = sm.update(_det(norm_pos=1.26), 0.23)
    assert d_rearm.previous_state == TriggerState.COOLDOWN
    assert d_rearm.state == TriggerState.ARMED
    assert d_rearm.cooldown_remaining_ms == 0.0
    assert d_rearm.should_click is False


def test_idle_requires_outside_gold_arming_and_cooldown_fallthrough() -> None:
    sm = TriggerStateMachine(TriggerConfig(cooldown_ms=100.0))

    # IDLE must NOT fall through to LATCHED when gauge appears with red already in Gold
    d_idle = sm.update(_det(norm_pos=0.0, confidence=0.0), 0.0)
    assert d_idle.should_click is False
    assert d_idle.previous_state == TriggerState.IDLE
    assert d_idle.state == TriggerState.IDLE
    assert d_idle.reason == "idle_waiting_outside_gold"
    assert d_idle.transitions == ()

    # Arm outside Gold, then enter Gold
    d_arm = sm.update(_det(norm_pos=-1.5), 0.01)
    assert d_arm.should_click is False
    assert d_arm.state == TriggerState.ARMED
    assert d_arm.transitions == (
        StateTransition(TriggerState.IDLE, TriggerState.ARMED, "gauge_acquired"),
    )

    d_first = sm.update(_det(norm_pos=0.0, confidence=0.0), 0.02)
    assert d_first.should_click is True
    assert d_first.previous_state == TriggerState.ARMED
    assert d_first.state == TriggerState.LATCHED

    # Exit Gold -> COOLDOWN
    d_out = sm.update(_det(norm_pos=1.5), 0.05)
    assert d_out.state == TriggerState.COOLDOWN
    assert d_out.should_click is False

    # Cooldown expires while red is already back in Gold -> COOLDOWN -> ARMED -> LATCHED
    d_second = sm.update(_det(norm_pos=0.2), 0.12)
    assert d_second.should_click is True
    assert d_second.previous_state == TriggerState.COOLDOWN
    assert d_second.state == TriggerState.LATCHED
    assert len(d_second.transitions) == 2


def test_hk_to_hk1_overlapping_gold_prevents_double_click() -> None:
    sm = TriggerStateMachine(TriggerConfig(cooldown_ms=200.0, entry_margin=0.95, exit_margin=1.25))

    # Arm outside Gold, then strike H_k click at t=0.0s
    assert sm.update(_det(norm_pos=-1.6, gold_span=(400.0, 438.0)), -0.02).should_click is False
    d0 = sm.update(_det(norm_pos=0.8, gold_span=(400.0, 438.0)), 0.0)
    assert d0.should_click is True
    assert d0.state == TriggerState.LATCHED

    # H_k -> H_{k+1} transition at t=0.20s (cooldown elapsed) with gold_span shift, red still <= 1.25
    d1 = sm.update(_det(norm_pos=0.9, gold_span=(410.0, 446.0)), 0.20)
    assert d1.state == TriggerState.LATCHED
    assert d1.should_click is False
    assert d1.cooldown_remaining_ms == 0.0

    d2 = sm.update(_det(norm_pos=1.20, gold_span=(410.0, 446.0)), 0.25)
    assert d2.state == TriggerState.LATCHED
    assert d2.should_click is False

    # Physical exit beyond exit_margin -> direct LATCHED -> ARMED (no COOLDOWN stop, no click)
    d3 = sm.update(_det(norm_pos=1.40, gold_span=(410.0, 446.0)), 0.267)
    assert d3.previous_state == TriggerState.LATCHED
    assert d3.state == TriggerState.ARMED
    assert d3.transitions == (
        StateTransition(TriggerState.LATCHED, TriggerState.ARMED, "exited_gold_cooldown_elapsed"),
    )
    assert d3.should_click is False

    # Next genuine pass triggers cleanly
    d4 = sm.update(_det(norm_pos=0.5, gold_span=(410.0, 446.0)), 0.40)
    assert d4.should_click is True
    assert sm.trigger_count == 2


def test_seg2_h9_rapid_second_pass_at_60hz() -> None:
    # Zero-latency sampling test: disable lead so the expected click indices are
    # the raw in-Gold samples.
    sm = TriggerStateMachine(TriggerConfig(lead_ms=0.0))
    dt = 1.0 / 60.0
    # 2 frames in Gold (33.3ms), 18 frames outside (300ms), 2 frames in Gold (33.3ms)
    positions = [
        -1.8,
        -1.3,
        -0.5,
        0.5,  # f=2,3 pass 1 (t=33.3ms, 50ms)
        1.4,
        1.8,
        2.1,
        2.3,
        2.4,
        2.3,
        2.1,
        1.9,
        1.7,
        1.5,
        1.4,
        1.35,
        1.3,
        1.28,
        1.32,
        1.45,
        1.6,
        1.3,  # 18 frames outside
        0.4,
        -0.6,  # f=22,23 pass 2 (t=366.7ms, 383.3ms)
        -1.5,
    ]
    click_indices = []
    for idx, pos in enumerate(positions):
        dec = sm.update(_det(norm_pos=pos, gold_span=(428.0, 444.0)), idx * dt)
        if dec.should_click:
            click_indices.append(idx)

    assert click_indices == [2, 22]


def test_velocity_lead_compensation_and_safety_guards() -> None:
    # Default lead_ms == 0.0 -> predicted == raw, no early trigger
    sm_no_lead = TriggerStateMachine(TriggerConfig(lead_ms=0.0, entry_margin=0.95))
    sm_no_lead.update(_det(norm_pos=-1.6), 0.0)
    d_nolead = sm_no_lead.update(_det(norm_pos=-1.0), 1.0 / 60.0)
    assert d_nolead.predicted_norm_pos == pytest.approx(d_nolead.raw_norm_pos)
    assert d_nolead.should_click is False

    # With lead_ms=20.0, max_lead_offset=0.5:
    cfg = TriggerConfig(lead_ms=20.0, max_lead_offset=0.5, entry_margin=0.95)
    sm = TriggerStateMachine(cfg)
    dt = 1.0 / 60.0

    # Moving toward Gold: -1.6 -> -1.2 over 16.67ms => v=+24/s, lead=+0.48, pred=-0.72 <= 0.95
    sm.update(_det(norm_pos=-1.6), 0.0)
    d_early = sm.update(_det(norm_pos=-1.2), dt)
    assert d_early.should_click is True
    assert d_early.reason == "predicted_gold_entry"
    assert d_early.predicted_norm_pos == pytest.approx(-0.72, abs=1e-3)

    # Far outside Gold: -2.5 -> -2.0 => v=+30/s, clamped lead=+0.5, pred=-1.5 -> must NOT click
    sm.reset()
    sm.update(_det(norm_pos=-2.5), 0.0)
    d_far = sm.update(_det(norm_pos=-2.0), dt)
    assert d_far.should_click is False
    assert d_far.predicted_norm_pos == pytest.approx(-1.5, abs=1e-3)

    # Moving away from Gold: -1.05 -> -1.2 => v=-9/s, pred=-1.38 -> must NOT click
    sm.reset()
    sm.update(_det(norm_pos=-1.05), 0.0)
    d_away = sm.update(_det(norm_pos=-1.2), dt)
    assert d_away.should_click is False
    assert d_away.predicted_norm_pos < -1.2

    # Fast pass where pred overshoots > +0.95 still triggers via raw_pos <= 0.95
    sm.reset()
    sm.update(_det(norm_pos=-1.4), 0.0)
    d_fast = sm.update(_det(norm_pos=0.8), dt)
    assert d_fast.should_click is True
    assert d_fast.reason == "entered_gold"
    assert d_fast.predicted_norm_pos > 0.95

    # Gold-zone jump (length re-roll) resets velocity to 0.0 (no false predicted click)
    sm.reset()
    sm.update(_det(norm_pos=-2.0, gold_span=(400.0, 440.0)), 0.0)
    d_jump = sm.update(_det(norm_pos=-1.1, gold_span=(320.0, 340.0)), dt)
    assert d_jump.velocity_norm_per_s == 0.0
    assert d_jump.predicted_norm_pos == pytest.approx(-1.1)
    assert d_jump.should_click is False

    # A pure screen translation of a still gauge (camera/avatar sway) must NOT
    # reset the velocity: only the Gold length may signal a new strike.
    sm.reset()
    sm.update(_det(norm_pos=-3.0, gold_span=(400.0, 440.0)), 0.0)
    d_sway = sm.update(_det(norm_pos=-2.5, gold_span=(480.0, 520.0)), dt)
    assert d_sway.velocity_norm_per_s != 0.0
    assert d_sway.should_click is False

    # Tilt-estimate changes shift the absolute projection by tens of pixels even
    # when the physical gauge is still; they must not count as a strike jump either.
    sm.reset()
    sm.update(_det(norm_pos=-3.0, gold_span=(322.9, 340.3)), 0.0)
    d_tilt = sm.update(_det(norm_pos=-2.5, gold_span=(370.5, 388.0)), dt)
    assert d_tilt.velocity_norm_per_s != 0.0
    assert d_tilt.should_click is False

    # Bounce direction reversal snaps velocity sign immediately
    sm.reset()
    sm.update(_det(norm_pos=-2.0), 0.0)
    sm.update(_det(norm_pos=-1.5), dt)  # +30/s
    d_rev = sm.update(_det(norm_pos=-2.0), 2 * dt)  # -30/s immediately
    assert d_rev.velocity_norm_per_s == pytest.approx(-30.0, abs=1e-6)


def test_release_delay_centres_the_bar_at_the_configured_judgement_time() -> None:
    # The game judges on pointer-up; release_delay_s should place the release
    # so the extrapolated bar position is 0 at now + lead_ms.
    cfg = TriggerConfig(lead_ms=16.7, max_lead_offset=2.0, entry_margin=0.95)
    sm = TriggerStateMachine(cfg)
    dt = 1.0 / 60.0
    sm.update(_det(norm_pos=-2.0), 1.0)
    dec = sm.update(_det(norm_pos=-1.2), 1.0 + dt)
    assert dec.should_click is True
    assert dec.reason == "predicted_gold_entry"
    assert dec.predicted_norm_pos == pytest.approx(-0.3984, abs=1e-3)
    assert dec.release_delay_s == pytest.approx(0.3984 / 48.0, abs=1e-3)

    # A raw in-Gold sample also gets a delay that moves the release to the centre.
    sm.reset()
    sm.update(_det(norm_pos=1.5), 2.0)
    dec2 = sm.update(_det(norm_pos=0.9), 2.0 + dt)
    assert dec2.should_click is True
    assert dec2.reason == "entered_gold"
    assert dec2.release_delay_s == pytest.approx(0.3 / 36.0, abs=1e-3)


def test_lost_frames_and_missing_red_handling() -> None:
    sm = TriggerStateMachine(TriggerConfig(rearm_lost_frames=3, cooldown_ms=200.0))
    assert sm.update(_det(norm_pos=-1.8), -0.02).should_click is False
    assert sm.update(_det(norm_pos=0.0), 0.0).should_click is True
    assert sm.state == TriggerState.LATCHED

    # 2 missing frames (< rearm_lost_frames=3) preserve LATCHED
    d1 = sm.update(_det(present=False), 0.02)
    d2 = sm.update(_det(present=False), 0.04)
    assert d1.state == TriggerState.LATCHED
    assert d2.state == TriggerState.LATCHED
    assert sm.lost_frames == 2

    # Reappearing in Gold does NOT double-click
    d_back = sm.update(_det(norm_pos=0.0), 0.06)
    assert d_back.state == TriggerState.LATCHED
    assert d_back.should_click is False
    assert sm.lost_frames == 0

    # 3 consecutive missing frames -> IDLE and clears last_trigger_s
    sm.update(_det(present=False), 0.08)
    sm.update(_det(present=False), 0.10)
    d3 = sm.update(_det(present=False), 0.12)
    assert d3.state == TriggerState.IDLE
    assert d3.transitions == (StateTransition(TriggerState.LATCHED, TriggerState.IDLE, "gauge_lost_rearm"),)
    assert sm.last_trigger_s is None

    # Direct norm_pos=0.0 after full lost-rearm must NOT click until armed outside Gold
    d_direct = sm.update(_det(norm_pos=0.0), 0.14)
    assert d_direct.state == TriggerState.IDLE
    assert d_direct.should_click is False
    assert sm.trigger_count == 1

    # Arm outside Gold, then enter Gold -> clicks cleanly
    d_arm = sm.update(_det(norm_pos=-1.8), 0.15)
    assert d_arm.state == TriggerState.ARMED
    assert d_arm.should_click is False

    d_re = sm.update(_det(norm_pos=0.0), 0.16)
    assert d_re.should_click is True
    assert d_re.state == TriggerState.LATCHED
    assert sm.trigger_count == 2

    # norm_pos=None and NaN while LATCHED stay LATCHED safely
    d_none = sm.update(_det(present=True, norm_pos=None), 0.16)
    assert d_none.state == TriggerState.LATCHED
    assert d_none.should_click is False
    assert d_none.velocity_norm_per_s == 0.0

    d_nan = sm.update(_det(present=True, norm_pos=float("nan")), 0.18)
    assert d_nan.state == TriggerState.LATCHED
    assert d_nan.should_click is False


def test_timestamp_validation_and_zero_cooldown() -> None:
    sm = TriggerStateMachine(TriggerConfig(cooldown_ms=0.0))

    for bad_t in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError):
            sm.update(_det(norm_pos=0.0), bad_t)  # type: ignore[arg-type]

    # Arm outside Gold, then trigger at t=1.0 -> LATCHED
    assert sm.update(_det(norm_pos=-1.5), 0.98).should_click is False
    assert sm.update(_det(norm_pos=0.0), 1.0).should_click is True
    # Duplicate and backward timestamps: no ZeroDivisionError, velocity 0.0
    d_dup = sm.update(_det(norm_pos=0.5), 1.0)
    assert d_dup.velocity_norm_per_s == 0.0
    d_back = sm.update(_det(norm_pos=0.6), 0.99)
    assert d_back.velocity_norm_per_s == 0.0

    # With cooldown_ms=0.0, exiting Gold goes straight to ARMED
    d_exit = sm.update(_det(norm_pos=1.4), 1.02)
    assert d_exit.state == TriggerState.ARMED
    assert d_exit.cooldown_remaining_ms == 0.0
    assert d_exit.should_click is False


def test_ieee754_exact_cooldown_boundary_and_edge_configs() -> None:
    # 1.0 -> 1.2 with cooldown_ms=200.0 suffers (1.2-1.0)*1000 == 199.99999999999994 without eps
    sm = TriggerStateMachine(TriggerConfig(cooldown_ms=200.0, entry_margin=0.95, exit_margin=1.25))
    assert sm.update(_det(norm_pos=-1.8), 0.98).state == TriggerState.ARMED
    assert sm.update(_det(norm_pos=0.0), 1.0).should_click is True
    assert sm.update(_det(norm_pos=1.5), 1.05).state == TriggerState.COOLDOWN

    # Outside Gold at exact t=1.2 -> expires to ARMED with 0.0 cooldown remaining
    d_boundary_arm = sm.update(_det(norm_pos=1.5), 1.2)
    assert d_boundary_arm.previous_state == TriggerState.COOLDOWN
    assert d_boundary_arm.state == TriggerState.ARMED
    assert d_boundary_arm.cooldown_remaining_ms == 0.0
    assert d_boundary_arm.should_click is False

    # And single-frame fall-through inside Gold at exact t=1.2 -> triggers immediately
    sm.reset()
    assert sm.update(_det(norm_pos=-1.8), 0.98).state == TriggerState.ARMED
    assert sm.update(_det(norm_pos=0.0), 1.0).should_click is True
    assert sm.update(_det(norm_pos=1.5), 1.05).state == TriggerState.COOLDOWN
    d_boundary = sm.update(_det(norm_pos=0.0), 1.2)
    assert d_boundary.previous_state == TriggerState.COOLDOWN
    assert d_boundary.state == TriggerState.LATCHED
    assert d_boundary.should_click is True
    assert d_boundary.cooldown_remaining_ms == pytest.approx(200.0)

    # norm_pos=None across COOLDOWN -> ARMED expiry transitions without clicking
    sm.reset()
    assert sm.update(_det(norm_pos=-1.8), 0.98).state == TriggerState.ARMED
    assert sm.update(_det(norm_pos=0.0), 1.0).should_click is True
    assert sm.update(_det(norm_pos=1.5), 1.05).state == TriggerState.COOLDOWN
    d_none_exp = sm.update(_det(present=True, norm_pos=None), 1.25)
    assert d_none_exp.previous_state == TriggerState.COOLDOWN
    assert d_none_exp.state == TriggerState.ARMED
    assert d_none_exp.should_click is False
    assert d_none_exp.effective_norm_pos is None

    # entry_margin == exit_margin and max_lead_offset == 0.0
    sm_tight = TriggerStateMachine(
        TriggerConfig(
            entry_margin=0.95,
            exit_margin=0.95,
            cooldown_ms=0.0,
            lead_ms=25.0,
            max_lead_offset=0.0,
        )
    )
    d_arm = sm_tight.update(_det(norm_pos=-1.2), 0.0)
    assert d_arm.state == TriggerState.ARMED
    assert d_arm.effective_norm_pos == pytest.approx(-1.2)
    d_wait = sm_tight.update(_det(norm_pos=-0.96), 1.0 / 60.0)
    assert d_wait.should_click is False
    assert d_wait.effective_norm_pos == pytest.approx(-0.96)
    d_trig = sm_tight.update(_det(norm_pos=0.95), 2.0 / 60.0)
    assert d_trig.should_click is True
    assert d_trig.effective_norm_pos == pytest.approx(0.95)
    d_exit = sm_tight.update(_det(norm_pos=0.96), 3.0 / 60.0)
    assert d_exit.state == TriggerState.ARMED
    assert d_exit.should_click is False

    # IDLE with lead_ms > 0 never fires on arming frame even if velocity would predict inside Gold
    sm_lead_idle = TriggerStateMachine(TriggerConfig(lead_ms=500.0, max_lead_offset=2.0, entry_margin=0.95))
    d_idle_lead = sm_lead_idle.update(_det(norm_pos=-1.0), 0.0)
    assert d_idle_lead.state == TriggerState.ARMED
    assert d_idle_lead.should_click is False
