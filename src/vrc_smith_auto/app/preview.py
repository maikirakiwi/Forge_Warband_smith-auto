from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vrc_smith_auto.control.state_machine import TriggerDecision, TriggerState
from vrc_smith_auto.vision.models import DetectionResult, OrientedBox

_COLOR_GOLD = (0, 215, 255)
_COLOR_RED = (0, 0, 255)
_COLOR_AXIS = (255, 255, 0)
_COLOR_ROI = (180, 180, 180)
_COLOR_TEXT = (245, 245, 245)
_COLOR_BG = (20, 20, 20)
_COLOR_PAUSED = (0, 120, 255)
_COLOR_CLICK = (0, 255, 120)


def _draw_oriented_box(
    canvas: np.ndarray,
    box: OrientedBox,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    rect = (
        (float(box.center[0]), float(box.center[1])),
        (float(box.size[0]), float(box.size[1])),
        float(box.angle_deg),
    )
    pts = np.int32(np.round(cv2.boxPoints(rect)))
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _resolve_state_text(
    state: str | TriggerState | TriggerDecision | None,
) -> tuple[str | None, bool]:
    if isinstance(state, TriggerDecision):
        return state.state.value, bool(state.should_click)
    if isinstance(state, TriggerState):
        return state.value, False
    if state is not None:
        return str(state), False
    return None, False


def draw_overlay(
    frame_bgr: np.ndarray,
    detection: DetectionResult | None = None,
    *,
    state: str | TriggerState | TriggerDecision | None = None,
    paused: bool = False,
    fps: float | None = None,
) -> np.ndarray:
    """Return a new annotated BGR uint8 frame with gauge geometry and state HUD."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("frame_bgr must be a non-empty numpy.ndarray")

    if frame_bgr.ndim == 2:
        canvas = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR))
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        canvas = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR))
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        canvas = np.ascontiguousarray(frame_bgr[:, :, :3].copy())
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        canvas = np.ascontiguousarray(frame_bgr.copy())
    else:
        raise ValueError(f"Unsupported frame_bgr shape: {frame_bgr.shape}")

    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8, copy=False)

    h, w = int(canvas.shape[0]), int(canvas.shape[1])

    if detection is not None:
        if detection.roi_used is not None:
            rx, ry, rw, rh = (int(v) for v in detection.roi_used)
            if rw > 0 and rh > 0:
                x0 = max(0, rx)
                y0 = max(0, ry)
                x1 = min(w - 1, rx + rw - 1)
                y1 = min(h - 1, ry + rh - 1)
                if x1 > x0 and y1 > y0:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), _COLOR_ROI, 1, cv2.LINE_AA)

        if detection.gauge_present:
            if detection.gold_box is not None:
                _draw_oriented_box(canvas, detection.gold_box, _COLOR_GOLD, thickness=2)

            if detection.red_box is not None:
                _draw_oriented_box(canvas, detection.red_box, _COLOR_RED, thickness=2)

            if detection.axis_unit is not None and detection.gold_box is not None:
                ux, uy = float(detection.axis_unit[0]), float(detection.axis_unit[1])
                cx, cy = float(detection.gold_box.center[0]), float(detection.gold_box.center[1])
                span = max(35.0, float(detection.gold_box.size[1]) * 1.4)
                p0 = (int(round(cx - ux * span)), int(round(cy - uy * span)))
                p1 = (int(round(cx + ux * span)), int(round(cy + uy * span)))
                cv2.arrowedLine(
                    canvas,
                    p0,
                    p1,
                    _COLOR_AXIS,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                    tipLength=0.12,
                )

    state_label, should_click = _resolve_state_text(state)
    lines: list[tuple[str, tuple[int, int, int]]] = []

    if paused:
        lines.append(("PAUSED (F8 resume / Esc stop)", _COLOR_PAUSED))
    elif state_label is not None:
        suffix = " [CLICK]" if should_click else ""
        color = _COLOR_CLICK if should_click else _COLOR_TEXT
        lines.append((f"State: {state_label}{suffix}", color))
    else:
        lines.append(("State: ACTIVE", _COLOR_TEXT))

    if detection is not None and detection.gauge_present:
        norm_txt = f"{detection.norm_pos:+.3f}" if detection.norm_pos is not None else "N/A"
        tilt_txt = f"{detection.tilt_deg:+.1f}deg" if detection.tilt_deg is not None else "N/A"
        lines.append((f"norm={norm_txt}  tilt={tilt_txt}  conf={detection.confidence:.2f}", _COLOR_TEXT))
    else:
        lines.append(("Gauge: not detected", (160, 160, 160)))

    if fps is not None:
        lines.append((f"FPS: {fps:.1f}", _COLOR_TEXT))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50 if w < 640 else 0.60
    thickness = 1 if w < 640 else 2
    line_step = 20 if w < 640 else 24

    box_w = min(w - 16, 390 if w >= 640 else max(180, w - 16))
    box_h = 12 + len(lines) * line_step
    if box_w > 20 and box_h < h - 8:
        cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), _COLOR_BG, -1)
        cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (80, 80, 80), 1)

    for idx, (text, color) in enumerate(lines):
        y = 26 + idx * line_step
        if y < h - 4:
            cv2.putText(canvas, text, (16, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas


class PreviewWindow:
    """Lazy OpenCV HighGUI preview window with clean resource cleanup."""

    def __init__(
        self,
        window_name: str = "vrc-smith-auto - Preview",
        *,
        wait_ms: int = 1,
    ) -> None:
        self._window_name = str(window_name)
        self._wait_ms = max(1, int(wait_ms))
        self._opened = False
        self._closed = False

    @property
    def window_name(self) -> str:
        return self._window_name

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detection: DetectionResult | None = None,
        *,
        state: str | TriggerState | TriggerDecision | None = None,
        paused: bool = False,
        fps: float | None = None,
    ) -> np.ndarray:
        return draw_overlay(frame_bgr, detection, state=state, paused=paused, fps=fps)

    def show(
        self,
        frame_bgr: np.ndarray,
        detection: DetectionResult | None = None,
        *,
        state: str | TriggerState | TriggerDecision | None = None,
        paused: bool = False,
        fps: float | None = None,
    ) -> bool:
        """Render overlay and display it. Returns False if user closed window or pressed 'q'."""
        if self._closed:
            return False

        annotated = self.annotate(frame_bgr, detection, state=state, paused=paused, fps=fps)
        if not self._opened:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._opened = True

        cv2.imshow(self._window_name, annotated)
        key = cv2.waitKey(self._wait_ms) & 0xFF
        if key in (ord("q"), ord("Q")):
            self.close()
            return False

        try:
            visible = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                self.close()
                return False
        except cv2.error:
            self.close()
            return False

        return True

    def close(self) -> None:
        if self._opened and not self._closed:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass
        self._opened = False
        self._closed = True

    def __enter__(self) -> PreviewWindow:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
