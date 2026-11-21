from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OrientedBox:
    """Oriented 2D bounding box in full-frame pixel coordinates."""

    center: tuple[float, float]
    size: tuple[float, float]
    angle_deg: float

    @property
    def area(self) -> float:
        return float(self.size[0] * self.size[1])


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Output of GaugeDetector for a single frame."""

    gauge_present: bool
    gold_box: OrientedBox | None = None
    red_box: OrientedBox | None = None
    axis_unit: tuple[float, float] | None = None
    gold_span: tuple[float, float] | None = None
    red_proj: float | None = None
    norm_pos: float | None = None
    tilt_deg: float | None = None
    confidence: float = 0.0
    roi_used: tuple[int, int, int, int] | None = None

    @property
    def red_detected(self) -> bool:
        return self.red_box is not None and self.red_proj is not None

    @property
    def is_in_gold(self) -> bool:
        return (
            self.gauge_present
            and self.norm_pos is not None
            and abs(self.norm_pos) <= 1.0
        )

    @classmethod
    def empty(
        cls,
        roi_used: tuple[int, int, int, int] | None = None,
    ) -> DetectionResult:
        return cls(gauge_present=False, confidence=0.0, roi_used=roi_used)
