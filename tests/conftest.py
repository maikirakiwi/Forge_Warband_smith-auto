from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from vrc_smith_auto.config import AppConfig

# Empirical BGR colors grounded in Survey 2 observations
SYNTH_BG_BGR = (12, 11, 18)
SYNTH_DARK_GRAY_BGR = (32, 34, 41)
SYNTH_SILVER_BGR = (184, 177, 174)
SYNTH_GOLD_BGR = (95, 207, 249)
SYNTH_RED_BGR = (74, 72, 239)

DEFAULT_UPLOADED_SCREENSHOT = Path(
    "C:/Users/wewe1/.gemini/antigravity/brain/0aa8bc95-c47c-47dc-90d8-42df58884912/"
    ".user_uploaded/media_1795156043546_6066a817.png"
)


@dataclass(frozen=True, slots=True)
class SyntheticGaugeFrame:
    """Synthetic rendered gauge frame with analytical ground truth."""

    image_bgr: np.ndarray
    expected_norm_pos: float | None
    expected_tilt_deg: float
    expected_axis_unit: tuple[float, float]
    expected_gold_center: tuple[float, float]
    gold_height_px: float
    gauge_present: bool


def _draw_oriented_rect(
    img: np.ndarray,
    center: tuple[float, float],
    along_u_len: float,
    along_v_len: float,
    u: tuple[float, float],
    v: tuple[float, float],
    color_bgr: tuple[int, int, int],
) -> None:
    hu = 0.5 * along_u_len
    hv = 0.5 * along_v_len
    cx, cy = center
    ux, uy = u
    vx, vy = v
    pts = np.array(
        [
            [cx - hu * ux - hv * vx, cy - hu * uy - hv * vy],
            [cx - hu * ux + hv * vx, cy - hu * uy + hv * vy],
            [cx + hu * ux + hv * vx, cy + hu * uy + hv * vy],
            [cx + hu * ux - hv * vx, cy + hu * uy - hv * vy],
        ],
        dtype=np.float64,
    )
    cv2.fillConvexPoly(img, np.rint(pts).astype(np.int32), color_bgr, lineType=cv2.LINE_8)


def render_synthetic_gauge(
    *,
    width: int = 1920,
    height: int = 1080,
    center: tuple[float, float] = (960.0, 540.0),
    tilt_deg: float = 0.0,
    scale: float = 1.0,
    gauge_length_px: float = 220.0,
    gauge_width_px: float = 38.0,
    gold_height_px: float = 28.0,
    gold_offset_px: float = 0.0,
    red_norm_pos: float | None = 0.0,
    red_width_px: float = 62.0,
    red_thickness_px: float = 7.0,
    gauge_present: bool = True,
    include_distractors: bool = False,
) -> SyntheticGaugeFrame:
    """Render an anti-alias-free synthetic timing gauge with exact geometric truth."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = SYNTH_BG_BGR

    theta = math.radians(tilt_deg)
    u = (math.sin(theta), math.cos(theta))
    v = (math.cos(theta), -math.sin(theta))

    gold_center = (
        center[0] + gold_offset_px * scale * u[0],
        center[1] + gold_offset_px * scale * u[1],
    )
    scaled_gold_h = float(gold_height_px * scale)

    if gauge_present:
        _draw_oriented_rect(
            img,
            center=center,
            along_u_len=(gauge_length_px + 44.0) * scale,
            along_v_len=gauge_width_px * scale,
            u=u,
            v=v,
            color_bgr=SYNTH_DARK_GRAY_BGR,
        )
        _draw_oriented_rect(
            img,
            center=center,
            along_u_len=gauge_length_px * scale,
            along_v_len=gauge_width_px * scale,
            u=u,
            v=v,
            color_bgr=SYNTH_SILVER_BGR,
        )
        _draw_oriented_rect(
            img,
            center=gold_center,
            along_u_len=scaled_gold_h,
            along_v_len=gauge_width_px * scale,
            u=u,
            v=v,
            color_bgr=SYNTH_GOLD_BGR,
        )
        if red_norm_pos is not None:
            half_len = 0.5 * scaled_gold_h
            red_center = (
                gold_center[0] + red_norm_pos * half_len * u[0],
                gold_center[1] + red_norm_pos * half_len * u[1],
            )
            _draw_oriented_rect(
                img,
                center=red_center,
                along_u_len=red_thickness_px * scale,
                along_v_len=red_width_px * scale,
                u=u,
                v=v,
                color_bgr=SYNTH_RED_BGR,
            )

    if include_distractors:
        # Off-gauge Gold UI icon (e.g., skill emblem without Silver neighbors)
        cv2.rectangle(img, (60, 60), (96, 96), SYNTH_GOLD_BGR, thickness=-1)
        # Off-gauge Red status pill (e.g., 0% badge)
        cv2.rectangle(img, (60, max(120, height - 120)), (104, max(132, height - 106)), SYNTH_RED_BGR, thickness=-1)

    return SyntheticGaugeFrame(
        image_bgr=img,
        expected_norm_pos=float(red_norm_pos) if (gauge_present and red_norm_pos is not None) else None,
        expected_tilt_deg=float(tilt_deg),
        expected_axis_unit=u,
        expected_gold_center=gold_center,
        gold_height_px=scaled_gold_h,
        gauge_present=gauge_present,
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def reference_video_path(repo_root: Path) -> Path:
    return repo_root / "example" / "2026-11-20 15-32-46.mp4"


@pytest.fixture(scope="session")
def reference_screenshot_path() -> Path | None:
    env_override = os.environ.get("VRC_SMITH_TEST_SCREENSHOT")
    if env_override:
        p = Path(env_override)
        return p if p.is_file() else None
    return DEFAULT_UPLOADED_SCREENSHOT if DEFAULT_UPLOADED_SCREENSHOT.is_file() else None


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def synthetic_gauge_factory() -> Callable[..., SyntheticGaugeFrame]:
    return render_synthetic_gauge
