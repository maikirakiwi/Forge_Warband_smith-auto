from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import (
    SYNTH_BG_BGR,
    SYNTH_GOLD_BGR,
    SYNTH_RED_BGR,
    SYNTH_SILVER_BGR,
    render_synthetic_gauge,
)
from vrc_smith_auto.config import AppConfig
from vrc_smith_auto.control import TriggerStateMachine
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.detector import GaugeDetector
from vrc_smith_auto.vision.models import DetectionResult


@pytest.mark.parametrize(
    "width,height",
    [(640, 480), (1280, 720), (1920, 1080), (2560, 1440)],
)
@pytest.mark.parametrize("scale", [0.6, 1.0, 1.8])
def test_resolution_and_scale_matrix_red_split_bridging(
    width: int,
    height: int,
    scale: float,
) -> None:
    cfg = AppConfig()
    for gold_h in (16.0, 30.0):
        sample = render_synthetic_gauge(
            width=width,
            height=height,
            center=(width * 0.5, height * 0.5),
            scale=scale,
            gold_height_px=gold_h,
            red_norm_pos=0.0,
            red_thickness_px=9.0,
            include_distractors=True,
        )
        det = GaugeDetector(cfg)
        res = det.detect(sample.image_bgr)
        assert res.gauge_present is True, f"Failed at {width}x{height} scale={scale} gold_h={gold_h}"
        assert res.gold_box is not None
        assert res.red_box is not None
        assert res.norm_pos is not None
        assert abs(res.norm_pos) <= 0.25
        assert res.confidence > 0.5


def test_720p_unscaled_center_and_shifted_gauges() -> None:
    cfg = AppConfig()
    for cx, cy in ((640.0, 360.0), (320.0, 220.0), (960.0, 500.0)):
        sample = render_synthetic_gauge(
            width=1280,
            height=720,
            center=(cx, cy),
            scale=1.0,
            gold_height_px=34.0,
            red_norm_pos=0.0,
            red_thickness_px=7.0,
        )
        res = GaugeDetector(cfg).detect(sample.image_bgr)
        assert res.gauge_present is True
        assert res.red_box is not None
        assert res.norm_pos is not None
        assert abs(res.norm_pos) <= 0.25


def test_lazy_red_guided_pair_fallback_bridges_thick_red_split() -> None:
    # Thick 17 px Red bar at 1080p exceeds k_sz=15 morph bridge and triggers pair fallback
    sample = render_synthetic_gauge(
        width=1920,
        height=1080,
        center=(850.0, 410.0),
        scale=1.0,
        gold_height_px=34.0,
        red_norm_pos=0.0,
        red_thickness_px=17.0,
        red_width_px=64.0,
        include_distractors=True,
    )
    res = GaugeDetector(AppConfig()).detect(sample.image_bgr)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.red_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.25


@pytest.mark.parametrize("gold_h", [12.0, 14.0, 16.0])
@pytest.mark.parametrize("red_norm", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("red_thick", [7.0, 9.0])
def test_narrow_gold_cold_start_edge_overlapping_red_detected(
    gold_h: float,
    red_norm: float,
    red_thick: float,
) -> None:
    sample = render_synthetic_gauge(
        width=1920,
        height=1080,
        center=(850.0, 410.0),
        gold_height_px=gold_h,
        red_norm_pos=red_norm,
        red_thickness_px=red_thick,
    )
    res = GaugeDetector(AppConfig()).detect(sample.image_bgr)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.red_box is not None
    assert res.norm_pos is not None


def test_edge_anchor_recovery_preserves_gold_length_under_edge_occlusion() -> None:
    det = GaugeDetector(AppConfig())
    # Frame 1: unoccluded red outside Gold -> establishes 2D edge anchor
    f1 = render_synthetic_gauge(gold_height_px=24.0, red_norm_pos=-2.0, red_thickness_px=7.0)
    r1 = det.detect(f1.image_bgr)
    assert r1.gauge_present is True and r1.gold_box is not None
    assert r1.gold_box.size[1] == pytest.approx(25.0, abs=2.0)

    # Frame 2: red overlaps top edge of Gold -> 2D edge anchor recovers occluded Gold edge
    f2 = render_synthetic_gauge(gold_height_px=24.0, red_norm_pos=-1.0, red_thickness_px=7.0)
    r2 = det.detect(f2.image_bgr)
    assert r2.gauge_present is True and r2.gold_box is not None
    assert r2.red_box is not None and r2.norm_pos is not None
    assert r2.gold_box.size[1] >= 22.0
    assert r2.norm_pos == pytest.approx(-1.0, abs=0.25)


def test_warm_locked_narrow_gold_edge_overlap_matrix() -> None:
    cfg = AppConfig()
    for gold_h in (12.0, 14.0, 16.0):
        for red_thick in (5.0, 7.0, 9.0):
            for red_norm in (-1.0, -0.8, -0.5, 0.0, 0.5, 0.8, 1.0):
                det = GaugeDetector(cfg)
                warm = render_synthetic_gauge(
                    gold_height_px=gold_h,
                    red_norm_pos=-2.5,
                    red_thickness_px=red_thick,
                )
                rw = det.detect(warm.image_bgr)
                assert rw.gauge_present is True

                probe = render_synthetic_gauge(
                    gold_height_px=gold_h,
                    red_norm_pos=red_norm,
                    red_thickness_px=red_thick,
                )
                res = det.detect(probe.image_bgr)
                assert res.gauge_present is True, f"Dropped gl={gold_h} rt={red_thick} rn={red_norm}"
                assert res.red_box is not None
                assert res.norm_pos is not None
                assert abs(res.norm_pos - red_norm) <= 0.35, (
                    f"Inaccurate norm_pos={res.norm_pos} for gl={gold_h} rt={red_thick} rn={red_norm}"
                )


@pytest.mark.parametrize(
    "gold_h,red_thick,speed,phase_frac",
    [
        (12.0, 5.0, 6.0, 0.5),
        (12.0, 9.0, 3.0, 0.0),
        (12.0, 9.0, 3.0, 0.5),
        (14.0, 9.0, 5.0, 0.5),
        (16.0, 9.0, 6.0, 0.5),
    ],
)
def test_continuous_60fps_narrow_gold_sweep_triggers_single_click(
    gold_h: float,
    red_thick: float,
    speed: float,
    phase_frac: float,
) -> None:
    cfg = AppConfig()
    det = GaugeDetector(cfg)
    sm = TriggerStateMachine(cfg)
    clicks: list[int] = []

    pos = -80.0 + phase_frac * speed
    idx = 0
    while pos <= 80.0:
        rn = pos / (0.5 * gold_h)
        frame = render_synthetic_gauge(
            gold_height_px=gold_h,
            red_norm_pos=rn,
            red_thickness_px=red_thick,
        )
        res = det.detect(frame.image_bgr)
        dec = sm.update(res, idx / 60.0)
        if dec.should_click:
            clicks.append(idx)
        pos += speed
        idx += 1

    assert len(clicks) == 1, f"Expected 1 click for gl={gold_h} rt={red_thick} speed={speed}, got {clicks}"


@pytest.mark.parametrize("dx", [55.0, 80.0, 180.0, 220.0])
@pytest.mark.parametrize("red_norm", [-0.8, 0.8])
def test_lateral_pan_preserves_ref_gold_len_under_edge_occlusion(
    dx: float,
    red_norm: float,
) -> None:
    det = GaugeDetector(AppConfig())
    f0 = render_synthetic_gauge(
        center=(500.0, 400.0),
        gold_height_px=28.0,
        red_norm_pos=-2.5,
        red_thickness_px=7.0,
    )
    r0 = det.detect(f0.image_bgr)
    assert r0.gauge_present is True

    f1 = render_synthetic_gauge(
        center=(500.0 + dx, 400.0),
        gold_height_px=28.0,
        red_norm_pos=red_norm,
        red_thickness_px=7.0,
    )
    r1 = det.detect(f1.image_bgr)
    assert r1.gauge_present is True
    assert r1.red_box is not None
    assert r1.norm_pos is not None
    assert abs(r1.norm_pos - red_norm) <= 0.25


def test_one_sided_gold_silver_distractors_rejected() -> None:
    cfg = AppConfig()
    # One-sided narrow Gold + Silver bar without opposite Silver band (with and without non-bridging Red)
    for with_red in (False, True):
        img = np.full((1080, 1920, 3), SYNTH_BG_BGR, dtype=np.uint8)
        cx, cy = 960, 540
        cv2.rectangle(img, (cx - 19, cy), (cx + 19, cy + 110), SYNTH_SILVER_BGR, thickness=-1)
        cv2.rectangle(img, (cx - 19, cy - 6), (cx + 19, cy), SYNTH_GOLD_BGR, thickness=-1)
        if with_red:
            # Red bar on top edge with black background beyond it (no opposite Silver band)
            cv2.rectangle(img, (cx - 31, cy - 15), (cx + 31, cy - 6), SYNTH_RED_BGR, thickness=-1)
        res = GaugeDetector(cfg).detect(img)
        assert res.gauge_present is False


@pytest.mark.parametrize("tilt_deg", [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0])
def test_tilt_invariance_across_45_degrees(tilt_deg: float) -> None:
    cfg = AppConfig()
    for red_norm in (0.0, -1.5):
        sample = render_synthetic_gauge(
            width=1920,
            height=1080,
            center=(960.0, 540.0),
            tilt_deg=tilt_deg,
            gold_height_px=28.0,
            red_norm_pos=red_norm,
            include_distractors=True,
        )
        res = GaugeDetector(cfg).detect(sample.image_bgr)
        assert res.gauge_present is True
        assert res.axis_unit is not None
        ux, uy = res.axis_unit
        assert uy >= 0.0
        assert math.hypot(ux, uy) == pytest.approx(1.0, abs=1e-5)
        assert res.tilt_deg is not None
        assert abs(res.tilt_deg - tilt_deg) <= 1.0
        assert res.norm_pos is not None
        tol = 0.10 if abs(red_norm) <= 1.0 else 0.20
        assert abs(res.norm_pos - red_norm) <= tol


def test_distractors_and_asymmetric_silver_rejected() -> None:
    cfg = AppConfig()
    no_gauge = render_synthetic_gauge(
        width=1920,
        height=1080,
        gauge_present=False,
        include_distractors=True,
    )
    res_none = GaugeDetector(cfg).detect(no_gauge.image_bgr)
    assert res_none.gauge_present is False
    assert res_none.gold_box is None
    assert res_none.red_box is None
    assert res_none.norm_pos is None

    # Asymmetric Gold at extreme edge of colored track (lacks bilateral Silver) -> rejected
    asym = render_synthetic_gauge(
        width=1920,
        height=1080,
        gauge_length_px=150.0,
        gold_height_px=38.0,
        gold_offset_px=55.0,
        red_norm_pos=0.0,
    )
    res_asym = GaugeDetector(cfg).detect(asym.image_bgr)
    assert res_asym.gauge_present is False


def test_non_gold_zones_stay_outside_unit_norm_pos() -> None:
    cfg = AppConfig()
    for norm in (-1.6, 1.6, -2.5, 2.5):
        sample = render_synthetic_gauge(gold_height_px=28.0, red_norm_pos=norm)
        res = GaugeDetector(cfg).detect(sample.image_bgr)
        assert res.gauge_present is True
        assert res.norm_pos is not None
        assert abs(res.norm_pos) > 1.0


def test_tracking_lock_big_shift_lost_frames_and_update_state_flag() -> None:
    cfg = AppConfig().with_overrides(geometry={"max_lost_frames": 3})
    det = GaugeDetector(cfg)
    assert det.is_locked is False
    assert det.last_roi is None

    # Frame 1 acquires and locks
    f1 = render_synthetic_gauge(center=(500.0, 400.0))
    r1 = det.detect(f1.image_bgr)
    assert r1.gauge_present is True
    assert det.is_locked is True
    assert det.last_roi is not None
    assert det.lost_frames == 0

    # Small shift stays inside locked ROI
    f2 = render_synthetic_gauge(center=(540.0, 430.0))
    r2 = det.detect(f2.image_bgr)
    assert r2.gauge_present is True
    assert r2.roi_used is not None
    assert r2.roi_used[2] < 1920
    assert det.lost_frames == 0

    # update_state=False does not mutate lock/lost state
    blank = render_synthetic_gauge(gauge_present=False).image_bgr
    r_ro = det.detect(blank, update_state=False)
    assert r_ro.gauge_present is False
    assert det.is_locked is True
    assert det.lost_frames == 0

    # Big shift (> pad) triggers full-frame re-acquisition cleanly
    f3 = render_synthetic_gauge(center=(1450.0, 720.0))
    r3 = det.detect(f3.image_bgr)
    assert r3.gauge_present is True and r3.gold_box is not None
    assert abs(r3.gold_box.center[0] - 1450.0) <= 2.0
    assert det.lost_frames == 0

    # max_lost_frames=3 clears lock on 3rd absent frame
    for expected_lost in (1, 2):
        r_miss = det.detect(blank)
        assert r_miss.gauge_present is False
        assert det.is_locked is True
        assert det.lost_frames == expected_lost

    r_clear = det.detect(blank)
    assert r_clear.gauge_present is False
    assert det.is_locked is False
    assert det.last_roi is None

    # reset() clears state after lock
    det.detect_packet(FramePacket(image_bgr=f1.image_bgr, timestamp_s=0.1, frame_idx=0))
    assert det.is_locked is True
    det.reset()
    assert det.is_locked is False
    assert det.last_roi is None
    assert det.lost_frames == 0


def test_invalid_and_empty_inputs_return_empty() -> None:
    det = GaugeDetector(AppConfig())
    bad_inputs: list[object] = [
        None,
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((100, 100), dtype=np.uint8),
        np.zeros((100, 100, 4), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.float32),
        np.full((3, 3, 3), SYNTH_BG_BGR, dtype=np.uint8),
    ]
    for bad in bad_inputs:
        res = det.detect(bad)  # type: ignore[arg-type]
        assert res == DetectionResult.empty() or res.gauge_present is False

    # Pure solid Gold or solid Red image without gauge structure -> rejected
    solid_gold = np.full((200, 200, 3), SYNTH_GOLD_BGR, dtype=np.uint8)
    solid_red = np.full((200, 200, 3), SYNTH_RED_BGR, dtype=np.uint8)
    assert det.detect(solid_gold).gauge_present is False
    assert det.detect(solid_red).gauge_present is False


def test_real_screenshot_silver_zone(reference_screenshot_path: Path | None) -> None:
    if reference_screenshot_path is None:
        pytest.skip("Reference screenshot not available on disk")
    img = cv2.imread(str(reference_screenshot_path))
    assert img is not None
    res = GaugeDetector(AppConfig()).detect(img)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.red_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.0
    assert res.tilt_deg is not None and res.tilt_deg < 0.0


def test_real_video_accuracy_and_latency_budget(reference_video_path: Path) -> None:
    if not reference_video_path.is_file():
        pytest.skip("Reference video not available on disk")

    cap = cv2.VideoCapture(str(reference_video_path))
    assert cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 1570

    cfg = AppConfig()
    # Warm up OpenCV kernel/resize caches
    warmup = GaugeDetector(cfg)
    for idx in range(10):
        warmup.detect(frames[idx])

    det = GaugeDetector(cfg)
    sm = TriggerStateMachine(cfg)
    latencies_ms: list[float] = []
    clicks: list[int] = []
    inactive_fp = 0
    active_fn = 0
    active_no_red = 0

    def _is_active(i: int) -> bool:
        return (118 <= i <= 630) or (725 <= i <= 1338)

    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = det.detect(frame)
        dec = sm.update(res, idx / fps)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        active = _is_active(idx)
        if not active and res.gauge_present:
            inactive_fp += 1
        elif active and not res.gauge_present:
            active_fn += 1
        elif active and res.gauge_present and res.red_box is None:
            active_no_red += 1
        if dec.should_click:
            clicks.append(idx)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    p95_ms = float(np.percentile(arr, 95))

    assert inactive_fp == 0
    assert active_fn == 0
    assert active_no_red == 0
    assert len(clicks) == 22
    assert mean_ms < 12.0, f"mean latency {mean_ms:.2f} ms >= 12.0 ms"
    assert p95_ms < 16.7, f"p95 latency {p95_ms:.2f} ms >= 16.7 ms"
