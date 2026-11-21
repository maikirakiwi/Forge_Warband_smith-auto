from __future__ import annotations

import ctypes
import inspect
from pathlib import Path
import sys
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# F1: Config Schema Boundaries & Validation (5 tests)
# ---------------------------------------------------------------------------


def test_f1_bva_invalid_capture_backend_rejected(e2e: Any) -> None:
    cfg_mod = e2e.import_pkg("vrc_smith_auto.config")
    with pytest.raises(ValueError):
        cfg_mod.CaptureConfig(backend="invalid_backend")
    with pytest.raises(ValueError):
        e2e.make_app_config(capture_backend="invalid_backend")


def test_f1_bva_inverted_hysteresis_margins_rejected(e2e: Any) -> None:
    cfg_mod = e2e.import_pkg("vrc_smith_auto.config")
    with pytest.raises(ValueError):
        cfg_mod.TriggerConfig(entry_margin=1.3, exit_margin=0.9)
    with pytest.raises(ValueError):
        cfg_mod.TriggerConfig(entry_margin=0.0, exit_margin=1.2)
    with pytest.raises(ValueError):
        e2e.make_app_config(entry_margin=1.3, exit_margin=0.9)


def test_f1_bva_non_positive_hold_or_negative_cooldown_rejected(e2e: Any) -> None:
    cfg_mod = e2e.import_pkg("vrc_smith_auto.config")
    with pytest.raises(ValueError):
        cfg_mod.TriggerConfig(hold_ms=0.0)
    with pytest.raises(ValueError):
        cfg_mod.TriggerConfig(cooldown_ms=-10.0)
    with pytest.raises(ValueError):
        e2e.make_app_config(hold_ms=0.0)


def test_f1_bva_missing_or_malformed_yaml_config_cli(e2e: Any, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    proc_missing = e2e.run_cli(["run", "--config", str(missing), "--dry-run"])
    assert proc_missing.returncode != 0

    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("trigger:\n  hold_ms: [unclosed\n", encoding="utf-8")
    proc_bad = e2e.run_cli(["run", "--config", str(bad_yaml), "--dry-run"])
    assert proc_bad.returncode != 0


def test_f1_bva_extreme_valid_config_roundtrip(e2e: Any, tmp_path: Path) -> None:
    cfg = e2e.make_app_config(
        hold_ms=1.0,
        cooldown_ms=0.0,
        entry_margin=0.05,
        exit_margin=2.0,
        lead_ms=15.0,
    )
    yaml_path = tmp_path / "extreme.yaml"
    yaml_path.write_text(cfg.to_yaml(), encoding="utf-8")
    reloaded = e2e.load_app_config_from_file(yaml_path)

    assert reloaded.hold_ms == pytest.approx(1.0)
    assert reloaded.cooldown_ms == pytest.approx(0.0)
    assert reloaded.entry_margin == pytest.approx(0.05)
    assert reloaded.exit_margin == pytest.approx(2.0)
    assert reloaded.lead_ms == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# F2: Core Protocols, Window & DPI Edge Cases (5 tests)
# ---------------------------------------------------------------------------


def test_f2_bva_nonexistent_vrchat_window_returns_none(e2e: Any) -> None:
    find_win = e2e.import_attr("vrc_smith_auto.window", "find_vrchat_window")
    assert find_win("__NoSuchVRChatWindow_XYZ__") is None
    assert find_win("__NoSuchVRChatWindow_XYZ__", class_name=None, process_name=None) is None


def test_f2_bva_dpi_awareness_init_idempotent(e2e: Any) -> None:
    enable_dpi = e2e.import_attr("vrc_smith_auto.window", "enable_dpi_awareness")
    results = [enable_dpi() for _ in range(3)]
    assert all(isinstance(r, bool) for r in results)
    if sys.platform == "win32":
        assert all(results)


def test_f2_bva_detection_result_blank_frame_contract(e2e: Any) -> None:
    _, np = e2e.import_cv2_np()
    blank = np.full((1080, 1920, 3), 24, dtype=np.uint8)
    det = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(det, blank)

    assert res.gauge_present is False
    assert res.gold_box is None
    assert res.norm_pos is None


def test_f2_bva_degenerate_tiny_frame_packet_safe(e2e: Any) -> None:
    _, np = e2e.import_cv2_np()
    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")
    det = e2e.make_gauge_detector(e2e.make_app_config())

    for size in (16, 1):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        pkt = FramePacket(image_bgr=arr, timestamp_s=0.0, frame_idx=0)
        res = e2e.detect_gauge(det, pkt)
        assert res.gauge_present is False
        assert res.norm_pos is None


def test_f2_bva_focus_guard_blocks_when_target_not_foreground(e2e: Any) -> None:
    win_mod = e2e.import_pkg("vrc_smith_auto.window")
    hwnd = win_mod.find_vrchat_window(
        "__NoSuchVRChat_FocusCheck__",
        class_name=None,
        process_name=None,
    )
    assert hwnd is None
    assert win_mod.is_foreground_window(0) is False
    assert win_mod.get_client_rect(0) is None


# ---------------------------------------------------------------------------
# F3: Capture & Offline Source Error Handling (5 tests)
# ---------------------------------------------------------------------------


def test_f3_bva_missing_image_file_raises_cleanly(e2e: Any, tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError, OSError, RuntimeError)):
        e2e.make_image_source(tmp_path / "missing_image.png")


def test_f3_bva_corrupt_video_file_raises_cleanly(e2e: Any, tmp_path: Path) -> None:
    zero_mp4 = tmp_path / "zero.mp4"
    zero_mp4.write_bytes(b"")
    text_mp4 = tmp_path / "not_a_video.mp4"
    text_mp4.write_text("definitely not an mp4 container", encoding="utf-8")

    for bad_path in (zero_mp4, text_mp4):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            src = e2e.make_video_source(bad_path)
            try:
                pkt = src.read()
                if pkt is None:
                    raise RuntimeError("Corrupt video returned None instead of raising")
            finally:
                src.close()


def test_f3_bva_image_source_single_frame_eof_and_double_close(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    src = e2e.make_image_source(reference_screenshot_path)
    pkt_1 = src.read()
    assert pkt_1 is not None
    assert pkt_1.frame_idx == 0

    pkt_2 = src.read()
    assert pkt_2 is None

    src.close()
    src.close()


def test_f3_bva_video_source_real_inactive_slice_boundaries(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    src = e2e.make_video_source(reference_video_path)
    try:
        packets = []
        for _ in range(118):
            pkt = src.read()
            assert pkt is not None
            packets.append(pkt)
        assert [p.frame_idx for p in packets] == list(range(118))
        assert packets[0].timestamp_s == pytest.approx(0.0, abs=1e-4)
        assert packets[-1].timestamp_s == pytest.approx(117.0 / 60.0, abs=0.02)
    finally:
        src.close()


def test_f3_bva_unsupported_live_backend_name_rejected(e2e: Any) -> None:
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        e2e.make_capture_backend("unknown_cap_backend")


# ---------------------------------------------------------------------------
# F4: Mouse Driver & Hotkey Edge Cases (5 tests)
# ---------------------------------------------------------------------------


def test_f4_bva_spy_mouse_hold_duration_exact_tick_boundary(e2e: Any) -> None:
    spy = e2e.make_spy_mouse()
    assert spy.press_left(30.0, timestamp_s=1.0) is True
    assert spy.is_pressed is True

    spy.tick(1.029)
    assert spy.is_pressed is True

    spy.tick(1.030)
    assert spy.is_pressed is False
    assert e2e.get_spy_click_count(spy) == 1


def test_f4_bva_spy_mouse_repress_while_held_ignored(e2e: Any) -> None:
    spy = e2e.make_spy_mouse()
    assert spy.press_left(30.0, timestamp_s=1.0) is True
    second = spy.press_left(30.0, timestamp_s=1.010)
    assert second is False
    spy.tick(1.035)
    assert spy.is_pressed is False
    assert e2e.get_spy_click_count(spy) == 1


def test_f4_bva_spy_mouse_release_all_clears_held_state(e2e: Any) -> None:
    spy = e2e.make_spy_mouse()
    assert spy.press_left(50.0, timestamp_s=2.0) is True
    assert spy.is_pressed is True
    spy.release_all()
    assert spy.is_pressed is False
    # Calling release_all again must be idempotent
    spy.release_all()
    assert spy.is_pressed is False


def test_f4_bva_win32_sendinput_struct_size_and_dry_run_isolation(e2e: Any) -> None:
    mod = e2e.import_pkg("vrc_smith_auto.input.win32_sendinput")
    # Collect all top-level INPUT ctypes.Structure classes without size pre-filtering
    excluded = {"MOUSEINPUT", "KEYBDINPUT", "HARDWAREINPUT"}
    input_classes: list[type[ctypes.Structure]] = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ctypes.Structure)
            and "INPUT" in name.upper()
            and name.upper() not in excluded
        ):
            input_classes.append(obj)
    assert input_classes, "Expected top-level INPUT ctypes.Structure in win32_sendinput"
    if sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) == 8:
        primary_input = getattr(mod, "INPUT", input_classes[0])
        assert ctypes.sizeof(primary_input) == 40

    spy = e2e.make_spy_mouse()
    assert spy.press_left(30.0, timestamp_s=1.0) is True
    spy.tick(1.040)
    assert e2e.get_spy_click_count(spy) == 1


def test_f4_bva_hotkey_edge_trigger_no_repeat_while_held(e2e: Any) -> None:
    hotkeys_mod = e2e.import_pkg("vrc_smith_auto.app.hotkeys")

    pressed_keys: set[int] = set()
    VK_F8 = 0x77
    VK_ESCAPE = 0x1B

    poller = None
    for cls_name in ("HotkeyController", "HotkeyPoller", "Hotkeys", "HotkeyListener", "HotkeyMonitor"):
        cls = getattr(hotkeys_mod, cls_name, None)
        if isinstance(cls, type):
            sig = inspect.signature(cls)
            kwargs: dict[str, Any] = {}
            if "key_state_fn" in sig.parameters:
                kwargs["key_state_fn"] = lambda vk: 0x8000 if vk in pressed_keys else 0
            elif "key_probe" in sig.parameters:
                kwargs["key_probe"] = lambda vk: vk in pressed_keys
            elif "is_key_down" in sig.parameters:
                kwargs["is_key_down"] = lambda vk: vk in pressed_keys
            elif "probe" in sig.parameters:
                kwargs["probe"] = lambda vk: vk in pressed_keys
            poller = cls(**kwargs)
            break

    assert poller is not None, "Expected public HotkeyController/HotkeyPoller class in vrc_smith_auto.app.hotkeys"
    poll_fn = getattr(poller, "poll", None) or getattr(poller, "update", None)
    assert callable(poll_fn)

    # Hold F8 across 5 polls -> toggle edge fires once only on rising edge
    pressed_keys.add(VK_F8)
    states = [poll_fn() for _ in range(5)]
    if hasattr(states[0], "toggle_armed"):
        edges = [bool(s.toggle_armed) for s in states]
        assert edges == [True, False, False, False, False]
    else:
        paused_vals = [getattr(s, "paused", getattr(poller, "paused", None)) for s in states]
        assert paused_vals[0] is True
        assert all(p is True for p in paused_vals)

    # Release F8, then press Esc -> emergency stop requested
    pressed_keys.clear()
    poll_fn()
    pressed_keys.add(VK_ESCAPE)
    esc_state = poll_fn()
    stop_flag = getattr(
        esc_state,
        "emergency_stop",
        getattr(
            esc_state,
            "stop",
            getattr(esc_state, "stop_requested", getattr(esc_state, "exit", getattr(poller, "stop_requested", False))),
        ),
    )
    assert stop_flag is True


# ---------------------------------------------------------------------------
# F5: Vision Detector Negative Zones, Tilts, Shifts, Scaling, Occlusion (10 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_f5_bva_non_gold_black_miss_zone_no_trigger(e2e: Any, side: float) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0, red_pos=side * 78.0)
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.5


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_f5_bva_non_gold_dark_gray_off_zone_no_trigger(e2e: Any, side: float) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0, red_pos=side * 58.0)
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.2


@pytest.mark.parametrize("norm_target", [-1.35, 1.35])
def test_f5_bva_non_gold_silver_good_zone_no_trigger(e2e: Any, norm_target: float) -> None:
    import synthetic_fixtures as sf

    base = sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0)
    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1920,
            height=1080,
            cx=850.0,
            cy=410.0,
            gold_len=30.0,
            red_pos=sf.red_pos_for_norm(base, norm_target),
        )
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.0


def test_f5_bva_no_gauge_blank_and_distractor_clutter_zero_detect(e2e: Any) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1920, height=1080, show_gauge=False, distractors=True, seed=42)
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is False


def test_f5_bva_real_video_inactive_frames_zero_detect(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    cv2, _ = e2e.import_cv2_np()
    cap = cv2.VideoCapture(str(reference_video_path))
    assert cap.isOpened()

    det = e2e.make_gauge_detector(e2e.make_app_config())
    sample_indices = [0, 30, 80, 106, 112, 117, 631, 675, 724, 1339, 1450, 1569]
    try:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            assert ok and frame is not None
            res = e2e.detect_gauge(det, frame)
            assert res.gauge_present is False, f"False gauge detection on inactive frame {idx}"
    finally:
        cap.release()


@pytest.mark.parametrize("tilt_deg", [-35.0, -15.0, -4.2, 4.2, 15.0, 35.0])
def test_f5_bva_extreme_tilt_angles_parametrized(e2e: Any, tilt_deg: float) -> None:
    import synthetic_fixtures as sf

    cfg = e2e.make_app_config()
    # Gold center
    gold_frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1920, height=1080, cx=960.0, cy=540.0, tilt_deg=tilt_deg, gold_len=30.0, red_pos=0.0)
    )
    res_gold = e2e.detect_gauge(e2e.make_gauge_detector(cfg), gold_frame)
    assert res_gold.gauge_present is True
    assert res_gold.norm_pos is not None
    assert abs(res_gold.norm_pos) <= 0.35

    # Silver zone
    base = sf.GaugeSpec(width=1920, height=1080, cx=960.0, cy=540.0, tilt_deg=tilt_deg, gold_len=30.0)
    silver_frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1920,
            height=1080,
            cx=960.0,
            cy=540.0,
            tilt_deg=tilt_deg,
            gold_len=30.0,
            red_pos=sf.red_pos_for_norm(base, -1.6),
        )
    )
    res_silver = e2e.detect_gauge(e2e.make_gauge_detector(cfg), silver_frame)
    assert res_silver.gauge_present is True
    assert res_silver.norm_pos is not None
    assert abs(res_silver.norm_pos) > 1.05


@pytest.mark.parametrize(
    "cx,cy",
    [(180.0, 180.0), (1740.0, 180.0), (180.0, 900.0), (1740.0, 900.0), (1400.0, 620.0)],
)
def test_f5_bva_extreme_position_shifts_1080p(e2e: Any, cx: float, cy: float) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1920, height=1080, cx=cx, cy=cy, gold_len=30.0, red_pos=0.0)
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.35


@pytest.mark.parametrize("width,height,scale", [(1280, 720, 720.0 / 1080.0), (1920, 1080, 1.0)])
def test_f5_bva_resolution_scaling_720p_and_1080p(
    e2e: Any,
    width: int,
    height: int,
    scale: float,
) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=width,
            height=height,
            cx=width * 0.5,
            cy=height * 0.5,
            scale=scale,
            gold_len=30.0,
            red_pos=0.0,
        )
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.35


@pytest.mark.parametrize("gold_len", [12.0, 14.0, 16.0])
def test_f5_bva_narrow_gold_strip_12_to_16px(e2e: Any, gold_len: float) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1920,
            height=1080,
            cx=850.0,
            cy=410.0,
            gold_len=gold_len,
            red_pos=0.0,
            red_thickness=5.0,
        )
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.50


def test_f5_bva_red_bar_bisecting_gold_occlusion_split(e2e: Any) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1920,
            height=1080,
            cx=850.0,
            cy=410.0,
            gold_len=20.0,
            red_thickness=9.0,
            red_pos=0.0,
            red_width=64.0,
        )
    )
    res = e2e.detect_gauge(e2e.make_gauge_detector(e2e.make_app_config()), frame)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.35


# ---------------------------------------------------------------------------
# F6: State Machine Boundary & Hysteresis Conditions (5 tests)
# ---------------------------------------------------------------------------


def _det_bva(e2e: Any, *, present: bool = True, norm_pos: float | None = None) -> Any:
    DetectionResult = e2e.import_attr("vrc_smith_auto.vision.models", "DetectionResult")
    OrientedBox = e2e.import_attr("vrc_smith_auto.vision.models", "OrientedBox")
    if not present:
        return DetectionResult.empty()
    box = OrientedBox(center=(500.0, 400.0), size=(38.0, 30.0), angle_deg=0.0)
    return DetectionResult(
        gauge_present=True,
        gold_box=box,
        red_box=box if norm_pos is not None else None,
        axis_unit=(0.0, 1.0),
        gold_span=(385.0, 415.0),
        red_proj=400.0 + (norm_pos * 15.0 if norm_pos is not None else 0.0),
        norm_pos=norm_pos,
        tilt_deg=0.0,
        confidence=0.95,
    )


def test_f6_bva_entry_margin_inside_vs_outside_boundary(e2e: Any) -> None:
    cfg = e2e.make_app_config(entry_margin=0.95, exit_margin=1.25, cooldown_ms=200.0)

    # Just outside entry_margin -> no click
    spy_out = e2e.make_spy_mouse()
    sm_out = e2e.make_trigger_state_machine(cfg, mouse=spy_out)
    e2e.step_state_machine(sm_out, _det_bva(e2e, norm_pos=-2.0), 1.000, mouse=spy_out)
    e2e.step_state_machine(sm_out, _det_bva(e2e, norm_pos=1.00), 1.016, mouse=spy_out)
    assert e2e.get_spy_click_count(spy_out) == 0

    # Just inside entry_margin -> 1 click
    spy_in = e2e.make_spy_mouse()
    sm_in = e2e.make_trigger_state_machine(cfg, mouse=spy_in)
    e2e.step_state_machine(sm_in, _det_bva(e2e, norm_pos=-2.0), 1.000, mouse=spy_in)
    e2e.step_state_machine(sm_in, _det_bva(e2e, norm_pos=0.93), 1.016, mouse=spy_in)
    assert e2e.get_spy_click_count(spy_in) == 1


def test_f6_bva_hysteresis_chatter_no_double_click(e2e: Any) -> None:
    cfg = e2e.make_app_config(entry_margin=0.95, exit_margin=1.25, cooldown_ms=200.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=-2.0), 1.000, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.85), 1.016, mouse=spy)
    # Chatter between 0.85 and 1.05 (< exit_margin=1.25), even past 200 ms
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=1.05), 1.250, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.85), 1.300, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1


def test_f6_bva_cooldown_blocks_fast_reentry_allows_after_expiry(e2e: Any) -> None:
    cfg = e2e.make_app_config(entry_margin=0.95, exit_margin=1.25, cooldown_ms=200.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=-1.8), 0.980, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.0), 1.000, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1

    # Exit Gold, re-enter at +100 ms (< 200 ms) -> blocked
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=1.6), 1.050, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.0), 1.100, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1

    # Exit Gold again, re-enter at +260 ms (> 200 ms) -> allowed
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=1.6), 1.160, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.0), 1.260, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 2


def test_f6_bva_lost_gauge_frames_resets_arming_safely(e2e: Any) -> None:
    cfg = e2e.make_app_config(rearm_lost_frames=3, cooldown_ms=200.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=-1.8), 1.000, mouse=spy)
    for i in range(10):
        e2e.step_state_machine(sm, _det_bva(e2e, present=False), 1.020 + i * 0.016, mouse=spy)
    # Entering Gold directly after lost frames must NOT click (proving ARMED -> IDLE reset)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.0), 1.200, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 0

    # Re-arming outside Gold and entering Gold recovers cleanly
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=-1.8), 1.220, mouse=spy)
    e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=0.0), 1.240, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1


def test_f6_bva_velocity_lead_zero_vs_clamped_distant_red(e2e: Any) -> None:
    for lead_ms in (0.0, 15.0):
        cfg = e2e.make_app_config(lead_ms=lead_ms, max_lead_offset=0.5)
        spy = e2e.make_spy_mouse()
        sm = e2e.make_trigger_state_machine(cfg, mouse=spy)
        for i, pos in enumerate([-3.2, -3.0, -2.8, -2.6]):
            e2e.step_state_machine(sm, _det_bva(e2e, norm_pos=pos), 1.000 + i * 0.016, mouse=spy)
        assert e2e.get_spy_click_count(spy) == 0


# ---------------------------------------------------------------------------
# F7: CLI & Benchmark Error Handling (5 tests)
# ---------------------------------------------------------------------------


def test_f7_bva_cli_verify_image_missing_file_nonzero(e2e: Any, tmp_path: Path) -> None:
    proc = e2e.run_cli(["verify-image", "--image", str(tmp_path / "missing.png"), "--json"])
    assert proc.returncode != 0


def test_f7_bva_cli_verify_video_missing_file_nonzero(e2e: Any, tmp_path: Path) -> None:
    proc = e2e.run_cli(["verify-video", "--video", str(tmp_path / "missing.mp4"), "--dry-run", "--json"])
    assert proc.returncode != 0


def test_f7_bva_cli_verify_image_expect_clicks_mismatch_fails(
    e2e: Any,
    reference_screenshot_path: Path,
    tmp_path: Path,
) -> None:
    import synthetic_fixtures as sf

    # Reference screenshot has Red in Silver -> 0 clicks; expecting 1 must fail
    proc_ref = e2e.run_cli(
        ["verify-image", "--image", str(reference_screenshot_path), "--expect-clicks", "1", "--json"]
    )
    assert proc_ref.returncode != 0

    # Synthetic Silver PNG -> 0 clicks; expecting 1 must fail
    base = sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, gold_len=34.0)
    silver_png = sf.write_png(
        tmp_path / "silver_mismatch.png",
        sf.render_gauge_frame(
            sf.GaugeSpec(
                width=1280,
                height=720,
                cx=640.0,
                cy=360.0,
                gold_len=34.0,
                red_pos=sf.red_pos_for_norm(base, -1.6),
            )
        ),
    )
    proc_silver = e2e.run_cli(
        ["verify-image", "--image", str(silver_png), "--expect-clicks", "1", "--json"]
    )
    assert proc_silver.returncode != 0


def test_f7_bva_cli_benchmark_unmet_threshold_fails_with_json(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0),
        [-60.0 + i * 6.0 for i in range(20)],
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="bench_fail.mp4")

    proc = e2e.run_cli(
        ["benchmark", "--video", str(mp4_path), "--iterations", "15", "--max-p95-ms", "0.001", "--json"]
    )
    assert proc.returncode != 0
    data = e2e.parse_cli_json(proc)
    assert "p95_ms" in data


def test_f7_bva_cli_invalid_subcommand_or_flag_nonzero(e2e: Any) -> None:
    proc_cmd = e2e.run_cli(["totally-unknown-subcommand"])
    assert proc_cmd.returncode != 0

    proc_backend = e2e.run_cli(["run", "--backend", "bogus_backend", "--dry-run"])
    assert proc_backend.returncode != 0


# ---------------------------------------------------------------------------
# F8: Documentation Completeness & Consistency (5 tests)
# ---------------------------------------------------------------------------


def test_f8_bva_tech_doc_required_sections_and_comparisons(e2e: Any) -> None:
    text = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8").lower()
    assert "dxcam" in text and "mss" in text and "gdi" in text
    assert "sendinput" in text
    assert "hsv" in text


def test_f8_bva_readme_hotkeys_and_cli_commands(e2e: Any) -> None:
    text = e2e.README_PATH.read_text(encoding="utf-8")
    assert "F8" in text
    assert "Esc" in text or "ESC" in text
    assert "--dry-run" in text
    for cmd in ("run", "verify-image", "verify-video", "benchmark"):
        assert cmd in text


def test_f8_bva_docs_no_unresolved_placeholders(e2e: Any) -> None:
    for doc_path in (e2e.README_PATH, e2e.TECH_DESIGN_PATH):
        text = doc_path.read_text(encoding="utf-8")
        for marker in ("TODO", "FIXME", "TBD"):
            assert marker not in text, f"Unresolved placeholder '{marker}' in {doc_path}"


def test_f8_bva_docs_cli_commands_match_live_help(e2e: Any) -> None:
    readme_text = e2e.README_PATH.read_text(encoding="utf-8")
    proc = e2e.run_cli(["--help"])
    assert proc.returncode == 0
    help_text = f"{proc.stdout}\n{proc.stderr}"
    for cmd in ("run", "verify-image", "verify-video", "benchmark"):
        assert cmd in readme_text and cmd in help_text


def test_f8_bva_docs_latency_budget_and_benchmark_cmd(e2e: Any) -> None:
    text = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8").lower()
    assert "benchmark" in text
    assert "33.3" in text or "16.7" in text


# ---------------------------------------------------------------------------
# F9: End-to-End Negative Sweeps & Failure-Path Contracts (5 tests)
# ---------------------------------------------------------------------------


def test_f9_bva_e2e_non_gold_sweep_video_zero_clicks(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    # Sweep strictly in Black, Dark Gray, and Silver on the upper side (never entering Gold)
    positions = [-78.0, -70.0, -62.0, -54.0, -45.0, -35.0, -25.0, -22.0, -25.0, -35.0, -50.0, -70.0]
    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0),
        positions,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="non_gold_sweep.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 0


def test_f9_bva_e2e_no_gauge_distractor_video_zero_clicks(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    frames = [
        sf.render_gauge_frame(
            sf.GaugeSpec(width=1920, height=1080, show_gauge=False, distractors=True, seed=i)
        )
        for i in range(60)
    ]
    mp4_path = synthetic_gauge_video_factory(frames, filename="distractor_only.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 0


def test_f9_bva_e2e_real_video_inactive_segments_zero_clicks(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    inactive_ranges = ((0, 117), (631, 724), (1339, 1569))
    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    src = e2e.make_video_source(reference_video_path)
    inactive_clicks: list[int] = []
    try:
        while True:
            pkt = src.read()
            if pkt is None:
                break
            prev_clicks = e2e.get_spy_click_count(spy)
            res = e2e.detect_gauge(det, pkt)
            e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
            spy.tick(pkt.timestamp_s)
            new_clicks = e2e.get_spy_click_count(spy)
            if new_clicks > prev_clicks:
                if any(lo <= pkt.frame_idx <= hi for lo, hi in inactive_ranges):
                    inactive_clicks.append(pkt.frame_idx)
    finally:
        src.close()

    assert inactive_clicks == [], f"Unexpected clicks on real inactive frames: {inactive_clicks}"


def test_f9_bva_e2e_near_miss_silver_vs_gold_center_video(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    base = sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0)
    # Near-miss peaking in Silver at s_norm = -1.333 (red_pos = -20.0, red_thickness=7.0 -> no gold overlap)
    near_miss_positions = [-70.0, -55.0, -40.0, -28.0, -20.0, -28.0, -42.0, -60.0]
    miss_frames = sf.generate_sweep_sequence(base, near_miss_positions)
    miss_mp4 = synthetic_gauge_video_factory(miss_frames, filename="near_miss_silver.mp4")

    proc_miss = e2e.run_cli(["verify-video", "--video", str(miss_mp4), "--dry-run", "--json"])
    assert proc_miss.returncode == 0, proc_miss.stderr
    c_miss, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_miss))
    assert c_miss == 0

    # Gold-center pass
    hit_positions = [-70.0 + i * 5.5 for i in range(26)]
    hit_frames = sf.generate_sweep_sequence(base, hit_positions)
    hit_mp4 = synthetic_gauge_video_factory(hit_frames, filename="gold_center_hit.mp4")

    proc_hit = e2e.run_cli(["verify-video", "--video", str(hit_mp4), "--dry-run", "--json"])
    assert proc_hit.returncode == 0, proc_hit.stderr
    c_hit, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_hit))
    assert c_hit == 1


def test_f9_bva_e2e_invalid_video_never_reports_zero_exit(
    e2e: Any,
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt_e2e.mp4"
    corrupt.write_bytes(b"corrupt_mp4_bytes")
    proc = e2e.run_cli(["verify-video", "--video", str(corrupt), "--dry-run", "--json"])
    assert proc.returncode != 0
