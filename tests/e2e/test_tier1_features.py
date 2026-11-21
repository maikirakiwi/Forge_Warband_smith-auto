from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# F1: Project Packaging & Config Schema (5 tests)
# ---------------------------------------------------------------------------


def test_f1_packaging_files_exist_and_valid_pyproject(e2e: Any) -> None:
    assert e2e.PYPROJECT_PATH.exists(), f"Missing {e2e.PYPROJECT_PATH}"
    assert e2e.REQUIREMENTS_PATH.exists(), f"Missing {e2e.REQUIREMENTS_PATH}"

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 fallback
        import tomli as tomllib

    data = tomllib.loads(e2e.PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = data.get("project", {})
    assert project.get("name") in ("vrc-smith-auto", "vrc_smith_auto")
    assert ">=3.10" in str(project.get("requires-python", ""))
    scripts = project.get("scripts", {})
    assert "vrc-smith-auto" in scripts


def test_f1_package_importable_and_core_submodules(e2e: Any) -> None:
    pkg = e2e.import_pkg("vrc_smith_auto")
    cfg_mod = e2e.import_pkg("vrc_smith_auto.config")
    proto_mod = e2e.import_pkg("vrc_smith_auto.protocols")
    win_mod = e2e.import_pkg("vrc_smith_auto.window")
    models_mod = e2e.import_pkg("vrc_smith_auto.vision.models")

    assert hasattr(pkg, "AppConfig")
    assert hasattr(cfg_mod, "AppConfig")
    assert hasattr(proto_mod, "FramePacket")
    assert hasattr(win_mod, "enable_dpi_awareness")
    assert hasattr(models_mod, "DetectionResult")


def test_f1_appconfig_defaults_match_contract(e2e: Any) -> None:
    cfg = e2e.make_app_config()
    assert cfg.capture_backend == "auto"
    assert cfg.window_title == "VRChat"
    assert cfg.require_foreground is True
    assert cfg.hold_ms == pytest.approx(1.0)
    assert cfg.cooldown_ms == pytest.approx(200.0)
    assert cfg.entry_margin == pytest.approx(0.95)
    assert cfg.exit_margin == pytest.approx(1.25)
    assert cfg.lead_ms == pytest.approx(8.3)
    assert cfg.trigger.max_lead_offset == pytest.approx(2.0)
    assert cfg.dry_run is False


def test_f1_appconfig_loads_yaml_and_json_overrides(
    e2e: Any,
    tmp_config_yaml: Any,
    tmp_config_json: Any,
) -> None:
    yaml_path = tmp_config_yaml(
        {
            "capture_backend": "gdi",
            "hold_ms": 36.0,
            "cooldown_ms": 240.0,
            "dry_run": True,
        }
    )
    json_path = tmp_config_json(
        {
            "capture_backend": "mss",
            "hold_ms": 28.0,
            "cooldown_ms": 180.0,
            "dry_run": True,
        }
    )

    cfg_y = e2e.load_app_config_from_file(yaml_path)
    assert cfg_y.capture_backend == "gdi"
    assert cfg_y.hold_ms == pytest.approx(36.0)
    assert cfg_y.cooldown_ms == pytest.approx(240.0)
    assert cfg_y.dry_run is True

    cfg_j = e2e.load_app_config_from_file(json_path)
    assert cfg_j.capture_backend == "mss"
    assert cfg_j.hold_ms == pytest.approx(28.0)
    assert cfg_j.cooldown_ms == pytest.approx(180.0)
    assert cfg_j.dry_run is True


def test_f1_cli_help_lists_all_contracted_subcommands(e2e: Any) -> None:
    proc = e2e.run_cli(["--help"])
    assert proc.returncode == 0, f"CLI --help failed: {proc.stderr}"
    combined = f"{proc.stdout}\n{proc.stderr}"
    for subcmd in ("run", "verify-image", "verify-video", "benchmark"):
        assert subcmd in combined, f"Missing subcommand '{subcmd}' in --help output"


# ---------------------------------------------------------------------------
# F2: Core Protocols & VRChat Window/DPI Tracking (5 tests)
# ---------------------------------------------------------------------------


def test_f2_frame_packet_dataclass_fields(e2e: Any) -> None:
    _, np = e2e.import_cv2_np()
    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")

    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pkt = FramePacket(image_bgr=arr, timestamp_s=1.25, frame_idx=7)
    assert pkt.image_bgr.shape == (1080, 1920, 3)
    assert pkt.timestamp_s == pytest.approx(1.25)
    assert pkt.frame_idx == 7
    assert pkt.width == 1920
    assert pkt.height == 1080


def test_f2_oriented_box_and_detection_result_fields(e2e: Any) -> None:
    OrientedBox = e2e.import_attr("vrc_smith_auto.vision.models", "OrientedBox")
    DetectionResult = e2e.import_attr("vrc_smith_auto.vision.models", "DetectionResult")

    gold = OrientedBox(center=(100.0, 200.0), size=(40.0, 30.0), angle_deg=-2.5)
    red = OrientedBox(center=(100.0, 202.0), size=(62.0, 7.0), angle_deg=-2.5)
    res = DetectionResult(
        gauge_present=True,
        gold_box=gold,
        red_box=red,
        axis_unit=(0.0, 1.0),
        gold_span=(185.0, 215.0),
        red_proj=202.0,
        norm_pos=0.133,
        tilt_deg=-2.5,
        confidence=0.95,
        roi_used=(50, 100, 200, 350),
    )
    assert res.gauge_present is True
    assert res.gold_box == gold
    assert res.red_box == red
    assert res.axis_unit == (0.0, 1.0)
    assert res.gold_span == (185.0, 215.0)
    assert res.red_proj == pytest.approx(202.0)
    assert res.norm_pos == pytest.approx(0.133)
    assert res.tilt_deg == pytest.approx(-2.5)
    assert res.confidence == pytest.approx(0.95)
    assert res.roi_used == (50, 100, 200, 350)


def test_f2_protocol_contracts_frame_source_and_mouse_backend(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    FrameSource = e2e.import_attr("vrc_smith_auto.protocols", "FrameSource")
    MouseBackend = e2e.import_attr("vrc_smith_auto.protocols", "MouseBackend")

    img_src = e2e.make_image_source(reference_screenshot_path)
    try:
        assert isinstance(img_src, FrameSource)
        assert callable(getattr(img_src, "read", None))
        assert callable(getattr(img_src, "close", None))
    finally:
        img_src.close()

    spy = e2e.make_spy_mouse()
    assert isinstance(spy, MouseBackend)
    for attr in ("press_left", "tick", "release_all", "is_pressed"):
        assert hasattr(spy, attr)


def test_f2_window_dpi_awareness_initialization_safe(e2e: Any) -> None:
    enable_dpi = e2e.import_attr("vrc_smith_auto.window", "enable_dpi_awareness")
    first = enable_dpi()
    second = enable_dpi()
    if sys.platform == "win32":
        assert first is True
        assert second is True
    else:
        assert isinstance(first, bool)


def test_f2_window_desktop_query_returns_safely(e2e: Any) -> None:
    win_mod = e2e.import_pkg("vrc_smith_auto.window")
    hwnd = win_mod.find_vrchat_window()
    assert hwnd is None or isinstance(hwnd, int)
    if hwnd is not None:
        rect = win_mod.get_client_rect(hwnd)
        assert rect is None or hasattr(rect, "as_ltrb")
    assert win_mod.is_foreground_window(0) is False


# ---------------------------------------------------------------------------
# F3: Multi-Backend Screen Capture & Offline Sources (5 tests)
# ---------------------------------------------------------------------------


def test_f3_image_frame_source_reads_reference_screenshot(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")
    src = e2e.make_image_source(reference_screenshot_path)
    try:
        pkt = src.read()
        assert isinstance(pkt, FramePacket)
        assert pkt.frame_idx == 0
        assert pkt.image_bgr.shape == (223, 325, 3)
        assert str(pkt.image_bgr.dtype) == "uint8"
    finally:
        src.close()


def test_f3_video_file_frame_source_reads_reference_video(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")
    src = e2e.make_video_source(reference_video_path)
    try:
        packets = []
        for _ in range(10):
            pkt = src.read()
            assert isinstance(pkt, FramePacket)
            assert pkt.image_bgr.shape == (1080, 1920, 3)
            packets.append(pkt)
        assert [p.frame_idx for p in packets] == list(range(10))
        for a, b in zip(packets, packets[1:]):
            assert b.timestamp_s > a.timestamp_s
    finally:
        src.close()


def test_f3_video_file_frame_source_returns_none_at_eof(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=640, height=480, cx=320.0, cy=240.0, scale=0.6),
        [-40.0, 0.0, 40.0],
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="three_frames.mp4")

    src = e2e.make_video_source(mp4_path)
    try:
        read_frames = []
        for _ in range(3):
            pkt = src.read()
            assert pkt is not None
            read_frames.append(pkt)
        assert len(read_frames) == 3
        assert src.read() is None
    finally:
        src.close()


def test_f3_gdi_bitblt_capture_grabs_desktop_region(e2e: Any) -> None:
    import ctypes

    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")
    FrameSource = e2e.import_attr("vrc_smith_auto.protocols", "FrameSource")
    CaptureError = e2e.import_optional_attr(
        ("vrc_smith_auto.capture", "vrc_smith_auto.capture.base"),
        ("CaptureError",),
    )
    GdiBitBltCapture = e2e.import_optional_attr(
        ("vrc_smith_auto.capture.gdi_source", "vrc_smith_auto.capture"),
        ("GdiBitBltCapture",),
    )

    cap = GdiBitBltCapture(region=(0, 0, 64, 64))
    try:
        assert isinstance(cap, FrameSource)
        try:
            pkt = cap.read()
        except CaptureError as exc:
            if sys.platform != "win32" or int(ctypes.windll.user32.GetForegroundWindow()) != 0:
                raise
            msg = str(exc).lower()
            assert any(
                token in msg
                for token in (
                    "bitblt",
                    "getdc",
                    "getdibits",
                    "winerror=5",
                    "access",
                    "denied",
                    "desktop",
                )
            ), f"Unexpected CaptureError message in headless session: {exc!r}"
        else:
            assert pkt is not None
            assert isinstance(pkt, FramePacket)
            assert pkt.image_bgr.ndim == 3
            assert pkt.image_bgr.shape[2] == 3
            assert pkt.image_bgr.size > 0
            assert pkt.frame_idx == 0
    finally:
        cap.close()
        cap.close()
    assert cap.read() is None


def test_f3_capture_backend_factory_resolves_gdi_and_auto(e2e: Any) -> None:
    FrameSource = e2e.import_attr("vrc_smith_auto.protocols", "FrameSource")
    cap_gdi = e2e.make_capture_backend("gdi")
    try:
        assert isinstance(cap_gdi, FrameSource)
    finally:
        cap_gdi.close()

    cap_auto = e2e.make_capture_backend("auto")
    try:
        assert isinstance(cap_auto, FrameSource)
    finally:
        cap_auto.close()


# ---------------------------------------------------------------------------
# F4: In-Place Win32 Mouse Driver & Hotkeys (5 tests)
# ---------------------------------------------------------------------------


def test_f4_spy_mouse_press_hold_and_tick_release(e2e: Any) -> None:
    spy = e2e.make_spy_mouse()
    assert spy.is_pressed is False

    started = spy.press_left(30.0, timestamp_s=1.0)
    assert started is True
    assert spy.is_pressed is True

    spy.tick(1.015)
    assert spy.is_pressed is True

    spy.tick(1.035)
    assert spy.is_pressed is False
    assert e2e.get_spy_click_count(spy) == 1


def test_f4_spy_mouse_release_all_clears_press_state(e2e: Any) -> None:
    spy = e2e.make_spy_mouse()
    assert spy.press_left(40.0, timestamp_s=2.0) is True
    assert spy.is_pressed is True

    spy.release_all()
    assert spy.is_pressed is False


def test_f4_win32_sendinput_mouse_conforms_and_safe_cleanup(e2e: Any) -> None:
    MouseBackend = e2e.import_attr("vrc_smith_auto.protocols", "MouseBackend")
    Win32SendInputMouse = e2e.import_optional_attr(
        ("vrc_smith_auto.input.win32_sendinput", "vrc_smith_auto.input"),
        ("Win32SendInputMouse",),
    )

    sig = inspect.signature(Win32SendInputMouse)
    kwargs: dict[str, Any] = {}
    if "dry_run" in sig.parameters:
        kwargs["dry_run"] = True
    mouse = Win32SendInputMouse(**kwargs)
    assert isinstance(mouse, MouseBackend)
    mouse.release_all()
    assert mouse.is_pressed is False


def test_f4_win32_sendinput_focus_guard_respects_unfocused_state(e2e: Any) -> None:
    win_mod = e2e.import_pkg("vrc_smith_auto.window")
    Win32SendInputMouse = e2e.import_optional_attr(
        ("vrc_smith_auto.input.win32_sendinput", "vrc_smith_auto.input"),
        ("Win32SendInputMouse",),
    )
    missing_hwnd = win_mod.find_vrchat_window(
        "__NoSuchVRChatWindow_FocusGuard__",
        class_name=None,
        process_name=None,
    )
    assert missing_hwnd is None
    assert win_mod.is_foreground_window(0) is False

    sig = inspect.signature(Win32SendInputMouse)
    params = sig.parameters
    if "focus_check" in params:
        mouse = Win32SendInputMouse(focus_check=lambda: False)
    elif "can_click" in params:
        mouse = Win32SendInputMouse(can_click=lambda: False)
    elif "is_foreground_fn" in params:
        kwargs: dict[str, Any] = {"is_foreground_fn": lambda _hwnd: False}
        if "require_foreground" in params:
            kwargs["require_foreground"] = True
        if "target_hwnd" in params:
            kwargs["target_hwnd"] = 1234
        if "check_uipi" in params:
            kwargs["check_uipi"] = False
        if "send_input_fn" in params:
            kwargs["send_input_fn"] = lambda _inp: 1
        mouse = Win32SendInputMouse(**kwargs)
    elif "require_foreground" in params and "window_title" in params:
        mouse = Win32SendInputMouse(
            require_foreground=True,
            window_title="__NoSuchVRChatWindow_FocusGuard__",
        )
    elif "require_foreground" in params:
        mouse = Win32SendInputMouse(require_foreground=True)
    elif "config" in params:
        cfg = e2e.make_app_config(
            require_foreground=True,
            window_title="__NoSuchVRChatWindow_FocusGuard__",
        )
        mouse = Win32SendInputMouse(config=cfg)
    else:
        pytest.fail("No supported focus-guard parameter found on Win32SendInputMouse")

    assert mouse.press_left(30.0, timestamp_s=1.0) is False
    assert mouse.is_pressed is False


def test_f4_hotkeys_nonblocking_poll_returns_valid_state(e2e: Any) -> None:
    hotkeys_mod = e2e.import_pkg("vrc_smith_auto.app.hotkeys")
    poller = None
    for cls_name in ("HotkeyController", "HotkeyPoller", "Hotkeys", "HotkeyListener", "HotkeyMonitor"):
        cls = getattr(hotkeys_mod, cls_name, None)
        if isinstance(cls, type):
            sig = inspect.signature(cls)
            kwargs: dict[str, Any] = {}
            if "key_state_fn" in sig.parameters:
                kwargs["key_state_fn"] = lambda _vk: 0
            elif "key_probe" in sig.parameters:
                kwargs["key_probe"] = lambda _vk: False
            elif "is_key_down" in sig.parameters:
                kwargs["is_key_down"] = lambda _vk: False
            poller = cls(**kwargs)
            break

    if poller is not None:
        for method_name in ("poll", "update", "step", "check"):
            fn = getattr(poller, method_name, None)
            if callable(fn):
                state = fn()
                assert state is not None or hasattr(poller, "paused")
                return

    for fn_name in ("poll_hotkeys", "check_hotkeys"):
        fn = getattr(hotkeys_mod, fn_name, None)
        if callable(fn):
            res = fn()
            assert res is not None
            return

    pytest.fail("vrc_smith_auto.app.hotkeys exposes no public poller class or poll function")


# ---------------------------------------------------------------------------
# F5: Tilt/Shift-Invariant Gauge Detector (6 tests)
# ---------------------------------------------------------------------------


def test_f5_detector_upright_synthetic_gauge_centered_gold(e2e: Any) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1280,
            height=720,
            cx=640.0,
            cy=360.0,
            scale=720.0 / 1080.0,
            gold_len=34.0,
            red_pos=0.0,
        )
    )
    detector = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(detector, frame)

    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.red_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.25
    assert res.confidence > 0.5


def test_f5_detector_invariant_to_screen_position_shift(e2e: Any) -> None:
    import synthetic_fixtures as sf

    cfg = e2e.make_app_config()
    det_a = e2e.make_gauge_detector(cfg)
    det_b = e2e.make_gauge_detector(cfg)

    frame_a = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1280,
            height=720,
            cx=320.0,
            cy=220.0,
            scale=720.0 / 1080.0,
            gold_len=34.0,
            red_pos=0.0,
        )
    )
    frame_b = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1280,
            height=720,
            cx=960.0,
            cy=500.0,
            scale=720.0 / 1080.0,
            gold_len=34.0,
            red_pos=0.0,
        )
    )

    res_a = e2e.detect_gauge(det_a, frame_a)
    res_b = e2e.detect_gauge(det_b, frame_b)

    assert res_a.gauge_present is True and res_b.gauge_present is True
    assert res_a.norm_pos is not None and res_b.norm_pos is not None
    assert abs(res_a.norm_pos) <= 0.30
    assert abs(res_b.norm_pos) <= 0.30
    assert abs(res_a.norm_pos - res_b.norm_pos) <= 0.30


def test_f5_detector_invariant_to_3d_tilt_angles(e2e: Any) -> None:
    import synthetic_fixtures as sf

    cfg = e2e.make_app_config()
    for tilt in (-12.0, 12.0):
        det = e2e.make_gauge_detector(cfg)
        frame = sf.render_gauge_frame(
            sf.GaugeSpec(width=1920, height=1080, cx=960.0, cy=540.0, tilt_deg=tilt, red_pos=0.0)
        )
        res = e2e.detect_gauge(det, frame)
        assert res.gauge_present is True
        assert res.norm_pos is not None
        assert abs(res.norm_pos) <= 0.35
        assert res.tilt_deg is not None


def test_f5_detector_bridges_red_split_occlusion_across_gold(e2e: Any) -> None:
    import synthetic_fixtures as sf

    frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1920,
            height=1080,
            cx=850.0,
            cy=410.0,
            gold_len=24.0,
            red_pos=0.0,
            red_width=64.0,
            red_thickness=9.0,
        )
    )
    det = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(det, frame)

    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.30


def test_f5_detector_measures_silver_zone_and_ignores_distractors(e2e: Any) -> None:
    import synthetic_fixtures as sf

    spec = sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0, distractors=True)
    silver_pos = sf.red_pos_for_norm(spec, -1.6)
    spec = sf.GaugeSpec(
        width=1920,
        height=1080,
        cx=850.0,
        cy=410.0,
        gold_len=30.0,
        red_pos=silver_pos,
        distractors=True,
    )
    frame = sf.render_gauge_frame(spec)

    det = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(det, frame)

    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.0


def test_f5_detector_reference_screenshot_silver_zone(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    src = e2e.make_image_source(reference_screenshot_path)
    try:
        pkt = src.read()
        assert pkt is not None
    finally:
        src.close()

    det = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(det, pkt)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert res.red_box is not None
    assert res.norm_pos is not None
    assert abs(res.norm_pos) > 1.0


# ---------------------------------------------------------------------------
# F6: Trigger Debounce & Cooldown State Machine (5 tests)
# ---------------------------------------------------------------------------


def _det(e2e: Any, *, present: bool = True, norm_pos: float | None = None) -> Any:
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


def test_f6_state_machine_arms_outside_and_triggers_in_gold(e2e: Any) -> None:
    cfg = e2e.make_app_config(cooldown_ms=200.0, hold_ms=30.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det(e2e, norm_pos=-2.0), 1.000, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 0

    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.0), 1.016, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1


def test_f6_state_machine_suppresses_double_click_inside_gold(e2e: Any) -> None:
    cfg = e2e.make_app_config(cooldown_ms=200.0, hold_ms=30.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det(e2e, norm_pos=-1.8), 1.000, mouse=spy)
    for i, s in enumerate([-0.6, -0.2, 0.0, 0.3, 0.7]):
        e2e.step_state_machine(sm, _det(e2e, norm_pos=s), 1.016 + i * 0.020, mouse=spy)

    assert e2e.get_spy_click_count(spy) == 1


def test_f6_state_machine_rearms_after_exit_and_cooldown(e2e: Any) -> None:
    cfg = e2e.make_app_config(cooldown_ms=200.0, hold_ms=30.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det(e2e, norm_pos=-1.8), 0.980, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.0), 1.000, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1

    # Exit Gold into Silver and wait past 200 ms cooldown
    e2e.step_state_machine(sm, _det(e2e, norm_pos=1.6), 1.150, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=1.6), 1.250, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.0), 1.350, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 2


def test_f6_state_machine_blocks_double_click_on_strike_transition(e2e: Any) -> None:
    cfg = e2e.make_app_config(cooldown_ms=200.0, hold_ms=30.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det(e2e, norm_pos=-1.6), 1.000, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.0), 1.016, mouse=spy)
    # Immediate H_k -> H_{k+1} transition within 50ms without exiting Gold
    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.1), 1.050, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=-0.2), 1.080, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1


def test_f6_state_machine_resets_cleanly_when_gauge_disappears(e2e: Any) -> None:
    cfg = e2e.make_app_config(cooldown_ms=200.0, hold_ms=30.0)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    e2e.step_state_machine(sm, _det(e2e, norm_pos=-1.8), 1.000, mouse=spy)
    e2e.step_state_machine(sm, _det(e2e, norm_pos=0.0), 1.020, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1

    for i in range(6):
        e2e.step_state_machine(sm, _det(e2e, present=False), 1.100 + i * 0.016, mouse=spy)

    # Reappearance in Silver must not trigger a spurious click
    e2e.step_state_machine(sm, _det(e2e, norm_pos=1.5), 1.300, mouse=spy)
    assert e2e.get_spy_click_count(spy) == 1


# ---------------------------------------------------------------------------
# F7: Runner, Debug Preview, CLI & Latency Benchmark (5 tests)
# ---------------------------------------------------------------------------


def test_f7_runner_processes_offline_video_with_spy_mouse(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    positions = [-65.0 + i * 5.0 for i in range(27)]  # -65 -> +65 across Gold
    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, scale=0.8, gold_len=34.0),
        positions,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="runner_1pass.mp4")

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    src = e2e.make_video_source(mp4_path)
    spy = e2e.make_spy_mouse()
    try:
        e2e.run_app_runner(source=src, mouse=spy, config=cfg)
    finally:
        src.close()

    assert e2e.get_spy_click_count(spy) == 1


def test_f7_cli_verify_image_reference_screenshot_zero_clicks(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    proc = e2e.run_cli(
        ["verify-image", "--image", str(reference_screenshot_path), "--expect-clicks", "0", "--json"]
    )
    assert proc.returncode == 0, f"verify-image failed: {proc.stderr}"
    data = e2e.parse_cli_json(proc)
    clicks, _ = e2e.extract_cli_clicks(data)
    assert clicks == 0


def test_f7_cli_verify_video_dry_run_synthetic_pass(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    positions = [-65.0 + i * 5.0 for i in range(27)]
    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, scale=0.8, gold_len=34.0),
        positions,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="cli_1pass.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, f"verify-video failed: {proc.stderr}"
    data = e2e.parse_cli_json(proc)
    clicks, _ = e2e.extract_cli_clicks(data)
    assert clicks == 1


def test_f7_cli_benchmark_outputs_latency_json(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    proc = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(reference_video_path),
            "--iterations",
            "30",
            "--max-p95-ms",
            "50.0",
            "--json",
        ]
    )
    assert proc.returncode == 0, f"benchmark failed: {proc.stderr}"
    data = e2e.parse_cli_json(proc)
    assert any(k in data for k in ("mean_ms", "avg_ms"))
    assert "p95_ms" in data


def test_f7_preview_overlay_renders_annotated_bgr_frame(e2e: Any) -> None:
    import synthetic_fixtures as sf

    preview_mod = e2e.import_pkg("vrc_smith_auto.app.preview")
    frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, red_pos=0.0)
    )
    det = _det(e2e, norm_pos=0.0)

    rendered = None
    for fn_name in ("draw_overlay", "render_overlay", "annotate_frame", "draw_debug_overlay"):
        fn = getattr(preview_mod, fn_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            for pname in sig.parameters:
                if pname in ("frame", "frame_bgr", "image", "image_bgr"):
                    kwargs[pname] = frame.copy()
                elif pname in ("detection", "result", "det"):
                    kwargs[pname] = det
                elif pname in ("state", "sm_state"):
                    kwargs[pname] = "ARMED"
                elif pname == "paused":
                    kwargs[pname] = False
            rendered = fn(**kwargs) if kwargs else fn(frame.copy(), det)
            break

    if rendered is None:
        for cls_name in ("PreviewWindow", "DebugPreview"):
            cls = getattr(preview_mod, cls_name, None)
            if isinstance(cls, type):
                inst = cls()
                for mname in ("annotate", "render", "draw_overlay"):
                    m = getattr(inst, mname, None)
                    if callable(m):
                        rendered = m(frame.copy(), det)
                        break

    assert rendered is not None
    assert rendered.shape == frame.shape
    assert str(rendered.dtype) == "uint8"


# ---------------------------------------------------------------------------
# F8: Technical Design Document & User Guide (5 tests)
# ---------------------------------------------------------------------------


def test_f8_technical_design_doc_exists_and_nonempty(e2e: Any) -> None:
    assert e2e.TECH_DESIGN_PATH.exists(), f"Missing {e2e.TECH_DESIGN_PATH}"
    text = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8")
    assert len(text) > 1000


def test_f8_technical_design_compares_capture_vision_input(e2e: Any) -> None:
    text = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8").lower()
    assert "dxcam" in text
    assert "mss" in text
    assert "gdi" in text
    assert "opencv" in text or "hsv" in text
    assert "sendinput" in text


def test_f8_technical_design_includes_benchmark_and_repro_commands(e2e: Any) -> None:
    text = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8").lower()
    assert "benchmark" in text
    assert "verify-video" in text or "pytest" in text
    assert "ms" in text


def test_f8_readme_documents_setup_and_cli_commands(e2e: Any) -> None:
    assert e2e.README_PATH.exists(), f"Missing {e2e.README_PATH}"
    text = e2e.README_PATH.read_text(encoding="utf-8")
    for cmd in ("run", "verify-image", "verify-video", "benchmark"):
        assert cmd in text, f"README.md missing CLI command '{cmd}'"


def test_f8_readme_documents_hotkeys_and_troubleshooting(e2e: Any) -> None:
    text = e2e.README_PATH.read_text(encoding="utf-8")
    assert "F8" in text
    assert "Esc" in text or "ESC" in text
    assert "--dry-run" in text
    assert "DPI" in text or "VRChat" in text


# ---------------------------------------------------------------------------
# F9: Opaque-Box End-to-End Verification & Acceptance Contracts (5 tests)
# ---------------------------------------------------------------------------


def test_f9_cli_verify_image_silver_nogauge_and_detector_gold(
    e2e: Any,
    tmp_path: Path,
) -> None:
    import synthetic_fixtures as sf

    # Silver-zone PNG -> verify-image expects 0 clicks (state machine starts IDLE, no prior arm)
    silver_spec = sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, gold_len=34.0)
    silver_frame = sf.render_gauge_frame(
        sf.GaugeSpec(
            width=1280,
            height=720,
            cx=640.0,
            cy=360.0,
            gold_len=34.0,
            red_pos=sf.red_pos_for_norm(silver_spec, -1.6),
        )
    )
    silver_png = sf.write_png(tmp_path / "silver_zone.png", silver_frame)
    proc_silver = e2e.run_cli(
        ["verify-image", "--image", str(silver_png), "--expect-clicks", "0", "--json"]
    )
    assert proc_silver.returncode == 0, proc_silver.stderr
    c_silver, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_silver))
    assert c_silver == 0

    # No-gauge PNG -> verify-image expects 0 clicks
    nogauge_png = sf.write_png(
        tmp_path / "no_gauge.png",
        sf.render_gauge_frame(sf.GaugeSpec(width=1280, height=720, show_gauge=False, distractors=True)),
    )
    proc_nogauge = e2e.run_cli(
        ["verify-image", "--image", str(nogauge_png), "--expect-clicks", "0", "--json"]
    )
    assert proc_nogauge.returncode == 0, proc_nogauge.stderr
    c_nogauge, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_nogauge))
    assert c_nogauge == 0

    # Gold-centered single PNG verified via GaugeDetector directly
    gold_frame = sf.render_gauge_frame(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, gold_len=34.0, red_pos=0.0)
    )
    det = e2e.make_gauge_detector(e2e.make_app_config())
    res = e2e.detect_gauge(det, gold_frame)
    assert res.gauge_present is True
    assert res.norm_pos is not None
    assert abs(res.norm_pos) <= 0.30


def test_f9_cli_verify_video_synthetic_gold_pass_triggers_once(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    positions = [-70.0 + i * 4.8 for i in range(30)]
    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=34.0),
        positions,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="f9_sweep.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, click_frames = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 1
    assert len(click_frames) == clicks
    assert 10 <= click_frames[0] <= 20


def test_f9_cli_verify_video_no_gauge_video_zero_clicks(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    frames = [
        sf.render_gauge_frame(sf.GaugeSpec(width=1280, height=720, show_gauge=False))
        for _ in range(20)
    ]
    mp4_path = synthetic_gauge_video_factory(frames, filename="f9_empty.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 0


def test_f9_cli_run_help_exposes_contracted_flags(e2e: Any) -> None:
    proc = e2e.run_cli(["run", "--help"])
    assert proc.returncode == 0, proc.stderr
    combined = f"{proc.stdout}\n{proc.stderr}"
    for flag in ("--config", "--backend", "--dry-run", "--preview", "--no-focus-guard"):
        assert flag in combined, f"Missing flag '{flag}' in `run --help`"


def test_f9_end_to_end_pipeline_determinism_on_repeated_runs(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    positions = [-68.0 + i * 5.0 for i in range(28)]
    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, scale=0.8, gold_len=32.0),
        positions,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="f9_determinism.mp4")

    proc_1 = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    proc_2 = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc_1.returncode == 0 and proc_2.returncode == 0

    c1, f1 = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_1))
    c2, f2 = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_2))
    assert c1 == c2 == 1
    assert f1 == f2
