from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest


def test_t3_c1_tilt_shift_distractors_and_red_occlusion_pipeline(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    strikes = [
        sf.StrikeSpec(
            frames=30,
            cx=1380.0,
            cy=320.0,
            tilt_deg=22.0,
            gold_offset=0.0,
            gold_len=18.0,
            red_start=-68.0,
            red_velocity_px_per_frame=4.8,
            distractors=True,
            red_width=64.0,
            red_thickness=8.0,
        ),
        sf.StrikeSpec(
            frames=20,
            cx=960.0,
            cy=540.0,
            show_gauge=False,
            distractors=True,
        ),
        sf.StrikeSpec(
            frames=30,
            cx=520.0,
            cy=680.0,
            tilt_deg=-18.0,
            gold_offset=0.0,
            gold_len=18.0,
            red_start=-68.0,
            red_velocity_px_per_frame=4.8,
            distractors=True,
            red_width=64.0,
            red_thickness=8.0,
        ),
    ]
    frames, truth = sf.generate_multi_strike_sequence(strikes)
    mp4_path = synthetic_gauge_video_factory(frames, filename="t3_c1_combo.mp4")

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    click_frames: list[int] = []
    src = e2e.make_video_source(mp4_path)
    try:
        while True:
            pkt = src.read()
            if pkt is None:
                break
            prev = e2e.get_spy_click_count(spy)
            res = e2e.detect_gauge(det, pkt)
            e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
            spy.tick(pkt.timestamp_s)
            if e2e.get_spy_click_count(spy) > prev:
                click_frames.append(pkt.frame_idx)
    finally:
        src.close()

    assert len(click_frames) == 2
    for fidx in click_frames:
        assert truth[fidx]["show_gauge"] is True
        assert truth[fidx]["any_overlap"] is True


def test_t3_c2_rapid_strike_transition_hk_to_hk1_no_double_click(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    strikes = [
        # Strike H_k: enters Gold around frame 10 -> 1 click, transitions at frame 15 (~83ms later)
        sf.StrikeSpec(
            frames=15,
            cx=850.0,
            cy=410.0,
            gold_offset=0.0,
            gold_len=38.0,
            red_start=-65.0,
            red_velocity_px_per_frame=5.0,
            bounce_bounds=(-72.0, 72.0),
        ),
        # Strike H_{k+1}: red carries over inside Gold, exits to +72, bounces back >250ms later -> 2nd click
        sf.StrikeSpec(
            frames=45,
            cx=850.0,
            cy=410.0,
            gold_offset=8.0,
            gold_len=24.0,
            red_start=None,
            red_velocity_px_per_frame=5.0,
            bounce_bounds=(-72.0, 72.0),
        ),
    ]
    frames, truth = sf.generate_multi_strike_sequence(strikes)
    mp4_path = synthetic_gauge_video_factory(frames, filename="t3_c2_transition.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, click_frames = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 2
    assert len(click_frames) == 2
    assert click_frames[1] - click_frames[0] >= 12
    # Judge on the frame the game processes the input (lead + dispatch + game tick).
    for fidx in click_frames:
        judged = fidx + 2
        assert 0 <= judged < len(truth)
        assert truth[judged]["show_gauge"] is True
        assert truth[judged]["any_overlap"] is True


def test_t3_c3_dry_run_verify_video_and_spy_mouse_parity(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    strikes = [
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=32.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
        sf.StrikeSpec(frames=18, cx=850.0, cy=410.0, show_gauge=False),
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=28.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
        sf.StrikeSpec(frames=18, cx=850.0, cy=410.0, show_gauge=False),
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=24.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
    ]
    frames, _ = sf.generate_multi_strike_sequence(strikes)
    mp4_path = synthetic_gauge_video_factory(frames, filename="t3_c3_parity.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    cli_clicks, cli_frames = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert cli_clicks == 3

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    spy = e2e.make_spy_mouse()
    src = e2e.make_video_source(mp4_path)
    try:
        e2e.run_app_runner(source=src, mouse=spy, config=cfg)
    finally:
        src.close()

    assert e2e.get_spy_click_count(spy) == cli_clicks == 3
    events = e2e.get_spy_click_events(spy)
    if cli_frames and events:
        if isinstance(events[0], dict) and "frame_idx" in events[0]:
            assert [int(ev["frame_idx"]) for ev in events] == cli_frames
        elif getattr(events[0], "frame_idx", None) is not None:
            assert [int(ev.frame_idx) for ev in events] == cli_frames


def test_t3_c4_config_overrides_cli_flags_and_benchmark_enforcement(
    e2e: Any,
    tmp_config_yaml: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    DetectionResult = e2e.import_attr("vrc_smith_auto.vision.models", "DetectionResult")
    OrientedBox = e2e.import_attr("vrc_smith_auto.vision.models", "OrientedBox")

    frames = sf.generate_sweep_sequence(
        sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=30.0),
        [-65.0 + i * 4.5 for i in range(30)],
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="t3_c4_bench.mp4")
    cfg_path = tmp_config_yaml(
        {
            "entry_margin": 0.85,
            "cooldown_ms": 220.0,
            "hold_ms": 35.0,
            "dry_run": True,
        }
    )

    loaded = e2e.load_app_config_from_file(cfg_path)
    assert loaded.entry_margin == pytest.approx(0.85)
    assert loaded.cooldown_ms == pytest.approx(220.0)
    assert loaded.hold_ms == pytest.approx(35.0)
    assert loaded.dry_run is True

    # Verify loaded YAML overrides actually control the pipeline via run_app_runner + SpyMouseBackend
    run_cfg = loaded.with_overrides(require_foreground=False) if hasattr(loaded, "with_overrides") else loaded
    spy = e2e.make_spy_mouse()
    src = e2e.make_video_source(mp4_path)
    try:
        e2e.run_app_runner(source=src, mouse=spy, config=run_cfg)
    finally:
        src.close()
    assert e2e.get_spy_click_count(spy) == 1
    events = e2e.get_spy_click_events(spy)
    if events and hasattr(events[0], "duration_ms"):
        assert events[0].duration_ms == pytest.approx(35.0)

    # Verify entry_margin=0.85 and cooldown_ms=220.0 overrides in TriggerStateMachine
    def _det_at(norm_pos: float) -> Any:
        box = OrientedBox(center=(850.0, 410.0), size=(38.0, 30.0), angle_deg=0.0)
        return DetectionResult(
            gauge_present=True,
            gold_box=box,
            red_box=box,
            axis_unit=(0.0, 1.0),
            gold_span=(395.0, 425.0),
            red_proj=410.0 + norm_pos * 15.0,
            norm_pos=norm_pos,
            tilt_deg=0.0,
            confidence=0.95,
        )

    spy_sm = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(run_cfg, mouse=spy_sm)
    e2e.step_state_machine(sm, _det_at(-1.8), 1.000, mouse=spy_sm)
    # 0.90 is inside default 0.95 but outside YAML override 0.85 -> no trigger
    e2e.step_state_machine(sm, _det_at(0.90), 1.016, mouse=spy_sm)
    assert e2e.get_spy_click_count(spy_sm) == 0
    # 0.80 is inside 0.85 -> triggers
    e2e.step_state_machine(sm, _det_at(0.80), 1.032, mouse=spy_sm)
    assert e2e.get_spy_click_count(spy_sm) == 1
    # Re-entry at +205 ms (< 220 ms override, though > 200 ms default) -> blocked
    e2e.step_state_machine(sm, _det_at(1.6), 1.150, mouse=spy_sm)
    e2e.step_state_machine(sm, _det_at(0.0), 1.237, mouse=spy_sm)
    assert e2e.get_spy_click_count(spy_sm) == 1
    # Re-entry after > 220 ms -> allowed
    e2e.step_state_machine(sm, _det_at(1.6), 1.260, mouse=spy_sm)
    e2e.step_state_machine(sm, _det_at(0.0), 1.300, mouse=spy_sm)
    assert e2e.get_spy_click_count(spy_sm) == 2

    # Contracted CLI benchmark invocations (no --config per PROJECT.md CLI Contract)
    proc_ok = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(mp4_path),
            "--iterations",
            "30",
            "--max-p95-ms",
            "25.0",
            "--json",
        ]
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    data_ok = e2e.parse_cli_json(proc_ok)
    assert float(data_ok["p95_ms"]) < 25.0

    proc_fail = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(mp4_path),
            "--iterations",
            "15",
            "--max-p95-ms",
            "0.001",
            "--json",
        ]
    )
    assert proc_fail.returncode != 0
    assert "p95_ms" in e2e.parse_cli_json(proc_fail)


def test_t3_c5_roi_reacquisition_across_inactive_gap_and_jump(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    strikes = [
        sf.StrikeSpec(
            frames=28,
            cx=420.0,
            cy=300.0,
            tilt_deg=-12.0,
            gold_len=30.0,
            red_start=-65.0,
            red_velocity_px_per_frame=4.8,
        ),
        sf.StrikeSpec(
            frames=20,
            cx=960.0,
            cy=540.0,
            show_gauge=False,
            distractors=True,
        ),
        sf.StrikeSpec(
            frames=28,
            cx=1450.0,
            cy=720.0,
            tilt_deg=28.0,
            gold_len=30.0,
            red_start=-65.0,
            red_velocity_px_per_frame=4.8,
        ),
    ]
    frames, truth = sf.generate_multi_strike_sequence(strikes)
    mp4_path = synthetic_gauge_video_factory(frames, filename="t3_c5_roi_jump.mp4")

    proc = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    clicks, click_frames = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert clicks == 2
    for fidx in click_frames:
        assert truth[fidx]["show_gauge"] is True
        assert truth[fidx]["any_overlap"] is True


def test_t3_c6_720p_scaling_with_custom_yaml_verify_image_and_video(
    e2e: Any,
    tmp_path: Path,
    tmp_config_yaml: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    scale_720p = 720.0 / 1080.0
    cfg_path = tmp_config_yaml(
        {
            "capture_backend": "gdi",
            "entry_margin": 0.95,
            "exit_margin": 1.25,
            "dry_run": True,
        },
        filename="custom_720p.yaml",
    )

    loaded = e2e.load_app_config_from_file(cfg_path)
    run_cfg = loaded.with_overrides(require_foreground=False) if hasattr(loaded, "with_overrides") else loaded

    # 720p Silver PNG -> runner & contracted verify-image CLI both yield 0 clicks
    silver_spec = sf.GaugeSpec(
        width=1280,
        height=720,
        cx=640.0,
        cy=360.0,
        scale=scale_720p,
        gold_len=30.0,
    )
    silver_png = sf.write_png(
        tmp_path / "720p_silver.png",
        sf.render_gauge_frame(
            sf.GaugeSpec(
                width=1280,
                height=720,
                cx=640.0,
                cy=360.0,
                scale=scale_720p,
                gold_len=30.0,
                red_pos=sf.red_pos_for_norm(silver_spec, -1.6),
            )
        ),
    )
    spy_img = e2e.make_spy_mouse()
    img_src = e2e.make_image_source(silver_png)
    try:
        e2e.run_app_runner(source=img_src, mouse=spy_img, config=run_cfg)
    finally:
        img_src.close()
    assert e2e.get_spy_click_count(spy_img) == 0

    proc_img = e2e.run_cli(
        [
            "verify-image",
            "--image",
            str(silver_png),
            "--expect-clicks",
            "0",
            "--json",
        ]
    )
    assert proc_img.returncode == 0, proc_img.stderr
    c_img, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_img))
    assert c_img == 0

    # 720p Silver -> Gold -> Silver pass -> runner & contracted verify-video CLI yield 1 click
    sweep_frames = sf.generate_sweep_sequence(
        silver_spec,
        [-68.0 + i * 5.0 for i in range(28)],
    )
    mp4_720 = synthetic_gauge_video_factory(sweep_frames, filename="720p_pass.mp4")

    spy_vid = e2e.make_spy_mouse()
    vid_src = e2e.make_video_source(mp4_720)
    try:
        e2e.run_app_runner(source=vid_src, mouse=spy_vid, config=run_cfg)
    finally:
        vid_src.close()
    assert e2e.get_spy_click_count(spy_vid) == 1

    proc_vid = e2e.run_cli(
        [
            "verify-video",
            "--video",
            str(mp4_720),
            "--dry-run",
            "--json",
        ]
    )
    assert proc_vid.returncode == 0, proc_vid.stderr
    c_vid, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_vid))
    assert c_vid == 1


def test_t3_c7_velocity_lead_fast_narrow_gold_with_spy_mouse_tick(e2e: Any) -> None:
    import synthetic_fixtures as sf

    cfg = e2e.make_app_config(
        lead_ms=8.0,
        entry_margin=0.90,
        exit_margin=1.25,
        hold_ms=30.0,
        cooldown_ms=200.0,
    )
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    # 360 px/s = 6 px/frame across 14 px Gold
    positions = [-66.0 + i * 6.0 for i in range(23)]
    spec = sf.GaugeSpec(width=1920, height=1080, cx=850.0, cy=410.0, gold_len=14.0, red_thickness=5.0)
    frames = sf.generate_sweep_sequence(spec, positions)

    dt = 1.0 / 60.0
    click_positions: list[float] = []
    for idx, frame in enumerate(frames):
        ts = idx * dt
        spy.tick(ts)
        prev = e2e.get_spy_click_count(spy)
        res = e2e.detect_gauge(det, frame)
        e2e.step_state_machine(sm, res, ts, mouse=spy, frame_idx=idx)
        if e2e.get_spy_click_count(spy) > prev:
            click_positions.append(positions[idx])

    # Advance clock past hold_ms=30 ms to confirm clean release
    spy.tick(len(frames) * dt + 0.050)
    assert e2e.get_spy_click_count(spy) == 1
    assert spy.is_pressed is False
    # Click should happen near/inside Gold (allowing up to 1 frame velocity lead)
    assert abs(click_positions[0]) <= 15.0


def test_t3_c8_focus_guard_vs_dry_run_mode_matrix(
    e2e: Any,
    tmp_path: Path,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    win_mod = e2e.import_pkg("vrc_smith_auto.window")
    Win32SendInputMouse = e2e.import_optional_attr(
        ("vrc_smith_auto.input.win32_sendinput", "vrc_smith_auto.input"),
        ("Win32SendInputMouse",),
    )

    # 1. Live focus guard blocks click when target VRChat window does not exist
    hwnd = win_mod.find_vrchat_window("__NoSuchVRChat_Matrix__", class_name=None, process_name=None)
    assert hwnd is None
    assert win_mod.is_foreground_window(0) is False
    sig = inspect.signature(Win32SendInputMouse)
    params = sig.parameters
    if "focus_check" in params:
        live_mouse = Win32SendInputMouse(focus_check=lambda: False)
    elif "can_click" in params:
        live_mouse = Win32SendInputMouse(can_click=lambda: False)
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
        live_mouse = Win32SendInputMouse(**kwargs)
    elif "require_foreground" in params and "window_title" in params:
        live_mouse = Win32SendInputMouse(
            require_foreground=True,
            window_title="__NoSuchVRChat_Matrix__",
        )
    elif "require_foreground" in params:
        live_mouse = Win32SendInputMouse(require_foreground=True)
    elif "config" in params:
        cfg = e2e.make_app_config(
            require_foreground=True,
            window_title="__NoSuchVRChat_Matrix__",
        )
        live_mouse = Win32SendInputMouse(config=cfg)
    else:
        pytest.fail("No supported focus-guard parameter found on Win32SendInputMouse")

    assert live_mouse.press_left(30.0, timestamp_s=1.0) is False
    assert live_mouse.is_pressed is False

    # 2. Dry-run verify-image on Silver PNG -> 0 clicks, exit 0
    silver_spec = sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, gold_len=34.0)
    silver_png = sf.write_png(
        tmp_path / "c8_silver.png",
        sf.render_gauge_frame(
            sf.GaugeSpec(
                width=1280,
                height=720,
                cx=640.0,
                cy=360.0,
                gold_len=34.0,
                red_pos=sf.red_pos_for_norm(silver_spec, -1.6),
            )
        ),
    )
    proc_silver = e2e.run_cli(
        ["verify-image", "--image", str(silver_png), "--expect-clicks", "0", "--json"]
    )
    assert proc_silver.returncode == 0, proc_silver.stderr

    # 3. Dry-run verify-video on Silver -> Gold -> Silver sequence -> 1 click, exit 0
    sweep_frames = sf.generate_sweep_sequence(silver_spec, [-65.0 + i * 5.0 for i in range(27)])
    mp4_path = synthetic_gauge_video_factory(sweep_frames, filename="c8_dry_run.mp4")
    proc_vid = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert proc_vid.returncode == 0, proc_vid.stderr
    clicks, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_vid))
    assert clicks == 1


def test_t3_c9_docs_cli_schema_and_end_to_end_contract_parity(
    e2e: Any,
    tmp_path: Path,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    readme = e2e.README_PATH.read_text(encoding="utf-8")
    tech = e2e.TECH_DESIGN_PATH.read_text(encoding="utf-8")
    for cmd in ("verify-image", "verify-video", "benchmark"):
        assert cmd in readme or cmd in tech

    # 1. verify-image on Silver PNG
    spec = sf.GaugeSpec(width=1280, height=720, cx=640.0, cy=360.0, gold_len=34.0)
    silver_png = sf.write_png(
        tmp_path / "c9_silver.png",
        sf.render_gauge_frame(
            sf.GaugeSpec(
                width=1280,
                height=720,
                cx=640.0,
                cy=360.0,
                gold_len=34.0,
                red_pos=sf.red_pos_for_norm(spec, -1.6),
            )
        ),
    )
    p_img = e2e.run_cli(["verify-image", "--image", str(silver_png), "--expect-clicks", "0", "--json"])
    assert p_img.returncode == 0, p_img.stderr
    e2e.parse_cli_json(p_img)

    # 2. verify-video on 1-pass MP4
    frames = sf.generate_sweep_sequence(spec, [-65.0 + i * 5.0 for i in range(27)])
    mp4_path = synthetic_gauge_video_factory(frames, filename="c9_video.mp4")
    p_vid = e2e.run_cli(["verify-video", "--video", str(mp4_path), "--dry-run", "--json"])
    assert p_vid.returncode == 0, p_vid.stderr
    c_vid, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(p_vid))
    assert c_vid == 1

    # 3. benchmark
    p_bench = e2e.run_cli(
        ["benchmark", "--video", str(mp4_path), "--iterations", "15", "--max-p95-ms", "50.0", "--json"]
    )
    assert p_bench.returncode == 0, p_bench.stderr
    b_data = e2e.parse_cli_json(p_bench)
    assert "p95_ms" in b_data


def test_t3_c10_hotkey_pause_resume_and_esc_emergency_release(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    hotkeys_mod = e2e.import_pkg("vrc_smith_auto.app.hotkeys")
    VK_F8 = 0x77
    VK_ESCAPE = 0x1B

    strikes = [
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=32.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=32.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
        sf.StrikeSpec(frames=26, cx=850.0, cy=410.0, gold_len=32.0, red_start=-65.0, red_velocity_px_per_frame=5.0),
    ]
    frames, _ = sf.generate_multi_strike_sequence(strikes)
    mp4_path = synthetic_gauge_video_factory(frames, filename="c10_hotkeys.mp4")

    pressed_keys: set[int] = set()
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

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    paused_state = False
    esc_armed = False
    esc_tested_while_pressed = False
    click_frames: list[int] = []

    src = e2e.make_video_source(mp4_path)
    try:
        while True:
            pkt = src.read()
            if pkt is None:
                break

            # Inject F8 before pass 2 (frame 26), F8 again before pass 3 (frame 52),
            # and Esc on the frame immediately following click #2 while spy.is_pressed is True
            pressed_keys.clear()
            if pkt.frame_idx in (26, 52):
                pressed_keys.add(VK_F8)
            elif esc_armed:
                pressed_keys.add(VK_ESCAPE)

            hk_state = poll_fn()
            if hasattr(hk_state, "toggle_armed"):
                if hk_state.toggle_armed:
                    paused_state = not paused_state
                paused = paused_state
            else:
                paused = bool(getattr(hk_state, "paused", getattr(poller, "paused", False)))

            stop_req = bool(
                getattr(
                    hk_state,
                    "emergency_stop",
                    getattr(
                        hk_state,
                        "stop",
                        getattr(hk_state, "stop_requested", getattr(hk_state, "exit", getattr(poller, "stop_requested", False))),
                    ),
                )
            )
            if stop_req:
                assert spy.is_pressed is True, "Expected mouse press active when Esc triggers emergency release"
                spy.release_all()
                assert spy.is_pressed is False
                esc_tested_while_pressed = True
                break

            spy.tick(pkt.timestamp_s)
            if paused:
                continue

            prev_clicks = e2e.get_spy_click_count(spy)
            res = e2e.detect_gauge(det, pkt)
            e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
            new_clicks = e2e.get_spy_click_count(spy)
            if new_clicks > prev_clicks:
                click_frames.append(pkt.frame_idx)
                if new_clicks == 2 and prev_clicks == 1:
                    esc_armed = True
    finally:
        spy.release_all()
        src.close()

    assert esc_tested_while_pressed is True
    assert e2e.get_spy_click_count(spy) == 2
    assert len(click_frames) == 2
    assert click_frames[0] < 26
    assert click_frames[1] >= 52
    assert spy.is_pressed is False
