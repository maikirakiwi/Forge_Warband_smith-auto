from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tests.conftest import render_synthetic_gauge
from vrc_smith_auto.app import (
    KEY_DOWN_MASK,
    VK_ESCAPE,
    VK_F8,
    HotkeyController,
    HotkeyEvents,
    PreviewWindow,
    Runner,
    RunResult,
    build_parser,
    draw_overlay,
    main,
    run_pipeline,
)
from vrc_smith_auto.capture import CaptureError, ImageFrameSource, VideoFileFrameSource
from vrc_smith_auto.config import AppConfig
from vrc_smith_auto.control import TriggerDecision, TriggerState, TriggerStateMachine
from vrc_smith_auto.input import SpyMouseBackend, Win32SendInputMouse
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.models import DetectionResult, OrientedBox


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float = 60.0) -> Path:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    assert writer.isOpened(), f"Failed to open VideoWriter at {path}"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


# ---------------------------------------------------------------------------
# Import-Time Win32 / DPI Purity
# ---------------------------------------------------------------------------


def test_app_submodules_import_time_win32_dpi_purity() -> None:
    script = (
        "import sys; "
        "import vrc_smith_auto.app; "
        "import vrc_smith_auto.app.cli; "
        "import vrc_smith_auto.app.runner; "
        "import vrc_smith_auto.app.preview; "
        "import vrc_smith_auto.app.hotkeys; "
        "from vrc_smith_auto import window; "
        "from vrc_smith_auto.capture import gdi_source; "
        "from vrc_smith_auto.input import win32_sendinput; "
        "from vrc_smith_auto.app import hotkeys; "
        "assert window._WIN32_API is None; "
        "assert gdi_source._GDI_API is None; "
        "assert win32_sendinput._USER32_SENDINPUT is None; "
        "assert hotkeys._GET_ASYNC_KEY_STATE is None; "
        "assert 'dxcam' not in sys.modules; "
        "assert 'mss' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_app_init_preserves_hotkey_exports() -> None:
    assert KEY_DOWN_MASK == 0x8000
    assert VK_ESCAPE == 0x1B
    assert VK_F8 == 0x77
    assert HotkeyController is not None
    assert HotkeyEvents is not None


# ---------------------------------------------------------------------------
# Preview Overlay & Window
# ---------------------------------------------------------------------------


def test_draw_overlay_non_mutating_and_handles_all_state_types() -> None:
    frame = render_synthetic_gauge(width=640, height=360, center=(320.0, 180.0), red_norm_pos=0.0).image_bgr
    original = frame.copy()

    det = DetectionResult(
        gauge_present=True,
        gold_box=OrientedBox(center=(320.0, 180.0), size=(38.0, 32.0), angle_deg=-3.5),
        red_box=OrientedBox(center=(320.0, 180.0), size=(42.0, 6.0), angle_deg=-3.5),
        axis_unit=(0.06, 0.998),
        gold_span=(164.0, 196.0),
        red_proj=180.0,
        norm_pos=0.0,
        tilt_deg=-3.5,
        confidence=0.95,
        roi_used=(120, 40, 400, 280),
    )

    decision = TriggerDecision(
        should_click=True,
        state=TriggerState.LATCHED,
        previous_state=TriggerState.ARMED,
        transitions=(),
        raw_norm_pos=0.0,
        predicted_norm_pos=0.0,
        velocity_norm_per_s=1.0,
        cooldown_remaining_ms=200.0,
        hold_ms=30.0,
        dry_run=True,
        reason="entered_gold",
        timestamp_s=1.0,
    )

    for state_arg in (None, "ARMED", TriggerState.COOLDOWN, decision):
        out = draw_overlay(frame, det, state=state_arg, paused=False, fps=60.0)
        assert out is not frame
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, original)

    paused_out = draw_overlay(frame, None, state="IDLE", paused=True, fps=None)
    assert paused_out.shape == frame.shape
    assert paused_out.dtype == np.uint8
    assert np.array_equal(frame, original)


def test_preview_window_lazy_open_and_close_on_q(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    next_key = ord("q")
    visible = 1.0

    monkeypatch.setattr(cv2, "namedWindow", lambda name, flags: calls.append(("namedWindow", name)))
    monkeypatch.setattr(cv2, "imshow", lambda name, img: calls.append(("imshow", img.shape)))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: visible)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: calls.append(("destroyWindow", name)))

    win = PreviewWindow("test-preview")
    assert win.is_open is False
    assert calls == []

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    # Pressing 'q' returns False and closes window
    assert win.show(frame, None, state="ARMED") is False
    assert win.is_open is False
    assert ("namedWindow", "test-preview") in calls
    assert ("destroyWindow", "test-preview") in calls

    # Calling show() after close stays closed
    assert win.show(frame, None) is False

    # Window closed via X button (WND_PROP_VISIBLE < 1)
    next_key = 0xFF
    visible = 0.0
    with PreviewWindow("test-x") as win2:
        assert win2.show(frame, None, paused=True) is False
        assert win2.is_open is False


# ---------------------------------------------------------------------------
# Runner Loop & Hotkey / Dry-Run / Monotonicity Behavior
# ---------------------------------------------------------------------------


def test_runner_offline_sweep_clicks_with_spy_mouse(tmp_path: Path) -> None:
    frames = [
        render_synthetic_gauge(
            width=1280,
            height=720,
            center=(640.0, 360.0),
            gold_height_px=34.0,
            red_norm_pos=-3.8 + i * 0.3,
        ).image_bgr
        for i in range(27)
    ]
    mp4 = _write_mp4(tmp_path / "sweep.mp4", frames)
    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False, hold_ms=35.0)
    spy = SpyMouseBackend()

    with VideoFileFrameSource(mp4) as src:
        res = Runner(source=src, mouse=spy, config=cfg).run()

    assert isinstance(res, RunResult)
    assert res.frames_processed == 27
    assert res.clicks == 1
    assert len(res.click_frames) == 1
    assert spy.click_count == 1
    assert spy.clicks[0].frame_idx == res.click_frames[0]
    assert spy.clicks[0].duration_ms == pytest.approx(35.0)
    assert spy.is_pressed is False


def test_runner_live_idle_none_monotonic_tick_and_hotkey_pause_stop() -> None:
    # Build synthetic frames for two passes separated by None idle gaps
    p1_frames = [
        render_synthetic_gauge(
            width=1280,
            height=720,
            center=(640.0, 360.0),
            gold_height_px=34.0,
            red_norm_pos=-3.8 + i * 0.35,
        ).image_bgr
        for i in range(22)
    ]
    p2_frames = list(p1_frames)

    class MockLiveSource:
        def __init__(self) -> None:
            self.step = 0
            self.packets: list[FramePacket | None] = [None]
            t = 100.0
            fidx = 0
            for img in p1_frames:
                self.packets.append(FramePacket(img, t, fidx))
                t += 1.0 / 60.0
                fidx += 1
            # Idle gap while mouse might be pressed or between passes
            self.packets.extend([None, None])
            t += 0.1
            for img in p2_frames:
                self.packets.append(FramePacket(img, t, fidx))
                t += 1.0 / 60.0
                fidx += 1
            self.packets.extend([None, None])

        def read(self) -> FramePacket | None:
            if self.step >= len(self.packets):
                return None
            pkt = self.packets[self.step]
            self.step += 1
            return pkt

        def close(self) -> None:
            pass

    wall_t = 5000.0

    def _clock() -> float:
        nonlocal wall_t
        wall_t += 0.025
        return wall_t

    pressed_keys: set[int] = set()
    hotkeys = HotkeyController(key_state_fn=lambda vk: KEY_DOWN_MASK if vk in pressed_keys else 0)
    spy = SpyMouseBackend(clock_fn=_clock)
    src = MockLiveSource()
    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False, hold_ms=30.0)

    def _on_frame(pkt: FramePacket, _det: Any, _dec: Any) -> None:
        pressed_keys.clear()
        if pkt.frame_idx == 21:
            # Pause via F8 before second pass
            pressed_keys.add(VK_F8)
        elif pkt.frame_idx == 35:
            # Emergency stop via Esc during second pass
            pressed_keys.add(VK_ESCAPE)

    res = run_pipeline(
        source=src,
        mouse=spy,
        config=cfg,
        hotkeys=hotkeys,
        stop_on_eof=False,
        idle_sleep_s=0.0,
        clock_fn=_clock,
        on_frame=_on_frame,
    )

    assert res.stopped_by_hotkey is True
    # Only pass 1 clicked; pass 2 was paused by F8 before Esc stopped the loop
    assert res.clicks == 1
    assert spy.click_count == 1
    assert spy.is_pressed is False
    # Release timestamp must remain in packet timestamp domain (~100.x), not polluted by wall_t=5000
    assert spy.clicks[0].release_timestamp_s is not None
    assert 100.0 <= spy.clicks[0].release_timestamp_s < 101.0


def test_runner_dry_run_never_invokes_win32_sendinput(tmp_path: Path) -> None:
    frames = [
        render_synthetic_gauge(
            width=1280,
            height=720,
            center=(640.0, 360.0),
            gold_height_px=34.0,
            red_norm_pos=-3.8 + i * 0.3,
        ).image_bgr
        for i in range(27)
    ]
    mp4 = _write_mp4(tmp_path / "dry_guard.mp4", frames)
    sent: list[int] = []
    win32_mouse = Win32SendInputMouse(
        require_foreground=False,
        check_uipi=False,
        send_input_fn=lambda inp: (sent.append(int(inp.mi.dwFlags)), 1)[1],
    )
    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False)
    with VideoFileFrameSource(mp4) as src:
        res = Runner(source=src, mouse=win32_mouse, config=cfg).run()

    assert res.clicks == 1
    assert sent == []
    assert win32_mouse.is_pressed is False


def test_runner_releases_mouse_on_exception() -> None:
    class FailingSource:
        def read(self) -> FramePacket | None:
            raise RuntimeError("capture crash")

        def close(self) -> None:
            pass

    spy = SpyMouseBackend()
    assert spy.press_left(500.0, timestamp_s=1.0) is True
    assert spy.is_pressed is True

    with pytest.raises(RuntimeError, match="capture crash"):
        Runner(source=FailingSource(), mouse=spy, stop_on_eof=False).run()

    assert spy.is_pressed is False


# ---------------------------------------------------------------------------
# CLI & __main__
# ---------------------------------------------------------------------------


def test_cli_parser_and_help_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run", "--preview", "--no-focus-guard", "--backend", "mss"])
    assert args.command == "run"
    assert args.dry_run is True
    assert args.preview is True
    assert args.no_focus_guard is True
    assert args.backend == "mss"

    for help_args in (["--help"], ["run", "--help"], ["verify-image", "--help"], ["verify-video", "--help"], ["benchmark", "--help"]):
        proc = subprocess.run(
            [sys.executable, "-m", "vrc_smith_auto", *help_args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def test_cli_verify_image_and_verify_video_and_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    silver = render_synthetic_gauge(
        width=640, height=360, center=(320.0, 180.0), gold_height_px=32.0, red_norm_pos=-2.0
    ).image_bgr
    img_path = tmp_path / "silver.png"
    cv2.imwrite(str(img_path), silver)

    # verify-image pass
    rc = main(["verify-image", "--image", str(img_path), "--expect-clicks", "0", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["clicks"] == 0
    assert out["click_frames"] == []
    assert out["passed"] is True

    # verify-image mismatch -> non-zero + still emits JSON
    rc_bad = main(["verify-image", "--image", str(img_path), "--expect-clicks", "1", "--json"])
    assert rc_bad != 0
    out_bad = json.loads(capsys.readouterr().out)
    assert out_bad["passed"] is False

    # verify-image missing -> non-zero
    assert main(["verify-image", "--image", str(tmp_path / "missing.png"), "--json"]) != 0
    capsys.readouterr()

    # verify-video synthetic pass
    frames = [
        render_synthetic_gauge(
            width=1280,
            height=720,
            center=(640.0, 360.0),
            gold_height_px=34.0,
            red_norm_pos=-3.8 + i * 0.3,
        ).image_bgr
        for i in range(27)
    ]
    mp4 = _write_mp4(tmp_path / "cli_sweep.mp4", frames)
    rc_vid = main(["verify-video", "--video", str(mp4), "--dry-run", "--json"])
    assert rc_vid == 0
    vid_json = json.loads(capsys.readouterr().out)
    assert vid_json["clicks"] == 1
    assert len(vid_json["click_frames"]) == 1

    # benchmark pass
    rc_bench_ok = main(["benchmark", "--video", str(mp4), "--iterations", "20", "--max-p95-ms", "100.0", "--json"])
    assert rc_bench_ok == 0
    b_ok = json.loads(capsys.readouterr().out)
    assert b_ok["passed"] is True
    assert b_ok["iterations"] == 20
    assert b_ok["p95_ms"] <= 100.0

    # benchmark breach -> non-zero AND still prints JSON with p95_ms
    rc_bench_fail = main(["benchmark", "--video", str(mp4), "--iterations", "10", "--max-p95-ms", "0.001", "--json"])
    assert rc_bench_fail != 0
    b_fail = json.loads(capsys.readouterr().out)
    assert b_fail["passed"] is False
    assert "p95_ms" in b_fail


def test_cli_run_respects_config_overrides_and_handles_capture_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrc_smith_auto.app.cli as cli_mod

    # Missing config -> non-zero
    assert main(["run", "--config", str(tmp_path / "nope.yaml"), "--dry-run"]) != 0

    # Invalid backend -> SystemExit from argparse
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--backend", "not_a_backend"])
    assert exc_info.value.code != 0

    # CaptureError -> clean non-zero exit
    monkeypatch.setattr(
        cli_mod,
        "create_live_capture",
        lambda *_a, **_kw: (_ for _ in ()).throw(CaptureError("headless")),
    )
    assert main(["run", "--dry-run"]) != 0
    assert main(["benchmark", "--live", "--iterations", "5", "--json"]) != 0

    # Valid YAML config with dry_run=True preserved when --dry-run is NOT passed on CLI
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        AppConfig().with_overrides(dry_run=True, window_class_name="", window_title="CustomTitle").to_yaml(),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    class DummyLiveSource:
        def read(self) -> FramePacket | None:
            return None

        def close(self) -> None:
            seen["closed"] = True

    class DummyRunner:
        def __init__(self, **kwargs: Any) -> None:
            seen["runner_kwargs"] = kwargs

        def run(self) -> RunResult:
            return RunResult(0, 0, 0, (), True, False, 0.01)

    monkeypatch.setattr(cli_mod, "enable_dpi_awareness", lambda: True)
    monkeypatch.setattr(cli_mod, "create_live_capture", lambda cfg, probe=True: DummyLiveSource())
    monkeypatch.setattr(cli_mod, "Runner", DummyRunner)

    rc = main(["run", "--config", str(cfg_path), "--no-focus-guard"])
    assert rc == 0
    used_cfg: AppConfig = seen["runner_kwargs"]["config"]
    assert used_cfg.dry_run is True
    assert used_cfg.require_foreground is False
    assert isinstance(seen["runner_kwargs"]["mouse"], SpyMouseBackend)
    assert seen["closed"] is True

    # Non-dry-run wires Win32SendInputMouse with class_name=None when empty and process_name=None for custom title
    find_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        cli_mod,
        "find_vrchat_window",
        lambda title, cls, process_name=None: (find_calls.append((title, cls, process_name)), 4242)[1],
    )
    cfg_live_custom = tmp_path / "custom_live.yaml"
    cfg_live_custom.write_text(
        AppConfig().with_overrides(dry_run=False, window_class_name="", window_title="Notepad").to_yaml(),
        encoding="utf-8",
    )
    rc2 = main(["run", "--config", str(cfg_live_custom)])
    assert rc2 == 0
    live_mouse = seen["runner_kwargs"]["mouse"]
    assert isinstance(live_mouse, Win32SendInputMouse)
    assert live_mouse._resolve_hwnd() == 4242
    assert find_calls == [("Notepad", None, None)]

    # Default VRChat title/class passes UnityWndClass + VRChat.exe
    find_calls.clear()
    rc3 = main(["run"])
    assert rc3 == 0
    live_mouse_default = seen["runner_kwargs"]["mouse"]
    assert isinstance(live_mouse_default, Win32SendInputMouse)
    assert live_mouse_default._resolve_hwnd() == 4242
    assert find_calls == [("VRChat", "UnityWndClass", "VRChat.exe")]

