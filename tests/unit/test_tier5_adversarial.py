from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from tests.conftest import SYNTH_BG_BGR, SYNTH_GOLD_BGR, SYNTH_SILVER_BGR, render_synthetic_gauge
from tests.unit.test_capture_backends import (
    _DummyCOMError,
    _FakeCamera,
    _FakeClock,
    _FakeDxcamModule,
    _FakeGdiApi,
)
from tests.unit.test_config_window import _FakeWin32Builder, _patch_win32
from vrc_smith_auto.app import cli, hotkeys as hotkeys_mod
from vrc_smith_auto.app.hotkeys import HotkeyController, HotkeyEvents, VK_ESCAPE, VK_F8
from vrc_smith_auto.app.preview import PreviewWindow, draw_overlay
from vrc_smith_auto.app.runner import Runner
from vrc_smith_auto.capture.base import (
    CaptureError,
    WindowRegionTracker,
    coerce_region,
    create_live_capture,
    normalize_bgr,
)
from vrc_smith_auto.capture.dxcam_source import DxcamCapture
from vrc_smith_auto.capture.gdi_source import HGDI_ERROR, GdiBitBltCapture, _resolve_gdi_monitor
from vrc_smith_auto.capture.mss_source import MssCapture, _select_mss_monitor
from vrc_smith_auto.capture.offline_source import ImageFrameSource, VideoFileFrameSource
from vrc_smith_auto.config import AppConfig, CaptureConfig, TriggerConfig, WindowConfig
from vrc_smith_auto.control.state_machine import TriggerState, TriggerStateMachine
from vrc_smith_auto.input import SpyMouseBackend
from vrc_smith_auto.input import win32_sendinput as sendinput_mod
from vrc_smith_auto.input.win32_sendinput import (
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    Win32SendInputMouse,
    _get_win32_send_input,
)
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.detector import GaugeDetector
from vrc_smith_auto.vision.models import DetectionResult, OrientedBox
from vrc_smith_auto.window import (
    WindowRect,
    enable_dpi_awareness,
    find_vrchat_window,
    get_monitor_rect,
    is_current_process_elevated,
    is_target_elevated,
)


def _det(present: bool = True, norm_pos: float | None = 0.0) -> DetectionResult:
    box = OrientedBox(center=(100.0, 100.0), size=(20.0, 20.0), angle_deg=0.0)
    return DetectionResult(
        gauge_present=present,
        gold_box=box if present else None,
        red_box=box if present else None,
        norm_pos=norm_pos if present else None,
        gold_span=(90.0, 110.0) if present else None,
    )


# ---------------------------------------------------------------------------
# Challenger 1 Adversarial Tests (1–9)
# ---------------------------------------------------------------------------


def test_t5_window_region_tracker_custom_title_and_transient_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG-01 + BUG-02 + WindowRegionTracker transient loss, recovery, and disjoint fallback."""
    clock = _FakeClock(10.0, step=0.0)
    calls_3arg: list[tuple[str, str | None, str | None]] = []

    def _find_3arg(
        title: str, class_name: str | None = None, *, process_name: str | None = None
    ) -> int | None:
        calls_3arg.append((title, class_name, process_name))
        return 101

    rect_state: dict[int, WindowRect | None] = {101: WindowRect(10, 20, 810, 620)}

    # Custom title -> process_name=None; empty class_name -> None
    tracker_custom = WindowRegionTracker(
        window_config=WindowConfig(title="CustomForgeTitle", class_name="", rect_refresh_s=1.0),
        crop_to_window=True,
        clock_fn=clock,
        find_window_fn=_find_3arg,
        get_client_rect_fn=lambda hwnd: rect_state.get(hwnd),
    )
    mon = WindowRect(0, 0, 1920, 1080)
    assert tracker_custom.resolve_screen_rect(mon) == WindowRect(10, 20, 810, 620)
    assert calls_3arg == [("CustomForgeTitle", None, None)]

    # Transient loss on refresh -> clears hwnd and falls back to monitor rect
    clock.now += 1.5
    rect_state[101] = None
    assert tracker_custom.resolve_screen_rect(mon) == mon
    assert tracker_custom.hwnd is None

    # Recovery on next refresh -> re-acquires hwnd and clamps rect
    clock.now += 1.5
    rect_state[101] = WindowRect(-50, 30, 1950, 1000)
    assert tracker_custom.resolve_screen_rect(mon) == WindowRect(0, 30, 1920, 1000)
    assert tracker_custom.hwnd == 101

    # Disjoint rect outside monitor -> falls back to monitor rect
    clock.now += 1.5
    rect_state[101] = WindowRect(3000, 3000, 3400, 3400)
    assert tracker_custom.resolve_screen_rect(mon) == mon

    # Default VRChat title passes process_name="VRChat.exe"
    calls_vrc: list[tuple[str, str | None, str | None]] = []

    def _find_vrc(
        title: str, class_name: str | None = None, *, process_name: str | None = None
    ) -> int | None:
        calls_vrc.append((title, class_name, process_name))
        return None

    tracker_vrc = WindowRegionTracker(
        window_config=WindowConfig(title="VRChat", class_name="UnityWndClass"),
        crop_to_window=True,
        clock_fn=clock,
        find_window_fn=_find_vrc,
        get_client_rect_fn=lambda _hwnd: None,
    )
    tracker_vrc.resolve_screen_rect(mon)
    assert calls_vrc == [("VRChat", "UnityWndClass", "VRChat.exe")]

    # Legacy 2-arg find_window_fn remains supported
    calls_2arg: list[tuple[str, str | None]] = []

    def _find_2arg(title: str, class_name: str | None = None) -> int | None:
        calls_2arg.append((title, class_name))
        return None

    WindowRegionTracker(
        window_config=WindowConfig(title="VRChat", class_name=""),
        crop_to_window=True,
        clock_fn=clock,
        find_window_fn=_find_2arg,
        get_client_rect_fn=lambda _hwnd: None,
    ).resolve_screen_rect(mon)
    assert calls_2arg == [("VRChat", None)]

    # BUG-02: find_vrchat_window with class_name="" acts as wildcard
    b = _FakeWin32Builder(
        windows=[
            {
                "hwnd": 555,
                "title": "CustomForgeTitle",
                "class_name": "CustomAnyClass",
                "pid": 2222,
                "image_path": r"C:\Apps\other.exe",
            }
        ]
    )
    _patch_win32(monkeypatch, b)
    assert find_vrchat_window("CustomForgeTitle", "", process_name=None) == 555
    assert find_vrchat_window("CustomForgeTitle", "UnityWndClass", process_name=None) is None


def test_t5_gdi_resize_reallocation_and_error_cleanup() -> None:
    """GDI bitmap resize deletes old bitmap, failure branches raise CaptureError and clean up."""
    clock = _FakeClock(step=0.0)
    rects = [WindowRect(0, 0, 800, 600), WindowRect(0, 0, 1024, 768)]
    idx = {"i": 0}

    def _get_rect(_hwnd: int) -> WindowRect | None:
        return rects[idx["i"]]

    api = _FakeGdiApi(monitors=[(WindowRect(0, 0, 1920, 1080), True)], hbitmap=777)
    cap = GdiBitBltCapture(
        crop_to_window=True,
        window_config=WindowConfig(rect_refresh_s=0.5),
        api=api,
        clock_fn=clock,
        find_window_fn=lambda *_args, **_kw: 1,
        get_client_rect_fn=_get_rect,
    )
    pkt1 = cap.read()
    assert pkt1 is not None and pkt1.image_bgr.shape == (600, 800, 3)

    clock.now += 1.0
    idx["i"] = 1
    pkt2 = cap.read()
    assert pkt2 is not None and pkt2.image_bgr.shape == (768, 1024, 3)
    assert ("DeleteObject", 777) in api.events
    assert any(ev[0] == "CreateCompatibleBitmap" and ev[2:] == (1024, 768) for ev in api.events)
    assert api.captured_bmi is not None and api.captured_bmi[0] == 1024 and api.captured_bmi[1] == -768
    cap.close()

    # CreateCompatibleBitmap == 0 -> CaptureError
    api_nobmp = _FakeGdiApi(hbitmap=0)
    cap_nobmp = GdiBitBltCapture(crop_to_window=False, api=api_nobmp, clock_fn=clock)
    with pytest.raises(CaptureError, match="CreateCompatibleBitmap"):
        cap_nobmp.read()
    cap_nobmp.close()
    assert any(ev[0] == "DeleteDC" for ev in api_nobmp.events)
    assert any(ev[0] == "ReleaseDC" for ev in api_nobmp.events)

    # GetDC == 0 -> CaptureError without leaking
    api_nodc = _FakeGdiApi(src_dc=0)
    with pytest.raises(CaptureError, match="GetDC"):
        GdiBitBltCapture(crop_to_window=False, api=api_nodc, clock_fn=clock)

    # SelectObject == HGDI_ERROR -> CaptureError
    api_badsel = _FakeGdiApi(old_obj=HGDI_ERROR)
    cap_badsel = GdiBitBltCapture(crop_to_window=False, api=api_badsel, clock_fn=clock)
    with pytest.raises(CaptureError, match="SelectObject"):
        cap_badsel.read()
    cap_badsel.close()

    # _resolve_gdi_monitor fallback & OOB errors
    api_nomon = _FakeGdiApi(monitors=[])
    assert _resolve_gdi_monitor(api_nomon, 0) == WindowRect(0, 0, 1920, 1080)
    with pytest.raises(CaptureError, match="out of range"):
        _resolve_gdi_monitor(api_nomon, 1)

    api_zero_metrics = _FakeGdiApi(monitors=[])
    api_zero_metrics.user32.GetSystemMetrics = lambda _idx: 0  # type: ignore[method-assign]
    with pytest.raises(CaptureError, match="Invalid primary screen metrics"):
        _resolve_gdi_monitor(api_zero_metrics, 0)


def test_t5_dxcam_fallback_partial_init_and_stop_exception_cleanup() -> None:
    """Dxcam multi-output fallback, 0x0 cleanup, close() stop exception, grab() tracker reset."""
    cam_ok = _FakeCamera(
        width=1280,
        height=720,
        frames=[np.full((720, 1280, 3), 42, dtype=np.uint8)],
    )
    create_calls: list[tuple[int, int | None]] = []

    class _OutObj:
        def __init__(self, primary: bool) -> None:
            self.is_primary = primary

    class _Factory:
        outputs = [[_OutObj(True), _OutObj(False)]]
        output_metadata: dict[Any, Any] = {}

    class _FallbackDxcam:
        _factory = _Factory()

        @staticmethod
        def create(*, device_idx=0, output_idx=None, **_kwargs):
            create_calls.append((device_idx, output_idx))
            if output_idx == 0:
                raise _DummyCOMError("busy")
            return cam_ok

    cap = DxcamCapture(
        monitor_idx=0,
        crop_to_window=False,
        dxcam_module=_FallbackDxcam(),
        clock_fn=_FakeClock(),
    )
    assert create_calls == [(0, 0), (0, 1)]
    pkt = cap.read()
    assert pkt is not None and pkt.image_bgr.shape == (720, 1280, 3)

    # close() when cam.is_capturing=True and stop() raises still calls release()
    cam_ok.is_capturing = True

    def _boom_stop() -> None:
        raise RuntimeError("stop failed")

    cam_ok.stop = _boom_stop  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stop failed"):
        cap.close()
    assert cam_ok.release_calls == 1

    # Invalid device_idx / empty outputs
    class _EmptyFactory:
        outputs: list[list[Any]] = []
        output_metadata: dict[Any, Any] = {}

    class _EmptyOutputs:
        _factory = _EmptyFactory()

    with pytest.raises(CaptureError, match="No DXGI outputs"):
        DxcamCapture(dxcam_module=_EmptyOutputs(), crop_to_window=False)

    with pytest.raises(CaptureError, match="out of range"):
        DxcamCapture(dxcam_module=_FallbackDxcam(), device_idx=3, crop_to_window=False)

    # 0x0 output dims calls cam.release() during __init__ cleanup
    cam_zero = _FakeCamera(width=0, height=0)
    mod_zero = _FakeDxcamModule()
    mod_zero.behaviors[(0, None)] = cam_zero
    with pytest.raises(CaptureError, match="invalid dimensions"):
        DxcamCapture(dxcam_module=mod_zero, crop_to_window=False)
    assert cam_zero.release_calls == 1

    # grab() COMError invalidates tracker and raises CaptureError
    cam_err = _FakeCamera(width=1280, height=720, grab_exc=_DummyCOMError("access lost"))
    mod_err = _FakeDxcamModule()
    mod_err.behaviors[(0, None)] = cam_err
    cap_err = DxcamCapture(
        dxcam_module=mod_err,
        crop_to_window=True,
        find_window_fn=lambda *_args, **_kw: 99,
        get_client_rect_fn=lambda _hwnd: WindowRect(0, 0, 1280, 720),
    )
    with pytest.raises(CaptureError, match="dxcam grab failed"):
        cap_err.read()
    assert cap_err._region_tracker.hwnd is None
    cap_err.close()


def test_t5_mss_and_create_live_capture_edge_cleanup() -> None:
    """_select_mss_monitor edges, MssCapture factory error, create_live_capture probe cleanup."""

    class _BrokenPrimarySct:
        monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
        ]

        @property
        def primary_monitor(self) -> Any:
            raise RuntimeError("primary probe failed")

    assert _select_mss_monitor(_BrokenPrimarySct(), 0) == WindowRect(0, 0, 1920, 1080)

    class _EmptySct:
        monitors = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]

    with pytest.raises(CaptureError, match="No physical monitors"):
        _select_mss_monitor(_EmptySct(), 0)

    class _MalformedSct:
        monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 0, "height": 1080},
        ]

    with pytest.raises(CaptureError, match="non-positive dimensions"):
        _select_mss_monitor(_MalformedSct(), 0)

    with pytest.raises(CaptureError, match="Failed to initialize mss"):
        MssCapture(mss_factory=lambda: (_ for _ in ()).throw(RuntimeError("mss init boom")))

    with pytest.raises(TypeError, match="Expected AppConfig or CaptureConfig"):
        create_live_capture("not-a-config")  # type: ignore[arg-type]

    # CaptureConfig + explicit window_config + probe cleanup when close() also raises
    close_calls = {"n": 0}

    class _FailingCloseSource:
        def read(self) -> FramePacket | None:
            raise RuntimeError("probe boom")

        def close(self) -> None:
            close_calls["n"] += 1
            raise RuntimeError("close also boom")

    with pytest.MonkeyPatch.context() as mp:
        captured_win_cfg: list[WindowConfig] = []

        def _fake_build(name, cap_cfg, win_cfg, region):
            captured_win_cfg.append(win_cfg)
            return _FailingCloseSource()

        mp.setattr("vrc_smith_auto.capture.base._build_backend", _fake_build)
        custom_win = WindowConfig(title="CustomTitle", class_name="")
        with pytest.raises(CaptureError, match="probe failed: probe boom"):
            create_live_capture(
                CaptureConfig(backend="gdi"),
                window_config=custom_win,
                probe=True,
            )
        assert captured_win_cfg == [custom_win]
        assert close_calls["n"] == 1


def test_t5_runner_preview_quit_pause_mid_hold_and_exception_cleanup() -> None:
    """Preview false mid-hold, on_frame exception cleanup, F8 pause mid-hold, max_frames, None idle tick."""
    sample_gold = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=0.0).image_bgr
    sample_arm = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=-1.6).image_bgr

    class _FakePreview:
        def __init__(self, stop_on_call: int) -> None:
            self.stop_on_call = stop_on_call
            self.calls = 0
            self.closed = False

        def show(self, *_args: Any, **_kwargs: Any) -> bool:
            self.calls += 1
            return self.calls < self.stop_on_call

        def close(self) -> None:
            self.closed = True

    class _SeqSource:
        def __init__(self, packets: list[FramePacket | None]) -> None:
            self._packets = list(packets)
            self._idx = 0

        def read(self) -> FramePacket | None:
            if self._idx >= len(self._packets):
                return None
            out = self._packets[self._idx]
            self._idx += 1
            return out

        def close(self) -> None:
            pass

    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False, hold_ms=100.0)

    # 1. Preview returns False mid-hold -> stopped_by_preview=True & mouse released
    pkts_1 = [
        FramePacket(image_bgr=sample_arm, timestamp_s=1.0, frame_idx=0),
        FramePacket(image_bgr=sample_gold, timestamp_s=1.02, frame_idx=1),
    ]
    mouse1 = SpyMouseBackend()
    prev1 = _FakePreview(stop_on_call=2)
    res1 = Runner(
        source=_SeqSource(pkts_1),
        mouse=mouse1,
        config=cfg,
        preview=prev1,  # type: ignore[arg-type]
        show_preview=True,
        stop_on_eof=True,
    ).run()
    assert res1.stopped_by_preview is True
    assert res1.clicks == 1
    assert mouse1.is_pressed is False
    assert mouse1.clicks[-1].released is True
    assert prev1.closed is False  # caller-provided preview is not owned

    # 2. Owned PreviewWindow cleanup when on_frame raises
    with pytest.MonkeyPatch.context() as mp:
        destroyed = {"n": 0}
        mp.setattr(PreviewWindow, "show", lambda self, *_a, **_kw: True)
        mp.setattr(
            PreviewWindow,
            "close",
            lambda self: destroyed.__setitem__("n", destroyed["n"] + 1),
        )
        mouse2 = SpyMouseBackend()
        pkts_2 = [
            FramePacket(image_bgr=sample_arm, timestamp_s=1.0, frame_idx=0),
            FramePacket(image_bgr=sample_gold, timestamp_s=1.02, frame_idx=1),
        ]

        def _boom_on_frame(pkt: FramePacket, _det: Any, _dec: Any) -> None:
            if pkt.frame_idx == 1:
                raise RuntimeError("hook boom")

        with pytest.raises(RuntimeError, match="hook boom"):
            Runner(
                source=_SeqSource(pkts_2),
                mouse=mouse2,
                config=cfg,
                show_preview=True,
                on_frame=_boom_on_frame,
                stop_on_eof=True,
            ).run()
        assert mouse2.is_pressed is False
        assert mouse2.clicks[-1].released is True
        assert destroyed["n"] == 1

    # 3. F8 toggle pause mid-hold releases mouse and resets state machine, second F8 resumes
    f8_state = {"down": False}

    def _poll_key(vk: int) -> int:
        if vk == VK_F8 and f8_state["down"]:
            return 0x8000
        return 0

    hotkeys = HotkeyController(key_state_fn=_poll_key)
    mouse3 = SpyMouseBackend()
    # Ensure frame 3 sets f8_state["down"] = False, frame 4 sets it True -> resumes before frame 5 (arm), clicks on frame 6
    pkts_3 = [
        FramePacket(image_bgr=sample_arm, timestamp_s=1.00, frame_idx=0),
        FramePacket(image_bgr=sample_gold, timestamp_s=1.02, frame_idx=1),  # click 1 + hold
        FramePacket(image_bgr=sample_arm, timestamp_s=1.04, frame_idx=2),  # sets f8=True -> pauses before frame 3 mid-hold
        FramePacket(image_bgr=sample_gold, timestamp_s=1.30, frame_idx=3),  # ignored while paused (sets f8=False)
        FramePacket(image_bgr=sample_arm, timestamp_s=1.35, frame_idx=4),  # ignored while paused (sets f8=True -> resumes before frame 5)
        FramePacket(image_bgr=sample_arm, timestamp_s=1.36, frame_idx=5),  # re-arms SM after resume
        FramePacket(image_bgr=sample_gold, timestamp_s=1.38, frame_idx=6),  # click 2
    ]

    class _HotkeyToggleSource(_SeqSource):
        def read(self) -> FramePacket | None:
            f8_state["down"] = self._idx in (2, 4)
            return super().read()

    res3 = Runner(
        source=_HotkeyToggleSource(pkts_3),
        mouse=mouse3,
        config=cfg,
        hotkeys=hotkeys,
        stop_on_eof=True,
    ).run()
    assert res3.clicks == 2
    assert res3.click_frames == (1, 6)
    assert mouse3.is_pressed is False

    # 4. max_frames=0 and max_frames=3
    res_zero = Runner(
        source=ImageFrameSource.from_array(sample_arm, repeat=5),
        mouse=SpyMouseBackend(),
        config=cfg,
        stop_on_eof=True,
        max_frames=0,
    ).run()
    assert res_zero.frames_processed == 0

    res_three = Runner(
        source=ImageFrameSource.from_array(sample_arm, repeat=10),
        mouse=SpyMouseBackend(),
        config=cfg,
        stop_on_eof=True,
        max_frames=3,
    ).run()
    assert res_three.frames_processed == 3

    # 5. Live source None-read tick releases hold via wall clock
    clock = _FakeClock(1.0, step=0.0)
    mouse5 = SpyMouseBackend()

    class _IdleThenStopSource:
        def __init__(self) -> None:
            self.n = 0

        def read(self) -> FramePacket | None:
            self.n += 1
            if self.n == 1:
                clock.now += 0.01
                return FramePacket(image_bgr=sample_arm, timestamp_s=1.01, frame_idx=0)
            if self.n == 2:
                clock.now += 0.02
                return FramePacket(image_bgr=sample_gold, timestamp_s=1.03, frame_idx=1)
            clock.now += 0.05
            return None

        def close(self) -> None:
            pass

    esc_calls = {"n": 0}

    def _esc_on_idle(vk: int) -> int:
        if vk == VK_ESCAPE:
            esc_calls["n"] += 1
            if esc_calls["n"] >= 6:
                return 0x8000
        return 0

    res5 = Runner(
        source=_IdleThenStopSource(),
        mouse=mouse5,
        config=cfg,
        hotkeys=HotkeyController(key_state_fn=_esc_on_idle),
        stop_on_eof=False,
        idle_sleep_s=0.0,
        clock_fn=clock,
    ).run()
    assert res5.stopped_by_hotkey is True
    assert res5.clicks == 1
    assert mouse5.clicks[0].release_timestamp_s is not None
    assert mouse5.clicks[0].release_timestamp_s >= 1.13


def test_t5_cli_live_benchmark_invalid_args_and_run_exception_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI benchmark --live success/timeout, invalid args, 0-frame video, run cleanup, plain text."""
    assert cli.main(["benchmark", "--video", "x.mp4", "--iterations", "0"]) == 2
    assert cli.main(["benchmark", "--video", "x.mp4", "--max-p95-ms", "0"]) == 2
    assert cli.main(["benchmark", "--video", "x.mp4", "--max-p95-ms", "nan"]) == 2
    assert cli.main(["verify-image", "--image", "x.png", "--expect-clicks", "-1"]) == 2

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert cli.main(["run", "--config", str(bad_json)]) == 2

    bad_schema = tmp_path / "schema.json"
    bad_schema.write_text(json.dumps({"unknown_section": 1}), encoding="utf-8")
    assert cli.main(["run", "--config", str(bad_schema)]) == 2

    # benchmark --live success with mocked create_live_capture
    sample = render_synthetic_gauge(gold_height_px=30.0, red_norm_pos=-1.5).image_bgr
    monkeypatch.setattr(cli, "enable_dpi_awareness", lambda: True)
    monkeypatch.setattr(
        cli,
        "create_live_capture",
        lambda *_a, **_kw: ImageFrameSource.from_array(sample, repeat=30, fps=60.0),
    )
    capsys.readouterr()
    rc_live = cli.main(["benchmark", "--live", "--iterations", "5", "--max-p95-ms", "200.0", "--json"])
    assert rc_live == 0
    payload_live = json.loads(capsys.readouterr().out.strip())
    assert payload_live["mode"] == "live" and payload_live["passed"] is True

    # benchmark --live timeout when source always returns None
    class _AlwaysNoneSource:
        def read(self) -> FramePacket | None:
            return None

        def close(self) -> None:
            pass

    t_fake = _FakeClock(0.0, step=2.0)
    monkeypatch.setattr(cli, "create_live_capture", lambda *_a, **_kw: _AlwaysNoneSource())
    monkeypatch.setattr(cli.time, "perf_counter", t_fake)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert cli.main(["benchmark", "--live", "--iterations", "3"]) == 1

    # 0-frame video for benchmark -> exit 1
    class _ZeroVideoSource:
        fps = 60.0

        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def read(self) -> FramePacket | None:
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "VideoFileFrameSource", _ZeroVideoSource)
    assert cli.main(["benchmark", "--video", "empty.mp4", "--iterations", "2"]) == 1

    # run KeyboardInterrupt -> exit 0 & closes preview/source; RuntimeError -> exit 1 & closes both
    for raised, expected_rc in [(KeyboardInterrupt(), 0), (RuntimeError("crash"), 1)]:
        state = {"src_closed": 0, "prev_closed": 0}

        class _LiveSrc:
            def read(self) -> FramePacket | None:
                return None

            def close(self) -> None:
                state["src_closed"] += 1

        class _FakePrev:
            def close(self) -> None:
                state["prev_closed"] += 1

        class _BoomRunner:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass

            def run(self) -> Any:
                raise raised

        with monkeypatch.context() as mp:
            mp.setattr(cli, "create_live_capture", lambda *_a, **_kw: _LiveSrc())
            mp.setattr(cli, "PreviewWindow", _FakePrev)
            mp.setattr(cli, "Runner", _BoomRunner)
            assert cli.main(["run", "--dry-run", "--preview"]) == expected_rc
            assert state == {"src_closed": 1, "prev_closed": 1}

    # Plain-text non-JSON outputs for verify-image and verify-video
    img_path = tmp_path / "shot.png"
    cv2.imwrite(str(img_path), sample)
    capsys.readouterr()
    assert cli.main(["verify-image", "--image", str(img_path), "--expect-clicks", "0"]) == 0
    assert "verify-image:" in capsys.readouterr().out

    class _SingleFrameVideo:
        fps = 60.0

        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self.done = False

        def read(self) -> FramePacket | None:
            if self.done:
                return None
            self.done = True
            return FramePacket(image_bgr=sample, timestamp_s=0.0, frame_idx=0)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "VideoFileFrameSource", _SingleFrameVideo)
    assert cli.main(["verify-video", "--video", "ok.mp4", "--dry-run"]) == 0
    assert "verify-video:" in capsys.readouterr().out


def test_t5_window_dpi_failure_token_errors_and_find_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DPI failure across 3 fallbacks, token elevation failures closing handles, monitor/find edges."""
    b = _FakeWin32Builder()
    b.dpi_context_ret = 0
    b.dpi_context_err = 0
    b.shcore_hr = -1
    b.dpi_aware_ret = 0
    _patch_win32(monkeypatch, b)
    assert enable_dpi_awareness() is False

    # Token query failures close opened handles and return None
    b_tok = _FakeWin32Builder(
        windows=[{"hwnd": 77, "title": "VRChat", "class_name": "UnityWndClass", "pid": 4444}]
    )
    b_tok.advapi32.GetTokenInformation = lambda *_a, **_kw: 0  # type: ignore[method-assign]
    _patch_win32(monkeypatch, b_tok)
    assert is_current_process_elevated() is None
    assert is_target_elevated(77) is None
    assert len(b_tok.closed_handles) >= 3

    # get_monitor_rect errors: invalid hwnd, MonitorFromWindow==0, GetMonitorInfoW==0
    b_mon = _FakeWin32Builder(
        windows=[{"hwnd": 88, "title": "VRChat", "class_name": "UnityWndClass"}]
    )
    _patch_win32(monkeypatch, b_mon)
    assert get_monitor_rect(0) is None
    b_mon.user32.MonitorFromWindow = lambda _h, _f: 0  # type: ignore[method-assign]
    assert get_monitor_rect(88) is None
    b_mon.user32.MonitorFromWindow = lambda _h, _f: 777  # type: ignore[method-assign]
    b_mon.user32.GetMonitorInfoW = lambda _hm, _mi: 0  # type: ignore[method-assign]
    assert get_monitor_rect(88) is None

    # find_vrchat_window whitespace title -> None; EnumWindows error -> None
    assert find_vrchat_window("   ") is None

    b_enum = _FakeWin32Builder()
    _patch_win32(monkeypatch, b_enum)
    b_enum.last_error = 5
    b_enum.user32.EnumWindows = lambda _cb, _lp: 0  # type: ignore[method-assign]
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)
    assert find_vrchat_window("VRChat") is None


def test_t5_config_and_offline_source_validation_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AppConfig/OfflineSource/capture/preview validation edges including BUG-03 (H,W,1)."""
    with pytest.raises(ValueError):
        AppConfig.from_dict({123: "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        AppConfig.from_dict({"trigger.": 1})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"bogus_section": {}})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"hsv": {"gold_low": "not-a-3-tuple"}})

    bad_list_json = tmp_path / "list.json"
    bad_list_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        AppConfig.from_file(bad_list_json)

    with pytest.raises(ValueError):
        AppConfig().with_overrides({"nonexistent_field": 1})
    with pytest.raises(ValueError):
        AppConfig().with_overrides({"cooldown_ms": -5.0})

    # VideoFileFrameSource validation edges
    with pytest.raises(ValueError):
        VideoFileFrameSource("any.mp4", max_frames=-1)
    with pytest.raises(ValueError):
        VideoFileFrameSource("any.mp4", default_fps=0.0)

    # Preload=False, loop=True, max_frames=2, fps=0 metadata fallback & max_frames=0 yields None
    fake_frames = [
        np.full((40, 60, 3), 10, dtype=np.uint8),
        np.full((40, 60, 3), 20, dtype=np.uint8),
        np.full((40, 60, 3), 30, dtype=np.uint8),
    ]

    class _FakeCap:
        def __init__(self, _path: str) -> None:
            self.idx = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return 0.0

        def set(self, _prop: int, _val: float) -> bool:
            self.idx = 0
            return True

        def read(self) -> tuple[bool, Any]:
            if self.idx >= len(fake_frames):
                return False, None
            fr = fake_frames[self.idx]
            self.idx += 1
            return True, fr

        def release(self) -> None:
            pass

    vid_path = tmp_path / "fake.mp4"
    vid_path.write_bytes(b"stub")
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

    vs_zero = VideoFileFrameSource(vid_path, max_frames=0)
    assert vs_zero.read() is None
    vs_zero.close()

    vs = VideoFileFrameSource(vid_path, preload=False, loop=True, max_frames=2, default_fps=45.0)
    assert vs.fps == pytest.approx(45.0)
    p0 = vs.read()
    p1 = vs.read()
    p2 = vs.read()
    assert p0 is not None and p1 is not None and p2 is not None
    assert p0.image_bgr[0, 0, 0] == 10
    assert p1.image_bgr[0, 0, 0] == 20
    assert p2.image_bgr[0, 0, 0] == 10
    vs.close()

    # ImageFrameSource validation edges
    with pytest.raises(ValueError):
        ImageFrameSource.from_array(np.zeros((10, 10, 3), dtype=np.uint8), fps=0.0)
    with pytest.raises(ValueError):
        ImageFrameSource.from_array(np.zeros((10, 10, 3), dtype=np.uint8), repeat=-1)
    assert ImageFrameSource.from_array(np.zeros((10, 10, 3), dtype=np.uint8), repeat=0).read() is None
    with pytest.raises(CaptureError):
        ImageFrameSource.from_array(np.zeros((10, 10, 3), dtype=np.float32))

    # coerce_region & normalize_bgr validation edges
    with pytest.raises(CaptureError):
        coerce_region((0, 0, 10, 0))
    with pytest.raises(CaptureError):
        coerce_region((0, 0, 10))  # type: ignore[arg-type]
    with pytest.raises(CaptureError):
        normalize_bgr(np.zeros((16,), dtype=np.uint8))

    # BUG-03: draw_overlay accepts (H, W, 1) single-channel 3D uint8 grayscale
    gray_3d = np.full((100, 200, 1), 80, dtype=np.uint8)
    overlay = draw_overlay(gray_3d, DetectionResult.empty())
    assert overlay.shape == (100, 200, 3) and overlay.dtype == np.uint8


def test_t5_win32_sendinput_uipi_nofg_and_hotkey_default_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Win32SendInputMouse UIPI w/ require_foreground=False, unknown elevation, non-win32, HotkeyController."""
    sent_flags: list[int] = []

    def _rec_send(inp: Any) -> int:
        sent_flags.append(int(inp.mi.dwFlags))
        return 1

    # require_foreground=False with target_hwnd still blocks when UIPI elevation mismatch
    m_uipi = Win32SendInputMouse(
        require_foreground=False,
        target_hwnd=888,
        check_uipi=True,
        send_input_fn=_rec_send,
        is_target_elevated_fn=lambda _h: True,
        is_current_elevated_fn=lambda: False,
    )
    assert m_uipi.press_left(30.0, timestamp_s=1.0) is False
    assert m_uipi.uipi_blocked is True
    assert m_uipi.last_blocked_reason == "uipi_elevation_mismatch"
    assert sent_flags == []

    # Unknown target elevation (None) proceeds to send LEFTDOWN & LEFTUP
    m_unk = Win32SendInputMouse(
        require_foreground=False,
        target_hwnd=888,
        check_uipi=True,
        send_input_fn=_rec_send,
        is_target_elevated_fn=lambda _h: None,
        is_current_elevated_fn=lambda: False,
    )
    assert m_unk.press_left(30.0, timestamp_s=1.0) is True
    assert sent_flags == [MOUSEEVENTF_LEFTDOWN]
    m_unk.release_all()
    assert sent_flags == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    # Non-win32 _get_win32_send_input raises RuntimeError
    monkeypatch.setattr(sendinput_mod.sys, "platform", "linux")
    monkeypatch.setattr(sendinput_mod, "_USER32_SENDINPUT", None)
    with pytest.raises(RuntimeError, match="Windows"):
        _get_win32_send_input()

    # HotkeyController non-win32 returns no events without loading user32
    monkeypatch.setattr(hotkeys_mod.sys, "platform", "linux")
    monkeypatch.setattr(hotkeys_mod, "_GET_ASYNC_KEY_STATE", None)
    hk_linux = HotkeyController()
    assert hk_linux.poll() == HotkeyEvents(False, False)

    # HotkeyController win32 lazy default binding detects rising edges
    monkeypatch.setattr(hotkeys_mod.sys, "platform", "win32")
    monkeypatch.setattr(hotkeys_mod, "_GET_ASYNC_KEY_STATE", None)
    pressed_vks: set[int] = set()

    class _FakeUser32:
        class GetAsyncKeyState:
            argtypes: list[Any] = []
            restype: Any = None

            def __new__(cls, vk: int) -> int:
                return 0x8000 if vk in pressed_vks else 0

    monkeypatch.setattr(hotkeys_mod.ctypes, "WinDLL", lambda *_a, **_kw: _FakeUser32())
    hk_win = HotkeyController()
    assert hk_win.poll() == HotkeyEvents(False, False)
    pressed_vks.add(VK_F8)
    assert hk_win.poll() == HotkeyEvents(toggle_armed=True, emergency_stop=False)
    assert hk_win.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=False)
    pressed_vks.add(VK_ESCAPE)
    assert hk_win.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=True)


# ---------------------------------------------------------------------------
# Challenger 2 Adversarial Tests (1–10)
# ---------------------------------------------------------------------------


def test_adv_cross_strike_gold_shrink_no_silver_false_click() -> None:
    """BUG-1: Warm-locked Gold shrink (38->12 px) must invalidate stale _ref_gold_len and not false-click."""
    cfg = AppConfig()
    detector = GaugeDetector(cfg)
    sm = TriggerStateMachine(cfg)

    t = 1.0
    for rn in (-2.2, -1.5, 0.0, 1.5, 2.2):
        res = detector.detect(
            render_synthetic_gauge(gold_height_px=38.0, red_norm_pos=rn).image_bgr
        )
        assert res.gauge_present is True
        sm.update(res, t)
        t += 0.0167
    assert sm.trigger_count == 1

    for _ in range(15):
        res = detector.detect(
            render_synthetic_gauge(gold_height_px=38.0, red_norm_pos=2.2).image_bgr
        )
        sm.update(res, t)
        t += 0.0167

    for rn in (-2.2, -1.8, -1.5, -1.3, -1.5, -1.8, -2.2):
        res = detector.detect(
            render_synthetic_gauge(gold_height_px=12.0, red_norm_pos=rn).image_bgr
        )
        assert res.gauge_present is True
        assert res.gold_box is not None
        assert res.gold_box.size[1] <= 15.0
        assert res.norm_pos is not None and abs(res.norm_pos) > 1.0
        assert res.is_in_gold is False
        dec = sm.update(res, t)
        assert dec.should_click is False
        t += 0.0167

    assert sm.trigger_count == 1


def test_adv_coarse_distractor_does_not_mask_valid_gauge() -> None:
    """BUG-2: High-scoring coarse-only distractor (122px) must not mask true 24px gauge at (1300, 540)."""
    sample = render_synthetic_gauge(
        width=1920,
        height=1080,
        center=(1300.0, 540.0),
        gold_height_px=24.0,
        red_norm_pos=0.0,
        include_distractors=False,
    )
    img = sample.image_bgr.copy()
    img[150:272, 200:322] = SYNTH_GOLD_BGR
    img[150:190, 200:322] = SYNTH_SILVER_BGR
    img[232:272, 200:322] = SYNTH_SILVER_BGR

    res = GaugeDetector(AppConfig()).detect(img)
    assert res.gauge_present is True
    assert res.gold_box is not None
    assert math.hypot(res.gold_box.center[0] - 1300.0, res.gold_box.center[1] - 540.0) <= 10.0


def test_adv_cold_start_thick_red_abutting_gold_edge() -> None:
    """BUG-3: Cold start with 17px Red abutting Gold-Silver edge on 1080p narrow Gold must succeed."""
    cfg = AppConfig()
    for gh in (16.0, 24.0, 34.0):
        for rn in (-1.0, -1.3, -1.6):
            sample = render_synthetic_gauge(
                width=1920,
                height=1080,
                gold_height_px=gh,
                red_thickness_px=17.0,
                red_norm_pos=rn,
            )
            res = GaugeDetector(cfg).detect(sample.image_bgr)
            assert res.gauge_present is True, f"Failed cold start for gh={gh}, rn={rn}"
            assert res.red_box is not None


def test_adv_many_gold_distractors_do_not_starve_split_or_narrow_fallbacks() -> None:
    """BUG-4: >16 off-gauge Gold distractors must not starve pair_cids or narrow_cids."""
    cfg = AppConfig()

    s_split = render_synthetic_gauge(
        width=1920,
        height=1080,
        center=(1350.0, 540.0),
        gold_height_px=18.0,
        red_thickness_px=12.0,
        red_norm_pos=0.0,
        include_distractors=False,
    )
    img_a = s_split.image_bgr.copy()
    for i in range(18):
        y0 = 60 + (i // 6) * 45
        x0 = 60 + (i % 6) * 45
        img_a[y0 : y0 + 25, x0 : x0 + 25] = SYNTH_GOLD_BGR
    res_a = GaugeDetector(cfg).detect(img_a)
    assert res_a.gauge_present is True
    assert res_a.gold_box is not None
    assert math.hypot(res_a.gold_box.center[0] - 1350.0, res_a.gold_box.center[1] - 540.0) <= 10.0

    s_narrow = render_synthetic_gauge(
        width=1920,
        height=1080,
        center=(1350.0, 540.0),
        gold_height_px=14.0,
        red_thickness_px=10.0,
        red_norm_pos=-0.8,
        include_distractors=False,
    )
    img_b = s_narrow.image_bgr.copy()
    for i in range(18):
        y0 = 60 + (i // 6) * 50
        x0 = 60 + (i % 6) * 50
        img_b[y0 : y0 + 30, x0 : x0 + 30] = SYNTH_GOLD_BGR
    res_b = GaugeDetector(cfg).detect(img_b)
    assert res_b.gauge_present is True
    assert res_b.gold_box is not None
    assert math.hypot(res_b.gold_box.center[0] - 1350.0, res_b.gold_box.center[1] - 540.0) <= 10.0


def test_adv_480p_scaled_narrow_gold_and_1080p_thin_sliver_split() -> None:
    """BUG-5: 480p scaled narrow Gold + 1080p thin-sliver Red-split at odd row coordinates."""
    cfg = AppConfig()

    for w, h in ((640, 480), (854, 480)):
        s_480 = render_synthetic_gauge(
            width=w,
            height=h,
            center=(w / 2.0, h / 2.0),
            scale=480.0 / 1080.0,
            gold_height_px=12.0,
            red_norm_pos=0.0,
        )
        res_480 = GaugeDetector(cfg).detect(s_480.image_bgr)
        assert res_480.gauge_present is True
        assert res_480.red_box is not None

    for cy, gh, rt in ((411.0, 12.0, 11.0), (540.0, 10.0, 7.0)):
        s_thin = render_synthetic_gauge(
            width=1920,
            height=1080,
            center=(850.0, cy),
            gold_height_px=gh,
            red_thickness_px=rt,
            red_norm_pos=0.0,
        )
        res_thin = GaugeDetector(cfg).detect(s_thin.image_bgr)
        assert res_thin.gauge_present is True, f"Failed thin sliver at cy={cy}, gh={gh}, rt={rt}"
        assert res_thin.red_box is not None


def test_adv_state_machine_epoch_timestamp_exact_cooldown_expiry() -> None:
    """BUG-6: Unix-epoch timestamps expire cooldown reliably at exact boundary despite float64 ULP."""
    t0 = 1795156043.0
    for cd_ms in (100.0, 200.0, 300.0):
        sm = TriggerStateMachine(TriggerConfig(cooldown_ms=cd_ms))
        assert sm.update(_det(True, -1.5), t0).reason == "gauge_acquired"
        t_trig = t0 + 0.01
        trig_dec = sm.update(_det(True, 0.0), t_trig)
        assert trig_dec.should_click is True
        assert (
            sm.update(_det(True, 1.5), t0 + 0.02).reason == "exited_gold_waiting_cooldown"
        )
        assert sm.state is TriggerState.COOLDOWN

        t_expire = t_trig + cd_ms / 1000.0
        exp_dec = sm.update(_det(True, 1.5), t_expire)
        assert exp_dec.state is TriggerState.ARMED
        assert exp_dec.reason == "cooldown_elapsed"


def test_adv_runner_hold_exceeding_cooldown_remains_consistent() -> None:
    """BUG-7: Runner keeps res.clicks == spy.click_count == sm.trigger_count when hold_ms > cooldown_ms."""
    cfg = AppConfig().with_overrides(
        dry_run=True,
        require_foreground=False,
        hold_ms=260.0,
        cooldown_ms=200.0,
    )
    img_arm = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=-1.5).image_bgr
    img_gold = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=0.0).image_bgr
    img_exit = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=1.5).image_bgr

    schedule = [
        (0.00, img_arm),
        (0.02, img_gold),
        (0.04, img_exit),
        (0.23, img_exit),
        (0.25, img_gold),
        (0.29, img_gold),
    ]
    packets = [
        FramePacket(image_bgr=bgr, timestamp_s=ts, frame_idx=i)
        for i, (ts, bgr) in enumerate(schedule)
    ]

    class _ListSource:
        def __init__(self, items: list[FramePacket]) -> None:
            self.items = items
            self.i = 0

        def read(self) -> FramePacket | None:
            if self.i >= len(self.items):
                return None
            out = self.items[self.i]
            self.i += 1
            return out

        def close(self) -> None:
            pass

    mouse = SpyMouseBackend()
    sm = TriggerStateMachine(cfg)
    res = Runner(
        source=_ListSource(packets),
        mouse=mouse,
        config=cfg,
        state_machine=sm,
        stop_on_eof=True,
    ).run()
    assert res.clicks == mouse.click_count == sm.trigger_count
    assert res.clicks == 2
    assert mouse.is_pressed is False


def test_adv_runner_non_finite_packet_timestamp_handling() -> None:
    """BUG-8: Non-finite FramePacket.timestamp_s raises ValueError on frame 0 and frame 1+, releasing mouse."""
    sample_arm = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=-1.5).image_bgr
    sample_gold = render_synthetic_gauge(gold_height_px=34.0, red_norm_pos=0.0).image_bgr
    cfg = AppConfig().with_overrides(dry_run=True, require_foreground=False, hold_ms=100.0)

    class _ListSource:
        def __init__(self, items: list[FramePacket]) -> None:
            self.items = items
            self.i = 0

        def read(self) -> FramePacket | None:
            if self.i >= len(self.items):
                return None
            out = self.items[self.i]
            self.i += 1
            return out

        def close(self) -> None:
            pass

    for bad_ts in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            Runner(
                source=_ListSource([FramePacket(image_bgr=sample_arm, timestamp_s=bad_ts, frame_idx=0)]),
                mouse=SpyMouseBackend(),
                config=cfg,
                stop_on_eof=True,
            ).run()

        mouse = SpyMouseBackend()
        pkts = [
            FramePacket(image_bgr=sample_arm, timestamp_s=1.0, frame_idx=0),
            FramePacket(image_bgr=sample_gold, timestamp_s=1.02, frame_idx=1),
            FramePacket(image_bgr=sample_arm, timestamp_s=bad_ts, frame_idx=2),
        ]
        with pytest.raises(ValueError, match="finite"):
            Runner(
                source=_ListSource(pkts),
                mouse=mouse,
                config=cfg,
                stop_on_eof=True,
            ).run()
        assert mouse.is_pressed is False
        assert mouse.clicks[-1].released is True


def test_adv_detector_tilt_limit_65deg_array_flags_4k_and_res_switch() -> None:
    """GAP-1 + GAP-2: ±60/64 deg accepted vs ±66 deg rejected; 4x4, read-only, flipped, 4K, res switch."""
    cfg = AppConfig()

    for tilt in (-64.0, -60.0, 60.0, 64.0):
        s_ok = render_synthetic_gauge(
            width=1920, height=1080, tilt_deg=tilt, gold_height_px=28.0, red_norm_pos=0.0
        )
        r_ok = GaugeDetector(cfg).detect(s_ok.image_bgr)
        assert r_ok.gauge_present is True, f"Expected tilt {tilt} accepted, got {r_ok}"

    for tilt in (-66.0, 66.0):
        s_rej = render_synthetic_gauge(
            width=1920, height=1080, tilt_deg=tilt, gold_height_px=28.0, red_norm_pos=0.0
        )
        r_rej = GaugeDetector(cfg).detect(s_rej.image_bgr)
        assert r_rej.gauge_present is False, f"Expected tilt {tilt} rejected, got {r_rej}"

    tiny = np.full((4, 4, 3), SYNTH_BG_BGR, dtype=np.uint8)
    assert GaugeDetector(cfg).detect(tiny).gauge_present is False

    s_ro = render_synthetic_gauge(
        width=1280, height=720, center=(640.0, 360.0), gold_height_px=28.0, red_norm_pos=0.0
    )
    ro_img = s_ro.image_bgr.copy()
    ro_img.setflags(write=False)
    assert GaugeDetector(cfg).detect(ro_img).gauge_present is True

    flipped_img = s_ro.image_bgr[:, ::-1, :]
    assert not flipped_img.flags["C_CONTIGUOUS"]
    assert GaugeDetector(cfg).detect(flipped_img).gauge_present is True

    s_4k = render_synthetic_gauge(
        width=3840,
        height=2160,
        center=(1920.0, 1080.0),
        scale=2.0,
        gold_height_px=28.0,
        red_norm_pos=0.0,
    )
    r_4k = GaugeDetector(cfg).detect(s_4k.image_bgr)
    assert r_4k.gauge_present is True and r_4k.red_box is not None

    det_switch = GaugeDetector(cfg)
    s_1080 = render_synthetic_gauge(width=1920, height=1080, gold_height_px=28.0, red_norm_pos=0.0)
    assert det_switch.detect(s_1080.image_bgr).gauge_present is True
    s_480 = render_synthetic_gauge(
        width=854,
        height=480,
        center=(427.0, 240.0),
        scale=480.0 / 1080.0,
        gold_height_px=28.0,
        red_norm_pos=0.0,
    )
    assert det_switch.detect(s_480.image_bgr).gauge_present is True


def test_adv_state_machine_and_models_nonfinite_and_lost_preserve() -> None:
    """GAP-3 + GAP-4: DetectionResult.is_in_gold non-finite norm_pos & StateMachine lost_preserve/non-finite."""
    box = OrientedBox(center=(10.0, 10.0), size=(10.0, 10.0), angle_deg=0.0)
    for bad_pos in (float("nan"), float("inf"), float("-inf")):
        dr = DetectionResult(gauge_present=True, gold_box=box, red_box=box, norm_pos=bad_pos)
        assert dr.is_in_gold is False

    sm = TriggerStateMachine(TriggerConfig(rearm_lost_frames=3, cooldown_ms=200.0))
    assert sm.update(_det(True, -1.5), 1.0).state is TriggerState.ARMED

    nf_armed = sm.update(_det(True, float("nan")), 1.01)
    assert nf_armed.state is TriggerState.ARMED and nf_armed.should_click is False

    for idx in range(2):
        lost = sm.update(_det(False, None), 1.02 + idx * 0.01)
        assert lost.state is TriggerState.ARMED and lost.reason == "gauge_absent_preserved"

    assert sm.update(_det(True, 0.0), 1.05).should_click is True
    assert sm.update(_det(True, 1.5), 1.06).state is TriggerState.COOLDOWN

    for idx in range(2):
        lost_cd = sm.update(_det(False, None), 1.07 + idx * 0.01)
        assert lost_cd.state is TriggerState.COOLDOWN and lost_cd.reason == "gauge_absent_preserved"

    lost_3 = sm.update(_det(False, None), 1.09)
    assert lost_3.state is TriggerState.IDLE and lost_3.reason == "gauge_lost_rearm"

    nf_idle = sm.update(_det(True, float("inf")), 1.10)
    assert nf_idle.state is TriggerState.IDLE and nf_idle.should_click is False
