from __future__ import annotations

import ctypes
import math
import subprocess
import sys
from ctypes import wintypes
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import vrc_smith_auto
from vrc_smith_auto import (
    AppConfig,
    CaptureConfig,
    DetectionResult,
    FramePacket,
    FrameSource,
    GeometryConfig,
    HSVConfig,
    HSVRange,
    MouseBackend,
    OrientedBox,
    TriggerConfig,
    WindowConfig,
    WindowRect,
)
from vrc_smith_auto import window as window_mod
from tests.conftest import SYNTH_GOLD_BGR, SYNTH_RED_BGR, SYNTH_SILVER_BGR


# ---------------------------------------------------------------------------
# Contracts, Protocols, Vision Models & Config Tests
# ---------------------------------------------------------------------------


def test_package_exports_and_version() -> None:
    assert vrc_smith_auto.__version__ == "0.1.0"
    for name in (
        "AppConfig",
        "CaptureConfig",
        "DetectionResult",
        "FramePacket",
        "FrameSource",
        "GeometryConfig",
        "HSVConfig",
        "HSVRange",
        "MouseBackend",
        "OrientedBox",
        "TriggerConfig",
        "WindowConfig",
        "WindowRect",
    ):
        assert hasattr(vrc_smith_auto, name), f"Missing public export: {name}"

    # Importing top-level package must NOT initialize Win32 DPI awareness
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vrc_smith_auto; from vrc_smith_auto import window; assert window._WIN32_API is None",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_frame_packet_and_protocols() -> None:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pkt = FramePacket(image_bgr=img, timestamp_s=12.5, frame_idx=7)
    assert pkt.height == 1080
    assert pkt.width == 1920
    assert pkt.timestamp_s == 12.5
    assert pkt.frame_idx == 7

    with pytest.raises(FrozenInstanceError):
        pkt.frame_idx = 8  # type: ignore[misc]

    class GoodSource:
        def read(self) -> FramePacket | None:
            return pkt

        def close(self) -> None:
            return None

    class BadSource:
        def read(self) -> FramePacket | None:
            return None

    assert isinstance(GoodSource(), FrameSource)
    assert not isinstance(BadSource(), FrameSource)

    class PropMouse:
        @property
        def is_pressed(self) -> bool:
            return False

        def press_left(self, duration_ms: float, timestamp_s: float | None = None) -> bool:
            return True

        def tick(self, now_s: float) -> None:
            return None

        def release_all(self) -> None:
            return None

    class AttrMouse:
        is_pressed = False

        def press_left(self, duration_ms: float, timestamp_s: float | None = None) -> bool:
            return True

        def tick(self, now_s: float) -> None:
            return None

        def release_all(self) -> None:
            return None

    class BadMouse:
        @property
        def is_pressed(self) -> bool:
            return False

        def press_left(self, duration_ms: float, timestamp_s: float | None = None) -> bool:
            return True

    assert isinstance(PropMouse(), MouseBackend)
    assert isinstance(AttrMouse(), MouseBackend)
    assert not isinstance(BadMouse(), MouseBackend)


def test_vision_models_and_is_in_gold() -> None:
    box = OrientedBox(center=(960.0, 540.0), size=(38.0, 28.0), angle_deg=-3.5)
    assert box.area == pytest.approx(38.0 * 28.0)

    empty = DetectionResult.empty(roi_used=(10, 20, 300, 400))
    assert empty.gauge_present is False
    assert empty.confidence == 0.0
    assert empty.roi_used == (10, 20, 300, 400)
    assert empty.red_detected is False
    assert empty.is_in_gold is False

    for pos, expected in [
        (-1.0, True),
        (0.0, True),
        (1.0, True),
        (-1.01, False),
        (1.01, False),
        (None, False),
    ]:
        res = DetectionResult(
            gauge_present=True,
            gold_box=box,
            red_box=OrientedBox((960.0, 540.0), (62.0, 7.0), -3.5) if pos is not None else None,
            red_proj=540.0 if pos is not None else None,
            norm_pos=pos,
        )
        assert res.is_in_gold is expected
        assert res.red_detected is (pos is not None)

    lost = DetectionResult(gauge_present=False, norm_pos=0.0)
    assert lost.is_in_gold is False


def test_config_defaults_match_empirical_baseline(default_config: AppConfig) -> None:
    assert default_config.capture_backend == "auto"
    assert default_config.window_title == "VRChat"
    assert default_config.require_foreground is True
    assert default_config.hold_ms == 1.0
    assert default_config.cooldown_ms == 200.0
    assert default_config.entry_margin == 0.95
    assert default_config.exit_margin == 1.25
    assert default_config.lead_ms == 8.3
    assert default_config.trigger.max_lead_offset == 2.0
    assert default_config.dry_run is False

    # Verify default HSV bounds match synthetic empirical Survey 2 BGR colors
    def _to_hsv(bgr: tuple[int, int, int]) -> np.ndarray:
        patch = np.full((4, 4, 3), bgr, dtype=np.uint8)
        return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    gold_lo, gold_hi = default_config.hsv.gold.to_bounds()
    assert int(cv2.inRange(_to_hsv(SYNTH_GOLD_BGR), gold_lo, gold_hi).min()) == 255

    silver_lo, silver_hi = default_config.hsv.silver.to_bounds()
    assert int(cv2.inRange(_to_hsv(SYNTH_SILVER_BGR), silver_lo, silver_hi).min()) == 255

    red_hsv = _to_hsv(SYNTH_RED_BGR)
    r_lo1, r_hi1 = default_config.hsv.red_low.to_bounds()
    r_lo2, r_hi2 = default_config.hsv.red_high.to_bounds()
    red_mask = cv2.inRange(red_hsv, r_lo1, r_hi1) | cv2.inRange(red_hsv, r_lo2, r_hi2)
    assert int(red_mask.min()) == 255


def test_config_yaml_json_roundtrip_and_overrides(tmp_path: Path, default_config: AppConfig) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yml_path = tmp_path / "cfg.yml"
    json_path = tmp_path / "cfg.json"
    empty_yaml = tmp_path / "empty.yaml"
    bad_ext = tmp_path / "cfg.toml"

    custom = default_config.with_overrides(
        {
            "capture_backend": "mss",
            "trigger.hold_ms": 35.0,
            "window": {"require_foreground": False},
            "hsv.gold.h_min": 16,
        },
        dry_run=True,
    )
    assert custom.capture_backend == "mss"
    assert custom.hold_ms == 35.0
    assert custom.require_foreground is False
    assert custom.dry_run is True
    assert custom.hsv.gold.h_min == 16

    yaml_path.write_text(custom.to_yaml(), encoding="utf-8")
    yml_path.write_text(custom.to_yaml(), encoding="utf-8")
    json_path.write_text(custom.to_json(), encoding="utf-8")
    empty_yaml.write_text("", encoding="utf-8")
    bad_ext.write_text("hold_ms = 30\n", encoding="utf-8")

    assert AppConfig.from_file(yaml_path) == custom
    assert AppConfig.from_file(yml_path) == custom
    assert AppConfig.from_file(json_path) == custom
    assert AppConfig.from_file(empty_yaml) == AppConfig()

    with pytest.raises(ValueError):
        AppConfig.from_file(bad_ext)
    with pytest.raises(ValueError):
        AppConfig.from_dict({"unknown_key": 123})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"trigger": {"unknown_nested": 1}})
    with pytest.raises(ValueError):
        AppConfig.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TriggerConfig(hold_ms=0.0),
        lambda: TriggerConfig(hold_ms=-5.0),
        lambda: TriggerConfig(hold_ms=math.nan),
        lambda: TriggerConfig(hold_ms=math.inf),
        lambda: TriggerConfig(hold_ms=True),  # type: ignore[arg-type]
        lambda: TriggerConfig(cooldown_ms=-1.0),
        lambda: TriggerConfig(entry_margin=0.0),
        lambda: TriggerConfig(entry_margin=1.1, exit_margin=1.0),
        lambda: TriggerConfig(lead_ms=-0.5),
        lambda: TriggerConfig(max_lead_offset=-0.1),
        lambda: TriggerConfig(rearm_lost_frames=0),
        lambda: CaptureConfig(backend="invalid"),  # type: ignore[arg-type]
        lambda: CaptureConfig(target_fps=0),
        lambda: CaptureConfig(monitor_idx=-1),
        lambda: WindowConfig(title="   "),
        lambda: WindowConfig(rect_refresh_s=0.0),
        lambda: HSVRange(25, 20, 0, 255, 0, 255),
        lambda: HSVRange(-1, 20, 0, 255, 0, 255),
        lambda: HSVRange(0, 180, 0, 255, 0, 255),
        lambda: HSVRange(0, 20, 0, 256, 0, 255),
        lambda: HSVRange(True, 20, 0, 255, 0, 255),  # type: ignore[arg-type]
        lambda: GeometryConfig(min_gold_width_px=50.0, max_gold_width_px=40.0),
        lambda: GeometryConfig(min_gold_fill_ratio=0.0),
        lambda: GeometryConfig(min_gold_fill_ratio=1.1),
        lambda: GeometryConfig(acquire_downscale=0.0),
        lambda: GeometryConfig(acquire_downscale=1.5),
        lambda: GeometryConfig(max_lost_frames=0),
    ],
)
def test_config_validation_errors(factory: Any) -> None:
    with pytest.raises(ValueError):
        factory()


def test_synthetic_gauge_fixture_tilted_and_split(
    synthetic_gauge_factory: Any,
    default_config: AppConfig,
) -> None:
    hsv_cfg = default_config.hsv
    for tilt in (-30.0, -3.0, 25.0):
        sample = synthetic_gauge_factory(
            tilt_deg=tilt,
            red_norm_pos=0.0,
            include_distractors=True,
        )
        assert sample.image_bgr.shape == (1080, 1920, 3)
        assert sample.image_bgr.dtype == np.uint8
        assert sample.gauge_present is True
        assert sample.expected_norm_pos == pytest.approx(0.0)
        assert sample.expected_tilt_deg == pytest.approx(tilt)

        hsv = cv2.cvtColor(sample.image_bgr, cv2.COLOR_BGR2HSV)
        gold_mask = cv2.inRange(hsv, *hsv_cfg.gold.to_bounds())
        silver_mask = cv2.inRange(hsv, *hsv_cfg.silver.to_bounds())
        red_mask = cv2.inRange(hsv, *hsv_cfg.red_low.to_bounds()) | cv2.inRange(
            hsv, *hsv_cfg.red_high.to_bounds()
        )
        assert int(np.count_nonzero(gold_mask)) > 100
        assert int(np.count_nonzero(silver_mask)) > 1000
        assert int(np.count_nonzero(red_mask)) > 100

    empty_sample = synthetic_gauge_factory(gauge_present=False, include_distractors=False)
    assert empty_sample.gauge_present is False
    assert empty_sample.expected_norm_pos is None
    assert int(np.count_nonzero(empty_sample.image_bgr[:, :, 0] != 12)) == 0


# ---------------------------------------------------------------------------
# WindowRect, Mocked Win32 & Real Windows Integration Tests
# ---------------------------------------------------------------------------


def test_window_rect_geometry_and_helpers() -> None:
    rect = WindowRect.from_xywh(100, 200, 1280, 720, hwnd=42)
    assert rect.left == 100
    assert rect.top == 200
    assert rect.right == 1380
    assert rect.bottom == 920
    assert rect.width == 1280
    assert rect.height == 720
    assert rect.is_valid is True
    assert rect.as_ltrb() == (100, 200, 1380, 920)
    assert rect.as_xywh() == (100, 200, 1280, 720)
    assert rect.as_mss_monitor() == {"left": 100, "top": 200, "width": 1280, "height": 720}

    inv = WindowRect(left=50, top=50, right=10, bottom=50)
    assert inv.width == 0
    assert inv.height == 0
    assert inv.is_valid is False

    primary = WindowRect(0, 0, 1920, 1080)
    partial = WindowRect(-50, 100, 1000, 1200, hwnd=7)
    clamped = partial.clamp_to_bounds(primary)
    assert clamped == WindowRect(0, 100, 1000, 1080, hwnd=7)
    assert partial.to_monitor_relative(primary) == (0, 100, 1000, 1080)

    secondary = WindowRect(-1920, 0, 0, 1080)
    on_sec = WindowRect(-1800, 120, -200, 1000, hwnd=9)
    assert on_sec.to_monitor_relative(secondary) == (120, 120, 1720, 1000)

    disjoint = WindowRect(2000, 0, 2500, 500)
    assert disjoint.clamp_to_bounds(primary) is None
    assert disjoint.to_monitor_relative(primary) is None


class _FakeWin32Builder:
    """In-memory Win32 API stub matching _Win32Api attributes used by window.py."""

    def __init__(self, windows: list[dict[str, Any]] | None = None, foreground_hwnd: int | None = None) -> None:
        self.windows = {w["hwnd"]: dict(w) for w in (windows or [])}
        self.order = [w["hwnd"] for w in (windows or [])]
        self.foreground_hwnd = foreground_hwnd
        self.last_error = 0
        self.dpi_context_ret = 1
        self.dpi_context_err = 0
        self.shcore_hr = 0
        self.dpi_aware_ret = 1
        self.closed_handles: list[int] = []
        self.next_handle = 1000
        self.open_proc_err = 0
        self.open_tok_err = 0
        self.current_elevated = False

        api = self

        class _User32:
            def __init__(self) -> None:
                self.SetProcessDpiAwarenessContext = self._set_dpi_awareness_context

            def _set_dpi_awareness_context(self, _ctx: Any) -> int:
                api.set_last_error(api.dpi_context_err)
                return api.dpi_context_ret

            def SetProcessDPIAware(self) -> int:
                return api.dpi_aware_ret

            def EnumWindows(self, cb: Any, lparam: int) -> int:
                api.set_last_error(0)
                if not api.order:
                    return 0
                for h in api.order:
                    if not cb(h, lparam):
                        break
                return 1

            def IsWindow(self, hwnd: int) -> int:
                return 1 if hwnd in api.windows else 0

            def IsWindowVisible(self, hwnd: int) -> int:
                return 1 if api.windows.get(hwnd, {}).get("visible", True) else 0

            def IsIconic(self, hwnd: int) -> int:
                return 1 if api.windows.get(hwnd, {}).get("iconic", False) else 0

            def GetWindowTextLengthW(self, hwnd: int) -> int:
                return len(api.windows.get(hwnd, {}).get("title", ""))

            def GetWindowTextW(self, hwnd: int, buf: Any, max_len: int) -> int:
                title = api.windows.get(hwnd, {}).get("title", "")[: max_len - 1]
                buf.value = title
                return len(title)

            def GetClassNameW(self, hwnd: int, buf: Any, max_len: int) -> int:
                cls = api.windows.get(hwnd, {}).get("class_name", "")[: max_len - 1]
                buf.value = cls
                return len(cls)

            def GetClientRect(self, hwnd: int, rc_ptr: Any) -> int:
                w = api.windows.get(hwnd)
                if w is None:
                    return 0
                l, t, r, b = w.get("client_rect", (0, 0, 1280, 720))
                rc = rc_ptr._obj
                rc.left, rc.top, rc.right, rc.bottom = l, t, r, b
                return 1

            def ClientToScreen(self, hwnd: int, pt_ptr: Any) -> int:
                w = api.windows.get(hwnd)
                if w is None:
                    return 0
                sx, sy = w.get("screen_origin", (100, 50))
                pt = pt_ptr._obj
                pt.x += sx
                pt.y += sy
                return 1

            def MonitorFromWindow(self, hwnd: int, _flags: int) -> int:
                return 777 if hwnd in api.windows else 0

            def GetMonitorInfoW(self, hmon: int, mi_ptr: Any) -> int:
                if not hmon:
                    return 0
                mi = mi_ptr._obj
                mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom = (0, 0, 1920, 1080)
                return 1

            def GetForegroundWindow(self) -> int | None:
                return api.foreground_hwnd

            def GetWindowThreadProcessId(self, hwnd: int, pid_ptr: Any) -> int:
                w = api.windows.get(hwnd)
                pid = w.get("pid", 0) if w else 0
                pid_ptr._obj.value = pid
                return 1 if pid else 0

        class _Shcore:
            def SetProcessDpiAwareness(self, _val: int) -> int:
                return api.shcore_hr

        class _Kernel32:
            def GetCurrentProcess(self) -> int:
                return 999

            def OpenProcess(self, _access: int, _inherit: bool, pid: int) -> int:
                for w in api.windows.values():
                    if w.get("pid") == pid:
                        err = w.get("open_proc_err", api.open_proc_err)
                        if err:
                            api.set_last_error(err)
                            return 0
                        api.next_handle += 1
                        h = api.next_handle
                        api._handle_pid[h] = pid
                        return h
                api.set_last_error(87)
                return 0

            def QueryFullProcessImageNameW(self, hproc: int, _flags: int, buf: Any, size_ptr: Any) -> int:
                pid = api._handle_pid.get(hproc)
                for w in api.windows.values():
                    if w.get("pid") == pid:
                        img = w.get("image_path", "")
                        buf.value = img
                        size_ptr._obj.value = len(img)
                        return 1 if img else 0
                return 0

            def CloseHandle(self, h: int) -> int:
                api.closed_handles.append(h)
                return 1

        class _Advapi32:
            def OpenProcessToken(self, hproc: int, _access: int, tok_ptr: Any) -> int:
                if hproc == 999:
                    api.next_handle += 1
                    tok_ptr._obj.value = api.next_handle
                    api._tok_elev[api.next_handle] = api.current_elevated
                    return 1
                pid = api._handle_pid.get(hproc)
                for w in api.windows.values():
                    if w.get("pid") == pid:
                        err = w.get("open_tok_err", api.open_tok_err)
                        if err:
                            api.set_last_error(err)
                            return 0
                        api.next_handle += 1
                        tok = api.next_handle
                        tok_ptr._obj.value = tok
                        api._tok_elev[tok] = bool(w.get("elevated", False))
                        return 1
                return 0

            def GetTokenInformation(self, htok: Any, _cls: int, buf_ptr: Any, _sz: int, ret_ptr: Any) -> int:
                tok_val = int(htok.value) if hasattr(htok, "value") else int(htok)
                elev_val = api._tok_elev.get(tok_val, False)
                buf_ptr._obj.TokenIsElevated = 1 if elev_val else 0
                ret_ptr._obj.value = 4
                return 1

        self._handle_pid: dict[int, int] = {}
        self._tok_elev: dict[int, bool] = {}
        self.user32 = _User32()
        self.shcore = _Shcore()
        self.kernel32 = _Kernel32()
        self.advapi32 = _Advapi32()

    def set_last_error(self, err: int) -> None:
        self.last_error = err


def _patch_win32(monkeypatch: pytest.MonkeyPatch, fake: _FakeWin32Builder) -> None:
    monkeypatch.setattr(window_mod.sys, "platform", "win32")
    monkeypatch.setattr(window_mod, "_get_win32", lambda: fake)
    monkeypatch.setattr(window_mod.ctypes, "set_last_error", fake.set_last_error)
    monkeypatch.setattr(window_mod.ctypes, "get_last_error", lambda: fake.last_error)


def test_enable_dpi_awareness_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWin32Builder()
    _patch_win32(monkeypatch, fake)

    # PMv2 direct success
    assert window_mod.enable_dpi_awareness() is True

    # Already locked -> ERROR_ACCESS_DENIED (5) is idempotent True
    fake.dpi_context_ret = 0
    fake.dpi_context_err = 5
    assert window_mod.enable_dpi_awareness() is True

    # PMv2 unsupported -> falls back to shcore (0 and E_ACCESSDENIED)
    delattr(fake.user32, "SetProcessDpiAwarenessContext")
    fake.shcore_hr = 0
    assert window_mod.enable_dpi_awareness() is True
    fake.shcore_hr = -2147024891  # 0x80070005 signed c_long
    assert window_mod.enable_dpi_awareness() is True

    # shcore absent -> falls back to SetProcessDPIAware
    fake.shcore = None
    fake.dpi_aware_ret = 1
    assert window_mod.enable_dpi_awareness() is True

    # Non-Windows returns False
    monkeypatch.setattr(window_mod.sys, "platform", "linux")
    assert window_mod.enable_dpi_awareness() is False


def test_find_vrchat_window_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty desktop where EnumWindows returns 0 with err=0
    empty_fake = _FakeWin32Builder(windows=[])
    _patch_win32(monkeypatch, empty_fake)
    assert window_mod.find_vrchat_window() is None

    # Rejects VS Code / Terminal / invisible / minimized; prefers VRChat.exe + exact title
    windows = [
        {
            "hwnd": 101,
            "visible": True,
            "iconic": False,
            "title": "VRChat - Visual Studio Code",
            "class_name": "Chrome_WidgetWin_1",
            "pid": 1001,
            "image_path": r"C:\Programs\Code.exe",
        },
        {
            "hwnd": 102,
            "visible": False,
            "iconic": False,
            "title": "VRChat",
            "class_name": "UnityWndClass",
            "pid": 1002,
            "image_path": r"C:\Games\VRChat\VRChat.exe",
        },
        {
            "hwnd": 103,
            "visible": True,
            "iconic": True,
            "title": "VRChat",
            "class_name": "UnityWndClass",
            "pid": 1003,
            "image_path": r"C:\Games\VRChat\VRChat.exe",
        },
        {
            "hwnd": 104,
            "visible": True,
            "iconic": False,
            "title": "VRChat",
            "class_name": "UnityWndClass",
            "pid": 1004,
            "image_path": r"C:\Games\VRChat\VRChat.exe",
        },
    ]
    fake = _FakeWin32Builder(windows=windows)
    _patch_win32(monkeypatch, fake)

    assert window_mod.find_vrchat_window() == 104
    # Minimized window matched only when include_minimized=True and live window absent
    fake.windows[104]["visible"] = False
    assert window_mod.find_vrchat_window() is None
    assert window_mod.find_vrchat_window(include_minimized=True) == 103

    # Custom class_name=None, process_name=None allows title-only matching
    assert (
        window_mod.find_vrchat_window(
            "Visual Studio Code",
            class_name=None,
            process_name=None,
        )
        == 101
    )


def test_get_client_rect_and_foreground_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWin32Builder(
        windows=[
            {
                "hwnd": 201,
                "visible": True,
                "iconic": False,
                "client_rect": (0, 0, 1280, 720),
                "screen_origin": (120, 80),
            },
            {
                "hwnd": 202,
                "visible": True,
                "iconic": True,
                "client_rect": (0, 0, 0, 0),
                "screen_origin": (0, 1052),
            },
            {
                "hwnd": 203,
                "visible": True,
                "iconic": False,
                "client_rect": (0, 0, 0, 0),
                "screen_origin": (10, 10),
            },
        ],
        foreground_hwnd=None,
    )
    _patch_win32(monkeypatch, fake)

    assert window_mod.get_client_rect(0) is None
    assert window_mod.get_client_rect(9999) is None
    assert window_mod.get_client_rect(202) is None
    assert window_mod.get_client_rect(203) is None

    rect = window_mod.get_client_rect(201)
    assert rect == WindowRect(left=120, top=80, right=1400, bottom=800, hwnd=201)
    assert window_mod.get_monitor_rect(201) == WindowRect(0, 0, 1920, 1080, hwnd=201)

    assert window_mod.is_foreground_window(0) is False
    assert window_mod.is_foreground_window(201) is False
    fake.foreground_hwnd = 201
    assert window_mod.is_foreground_window(201) is True
    assert window_mod.is_foreground_window(202) is False


def test_is_target_elevated_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWin32Builder(
        windows=[
            {"hwnd": 301, "pid": 4001, "open_proc_err": 5},  # Protected/System -> True
            {"hwnd": 302, "pid": 4002, "open_proc_err": 87},  # Dead/invalid PID -> None
            {"hwnd": 303, "pid": 4003, "open_tok_err": 5},  # Elevated token denied -> True
            {"hwnd": 304, "pid": 4004, "elevated": True},  # Elevated token readable -> True
            {"hwnd": 305, "pid": 4005, "elevated": False},  # Normal process -> False
            {"hwnd": 306, "pid": 0},  # PID 0 -> None
        ]
    )
    _patch_win32(monkeypatch, fake)

    assert window_mod.is_target_elevated(0) is None
    assert window_mod.is_target_elevated(301) is True
    assert window_mod.is_target_elevated(302) is None
    assert window_mod.is_target_elevated(303) is True
    assert window_mod.is_target_elevated(304) is True
    assert window_mod.is_target_elevated(305) is False
    assert window_mod.is_target_elevated(306) is None
    # Verify handles opened for 303, 304, 305 were properly closed
    assert len(fake.closed_handles) >= 5


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live ctypes test")
def test_windows_live_dpi_and_null_guards() -> None:
    assert window_mod.enable_dpi_awareness() is True
    assert window_mod.enable_dpi_awareness() is True
    assert window_mod.get_client_rect(0) is None
    assert window_mod.get_monitor_rect(0) is None
    assert window_mod.is_foreground_window(0) is False
    assert window_mod.is_target_elevated(0) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only live HWND test")
def test_windows_ephemeral_hwnd_integration() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    ws_overlappedwindow = 0x00CF0000
    ws_visible = 0x10000000
    title = "VRChatTestDummy"

    hwnd_ptr = user32.CreateWindowExW(
        0,
        "STATIC",
        title,
        ws_overlappedwindow | ws_visible,
        40,
        40,
        320,
        240,
        None,
        None,
        kernel32.GetModuleHandleW(None),
        None,
    )
    assert hwnd_ptr, f"CreateWindowExW failed with last_error={ctypes.get_last_error()}"
    hwnd = int(hwnd_ptr)
    try:
        rect = window_mod.get_client_rect(hwnd)
        assert rect is not None
        assert rect.is_valid
        assert rect.width > 0 and rect.height > 0
        assert rect.hwnd == hwnd

        mon_rect = window_mod.get_monitor_rect(hwnd)
        assert mon_rect is not None and mon_rect.is_valid

        assert (
            window_mod.find_vrchat_window(
                title,
                class_name="Static",
                process_name=None,
            )
            == hwnd
        )
        assert window_mod.find_vrchat_window(title, class_name="UnityWndClass") is None
        assert window_mod.is_target_elevated(hwnd) == window_mod.is_current_process_elevated()
    finally:
        user32.DestroyWindow(hwnd_ptr)
