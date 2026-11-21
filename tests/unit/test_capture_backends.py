from __future__ import annotations

import ctypes
import subprocess
import sys
import warnings
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from vrc_smith_auto.capture import (
    CaptureError,
    DxcamCapture,
    GdiBitBltCapture,
    ImageFrameSource,
    MssCapture,
    VideoFileFrameSource,
    WindowRegionTracker,
    create_live_capture,
    normalize_bgr,
)
from vrc_smith_auto.capture import base as capture_base
from vrc_smith_auto.config import AppConfig, CaptureConfig, WindowConfig
from vrc_smith_auto.protocols import FramePacket, FrameSource
from vrc_smith_auto.window import WindowRect


# ---------------------------------------------------------------------------
# Helpers & Test Doubles
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 100.0, step: float = 0.016) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        val = self.now
        self.now += self.step
        return val


class _DummyCOMError(Exception):
    pass


_DummyCOMError.__name__ = "COMError"


class _DummyScreenShotError(Exception):
    pass


_DummyScreenShotError.__name__ = "ScreenShotError"


class _FakeCamera:
    def __init__(
        self,
        *,
        left: int = 0,
        top: int = 0,
        width: int = 1920,
        height: int = 1080,
        frames: list[np.ndarray | None] | None = None,
        grab_exc: BaseException | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self._output = SimpleNamespace(
            desc=SimpleNamespace(
                DesktopCoordinates=SimpleNamespace(
                    left=left,
                    top=top,
                    right=left + width,
                    bottom=top + height,
                )
            )
        )
        self._frames = list(frames) if frames is not None else []
        self._grab_exc = grab_exc
        self.grab_calls: list[tuple[Any, bool, bool]] = []
        self.release_calls = 0
        self.stop_calls = 0
        self.is_capturing = False

    def grab(
        self,
        region: tuple[int, int, int, int] | None = None,
        copy: bool = True,
        new_frame_only: bool = True,
    ) -> np.ndarray | None:
        self.grab_calls.append((region, copy, new_frame_only))
        if self._grab_exc is not None:
            raise self._grab_exc
        if not self._frames:
            return None
        return self._frames.pop(0)

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_capturing = False

    def release(self) -> None:
        self.release_calls += 1


class _FakeDxcamModule:
    def __init__(
        self,
        outputs: list[list[Any]] | None = None,
        *,
        output_metadata: dict[str, list[Any]] | None = None,
        use_double_underscore: bool = True,
    ) -> None:
        if outputs is not None:
            factory = SimpleNamespace(
                outputs=outputs,
                output_metadata=output_metadata if output_metadata is not None else {},
            )
            setattr(self, "__factory" if use_double_underscore else "_factory", factory)
        self.create_calls: list[dict[str, Any]] = []
        self.behaviors: dict[tuple[int, int | None], Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        key = (kwargs.get("device_idx", 0), kwargs.get("output_idx"))
        outcome = self.behaviors.get(key)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome


class _FakeMssShot:
    def __init__(self, bgra: np.ndarray) -> None:
        self._bgra = bgra

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self._bgra
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


class _FakeMss:
    def __init__(
        self,
        monitors: list[dict[str, Any]],
        *,
        primary_monitor: dict[str, Any] | None = None,
        shots: list[np.ndarray] | None = None,
        grab_exc: BaseException | None = None,
    ) -> None:
        self.monitors = monitors
        if primary_monitor is not None:
            self.primary_monitor = primary_monitor
        self._shots = list(shots) if shots is not None else []
        self._grab_exc = grab_exc
        self.grab_rects: list[dict[str, int]] = []
        self.close_calls = 0

    def grab(self, monitor: dict[str, int]) -> _FakeMssShot:
        self.grab_rects.append(dict(monitor))
        if self._grab_exc is not None:
            raise self._grab_exc
        bgra = self._shots.pop(0)
        return _FakeMssShot(bgra)

    def close(self) -> None:
        self.close_calls += 1


class _FakeGdiApi:
    def __init__(
        self,
        *,
        monitors: list[tuple[WindowRect, bool]] | None = None,
        src_dc: int = 1001,
        mem_dc: int = 2002,
        hbitmap: int = 3003,
        old_obj: int = 4004,
        bitblt_ok: bool = True,
        scanlines: int | None = None,
        last_error: int = 0,
        pixel_bgra: tuple[int, int, int, int] = (16, 32, 48, 255),
    ) -> None:
        self._monitors = (
            [(WindowRect(0, 0, 1280, 720), True)] if monitors is None else monitors
        )
        self.last_error = last_error
        self.events: list[tuple[Any, ...]] = []
        self.captured_bmi: tuple[int, int, int] | None = None

        outer = self

        class _User32:
            def GetDC(self, hwnd: Any) -> int:
                outer.events.append(("GetDC", hwnd))
                return src_dc

            def ReleaseDC(self, hwnd: Any, hdc: int) -> int:
                outer.events.append(("ReleaseDC", hwnd, hdc))
                return 1

            def GetSystemMetrics(self, idx: int) -> int:
                return 1920 if idx == 0 else 1080

        class _Gdi32:
            def CreateCompatibleDC(self, hdc: int) -> int:
                outer.events.append(("CreateCompatibleDC", hdc))
                return mem_dc

            def DeleteDC(self, hdc: int) -> int:
                outer.events.append(("DeleteDC", hdc))
                return 1

            def CreateCompatibleBitmap(self, hdc: int, w: int, h: int) -> int:
                outer.events.append(("CreateCompatibleBitmap", hdc, w, h))
                return hbitmap

            def SelectObject(self, hdc: int, obj: int) -> int:
                outer.events.append(("SelectObject", hdc, obj))
                return old_obj

            def DeleteObject(self, obj: int) -> int:
                outer.events.append(("DeleteObject", obj))
                return 1

            def BitBlt(
                self,
                dst_dc: int,
                x: int,
                y: int,
                w: int,
                h: int,
                src: int,
                sx: int,
                sy: int,
                rop: int,
            ) -> int:
                outer.events.append(("BitBlt", dst_dc, x, y, w, h, src, sx, sy, rop))
                return 1 if bitblt_ok else 0

            def GetDIBits(
                self,
                hdc: int,
                hbmp: int,
                start: int,
                lines: int,
                lpv_bits: Any,
                lp_bmi: Any,
                usage: int,
            ) -> int:
                outer.events.append(("GetDIBits", hdc, hbmp, start, lines, usage))
                bmi_ptr = ctypes.cast(
                    lp_bmi, ctypes.POINTER(capture_gdi_bmi_type())
                )
                hdr = bmi_ptr.contents.bmiHeader
                outer.captured_bmi = (int(hdr.biWidth), int(hdr.biHeight), int(hdr.biBitCount))
                w = int(hdr.biWidth)
                h = abs(int(hdr.biHeight))
                out_lines = h if scanlines is None else scanlines
                if out_lines > 0:
                    fill = np.empty((h, w, 4), dtype=np.uint8)
                    fill[:, :] = pixel_bgra
                    raw_bytes = fill.tobytes()
                    ctypes.memmove(lpv_bits, raw_bytes, len(raw_bytes))
                return out_lines

        self.user32 = _User32()
        self.gdi32 = _Gdi32()

    def enum_monitors(self) -> list[tuple[WindowRect, bool]]:
        return list(self._monitors)


def capture_gdi_bmi_type() -> Any:
    from vrc_smith_auto.capture.gdi_source import _BITMAPINFO

    return _BITMAPINFO


# ---------------------------------------------------------------------------
# normalize_bgr & WindowRegionTracker
# ---------------------------------------------------------------------------


def test_normalize_bgr_formats_and_validation() -> None:
    gray_2d = np.full((8, 10), 120, dtype=np.uint8)
    out_2d = normalize_bgr(gray_2d)
    assert out_2d.shape == (8, 10, 3)
    assert out_2d.dtype == np.uint8
    assert out_2d.flags["C_CONTIGUOUS"]
    assert np.all(out_2d == 120)

    gray_hw1 = np.full((8, 10, 1), 77, dtype=np.uint8)
    out_hw1 = normalize_bgr(gray_hw1)
    assert out_hw1.shape == (8, 10, 3)
    assert out_hw1.flags["C_CONTIGUOUS"]
    assert np.all(out_hw1 == 77)

    bgra = np.zeros((6, 7, 4), dtype=np.uint8)
    bgra[:, :] = (11, 22, 33, 255)
    non_contig = bgra[:, :, :3]
    assert not non_contig.flags["C_CONTIGUOUS"]

    out_bgr = normalize_bgr(non_contig)
    assert out_bgr.shape == (6, 7, 3)
    assert out_bgr.flags["C_CONTIGUOUS"]
    assert tuple(out_bgr[0, 0]) == (11, 22, 33)

    out_bgra = normalize_bgr(bgra)
    assert out_bgra.shape == (6, 7, 3)
    assert out_bgra.flags["C_CONTIGUOUS"]
    assert tuple(out_bgra[0, 0]) == (11, 22, 33)

    # (1, 1, 4) slice is already C_CONTIGUOUS; normalize_bgr must still return a non-aliasing copy
    bgra_1x1 = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
    out_1x1 = normalize_bgr(bgra_1x1)
    assert out_1x1.shape == (1, 1, 3)
    assert out_1x1.dtype == np.uint8
    assert out_1x1.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(out_1x1, bgra_1x1)
    bgra_1x1[:] = 0
    assert tuple(out_1x1[0, 0]) == (10, 20, 30)

    for bad in (
        [[[1, 2, 3]]],
        np.zeros((4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 2), dtype=np.uint8),
        np.zeros((2, 2, 2, 3), dtype=np.uint8),
    ):
        with pytest.raises(CaptureError):
            normalize_bgr(bad)  # type: ignore[arg-type]


def test_window_region_tracker_refresh_and_empty_class_name() -> None:
    seen_calls: list[tuple[str, str | None]] = []
    clock = _FakeClock(start=10.0, step=0.25)
    rects = [
        WindowRect(100, 100, 900, 700, hwnd=55),
        WindowRect(120, 110, 920, 710, hwnd=55),
    ]

    def _find(title: str, class_name: str | None) -> int | None:
        seen_calls.append((title, class_name))
        return 55

    def _get_rect(hwnd: int) -> WindowRect | None:
        assert hwnd == 55
        return rects[0]

    cfg = WindowConfig(title="VRChat", class_name="", rect_refresh_s=1.0)
    tracker = WindowRegionTracker(
        window_config=cfg,
        crop_to_window=True,
        clock_fn=clock,
        find_window_fn=_find,
        get_client_rect_fn=_get_rect,
    )
    mon = WindowRect(0, 0, 1920, 1080)

    # Refresh at t=10.0
    assert tracker.resolve_screen_rect(mon) == WindowRect(100, 100, 900, 700, hwnd=55)
    assert seen_calls == [("VRChat", None)]

    # Mutate rect list, still before rect_refresh_s=1.0 (t=10.25)
    rects[0] = rects[1]
    assert tracker.resolve_screen_rect(mon) == WindowRect(100, 100, 900, 700, hwnd=55)

    # Advance past 1.0s refresh interval (t=10.5, 10.75, 11.0)
    tracker.resolve_screen_rect(mon)
    tracker.resolve_screen_rect(mon)
    assert tracker.resolve_screen_rect(mon) == WindowRect(120, 110, 920, 710, hwnd=55)

    # Invalidate forces immediate re-find
    tracker.invalidate()
    tracker.resolve_screen_rect(mon)
    assert len(seen_calls) == 2

    # Disjoint static region raises CaptureError
    bad_tracker = WindowRegionTracker(static_region=(2000, 0, 2200, 200))
    with pytest.raises(CaptureError):
        bad_tracker.resolve_screen_rect(mon)


# ---------------------------------------------------------------------------
# Offline Sources: VideoFileFrameSource & ImageFrameSource
# ---------------------------------------------------------------------------


def test_video_file_source_real_reference_mp4(reference_video_path: Path) -> None:
    assert reference_video_path.is_file(), f"Missing reference MP4: {reference_video_path}"

    with VideoFileFrameSource(reference_video_path) as src:
        assert isinstance(src, FrameSource)
        assert src.fps == pytest.approx(60.0)
        assert src.width == 1920
        assert src.height == 1080
        assert src.frame_count == 1570

        count = 0
        first_pkt: FramePacket | None = None
        last_pkt: FramePacket | None = None
        while True:
            pkt = src.read()
            if pkt is None:
                break
            if first_pkt is None:
                first_pkt = pkt
            last_pkt = pkt
            assert pkt.frame_idx == count
            assert pkt.timestamp_s == pytest.approx(count / 60.0)
            count += 1

        assert count == 1570
        assert first_pkt is not None and last_pkt is not None
        assert first_pkt.image_bgr.shape == (1080, 1920, 3)
        assert first_pkt.image_bgr.dtype == np.uint8
        assert first_pkt.image_bgr.flags["C_CONTIGUOUS"]
        assert last_pkt.frame_idx == 1569
        assert last_pkt.image_bgr.shape == (1080, 1920, 3)
        assert src.read() is None

    # Idempotent close & read after close
    src.close()
    assert src.read() is None


def test_video_file_source_loop_preload_and_errors(tmp_path: Path) -> None:
    vid_path = tmp_path / "mini.avi"
    writer = cv2.VideoWriter(
        str(vid_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48)
    )
    assert writer.isOpened()
    for val in (10, 80, 200):
        frame = np.full((48, 64, 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with VideoFileFrameSource(vid_path, loop=True, preload=False) as src:
        assert src.fps == pytest.approx(30.0)
        packets = [src.read() for _ in range(5)]
        assert all(p is not None for p in packets)
        assert [p.frame_idx for p in packets if p is not None] == [0, 1, 2, 3, 4]
        assert packets[3].timestamp_s == pytest.approx(3 / 30.0)  # type: ignore[union-attr]

    with VideoFileFrameSource(vid_path, loop=True, preload=True, max_frames=2) as pre:
        assert pre.frame_count == 2
        pkts = [pre.read() for _ in range(3)]
        assert [p.frame_idx for p in pkts if p is not None] == [0, 1, 2]
        assert np.array_equal(pkts[0].image_bgr, pkts[2].image_bgr)  # type: ignore[union-attr]

    with pytest.raises(FileNotFoundError):
        VideoFileFrameSource(tmp_path / "missing.mp4")

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a valid video")
    with pytest.raises(CaptureError):
        VideoFileFrameSource(corrupt)


def test_image_frame_source_real_and_synthetic(
    tmp_path: Path, reference_screenshot_path: Path | None
) -> None:
    if reference_screenshot_path is not None:
        with ImageFrameSource(reference_screenshot_path, repeat=1, fps=60.0) as src:
            assert isinstance(src, FrameSource)
            assert (src.width, src.height) == (325, 223)
            pkt = src.read()
            assert pkt is not None
            assert pkt.frame_idx == 0
            assert pkt.timestamp_s == 0.0
            assert pkt.image_bgr.shape == (223, 325, 3)
            assert pkt.image_bgr.dtype == np.uint8
            assert pkt.image_bgr.flags["C_CONTIGUOUS"]
            assert src.read() is None
        src.close()
        assert src.read() is None

    # Synthetic 4-channel BGRA PNG + repeat=3 + copy isolation
    bgra_path = tmp_path / "bgra.png"
    bgra = np.zeros((20, 30, 4), dtype=np.uint8)
    bgra[:, :] = (10, 20, 30, 128)
    assert cv2.imwrite(str(bgra_path), bgra)

    with ImageFrameSource(bgra_path, repeat=3, fps=50.0) as img_src:
        p0 = img_src.read()
        assert p0 is not None
        p0.image_bgr[:] = 255  # Mutate caller copy
        p1 = img_src.read()
        p2 = img_src.read()
        p3 = img_src.read()
        assert p1 is not None and p2 is not None and p3 is None
        assert tuple(p1.image_bgr[0, 0]) == (10, 20, 30)
        assert p2.frame_idx == 2
        assert p2.timestamp_s == pytest.approx(2 / 50.0)

    # Grayscale PNG & repeat=0 & from_array repeat=None
    gray_path = tmp_path / "gray.png"
    assert cv2.imwrite(str(gray_path), np.full((12, 14), 90, dtype=np.uint8))
    g_src = ImageFrameSource(gray_path, repeat=0)
    assert g_src.read() is None

    raw_arr = np.full((10, 12, 3), 42, dtype=np.uint8)
    arr_src = ImageFrameSource.from_array(raw_arr, repeat=None)
    raw_arr[:] = 0  # External mutation must NOT corrupt stored array
    assert isinstance(arr_src, FrameSource)
    for expected_idx in range(5):
        p = arr_src.read()
        assert p is not None and p.frame_idx == expected_idx
        assert np.all(p.image_bgr == 42)
    arr_src.close()
    assert arr_src.read() is None

    with pytest.raises(FileNotFoundError):
        ImageFrameSource(tmp_path / "nope.png")

    bad_img = tmp_path / "bad.png"
    bad_img.write_bytes(b"broken")
    with pytest.raises(CaptureError):
        ImageFrameSource(bad_img)


# ---------------------------------------------------------------------------
# Live Backends (Mocked): DxcamCapture, MssCapture, GdiBitBltCapture, Factory
# ---------------------------------------------------------------------------


def test_dxcam_capture_mocked_lifecycle_and_monitor_relative() -> None:
    frame_a = np.full((100, 200, 3), 55, dtype=np.uint8)
    cam = _FakeCamera(
        left=1920,
        top=0,
        width=1920,
        height=1080,
        frames=[None, frame_a],
    )
    fake_dxcam = _FakeDxcamModule(
        outputs=[
            [SimpleNamespace(devicename=r"\\.\DISPLAY2")],
            [SimpleNamespace(devicename=r"\\.\DISPLAY1")],
        ],
        output_metadata={
            r"\\.\DISPLAY2": ["Secondary GPU", False, []],
            r"\\.\DISPLAY1": ["Primary GPU", True, []],
        },
    )
    # Primary output on (dev=1, out=0) succeeds; (dev=0, out=0) raises COMError if tried
    fake_dxcam.behaviors[(0, 0)] = _DummyCOMError("access denied")
    fake_dxcam.behaviors[(1, 0)] = cam

    clock = _FakeClock(start=50.0, step=0.01)
    src = DxcamCapture(
        monitor_idx=0,
        crop_to_window=True,
        window_config=WindowConfig(title="VRChat", class_name=""),
        dxcam_module=fake_dxcam,
        clock_fn=clock,
        find_window_fn=lambda title, cls: 99 if cls is None else None,
        get_client_rect_fn=lambda hwnd: WindowRect(2020, 100, 2220, 200, hwnd=hwnd),
    )
    assert isinstance(src, FrameSource)
    assert src.monitor_rect == WindowRect(1920, 0, 3840, 1080)
    assert len(fake_dxcam.create_calls) == 1
    assert fake_dxcam.create_calls[0]["device_idx"] == 1
    assert fake_dxcam.create_calls[0]["output_idx"] == 0

    # Idle frame (grab returns None) does NOT increment frame_idx
    assert src.read() is None
    assert src.frame_idx == 0

    pkt = src.read()
    assert pkt is not None
    assert pkt.frame_idx == 0
    assert pkt.image_bgr.shape == (100, 200, 3)
    assert pkt.image_bgr.flags["C_CONTIGUOUS"]
    # Screen-space (2020, 100, 2220, 200) -> monitor-relative (100, 100, 300, 200)
    assert cam.grab_calls[-1] == ((100, 100, 300, 200), True, True)

    src.close()
    src.close()
    assert cam.release_calls == 1
    assert src.read() is None


def test_dxcam_monitor_idx_ordering_and_real_factory_subprocess() -> None:
    # When non-primary output appears at index 0 and primary at index 1,
    # monitor_idx=0 must pick primary (1, 0) and monitor_idx=1 must pick secondary (0, 0).
    sec_cam = _FakeCamera(left=-1920, top=0, width=1920, height=1080)
    pri_cam = _FakeCamera(left=0, top=0, width=1920, height=1080)
    fake_dxcam = _FakeDxcamModule(
        outputs=[
            [SimpleNamespace(devicename=r"\\.\DISPLAY2")],
            [SimpleNamespace(devicename=r"\\.\DISPLAY1")],
        ],
        output_metadata={
            r"\\.\DISPLAY2": ["Adapter0", False, []],
            r"\\.\DISPLAY1": ["Adapter1", True, []],
        },
    )
    fake_dxcam.behaviors[(0, 0)] = sec_cam
    fake_dxcam.behaviors[(1, 0)] = pri_cam

    with DxcamCapture(monitor_idx=0, crop_to_window=False, dxcam_module=fake_dxcam) as s0:
        assert s0.monitor_rect == WindowRect(0, 0, 1920, 1080)
    with DxcamCapture(monitor_idx=1, crop_to_window=False, dxcam_module=fake_dxcam) as s1:
        assert s1.monitor_rect == WindowRect(-1920, 0, 0, 1080)

    assert [(c["device_idx"], c["output_idx"]) for c in fake_dxcam.create_calls] == [
        (1, 0),
        (0, 0),
    ]

    # DXFactory() fallback when __factory/_factory are not present on module double
    factory_obj = SimpleNamespace(
        outputs=[
            [SimpleNamespace(devicename=r"\\.\DISPLAY2"), SimpleNamespace(devicename=r"\\.\DISPLAY1")]
        ],
        output_metadata={
            r"\\.\DISPLAY2": ["GPU", False, []],
            r"\\.\DISPLAY1": ["GPU", True, []],
        },
    )
    from vrc_smith_auto.capture.dxcam_source import _select_dxcam_candidates

    dx_cls_double = SimpleNamespace(DXFactory=lambda: factory_obj)
    assert _select_dxcam_candidates(dx_cls_double, 0, None) == [(0, 1), (0, 0)]
    assert _select_dxcam_candidates(dx_cls_double, 1, None) == [(0, 0)]

    # Isolated subprocess test against real .venv dxcam module (keeps main process DPI-pure)
    if sys.platform == "win32":
        script = (
            "import dxcam; "
            "from vrc_smith_auto.capture.dxcam_source import _select_dxcam_candidates; "
            "factory = getattr(dxcam, '__factory', None); "
            "assert factory is not None, 'dxcam.__factory missing'; "
            "assert isinstance(factory.outputs, list) and any(factory.outputs), factory.outputs; "
            "cands = _select_dxcam_candidates(dxcam, 0, None); "
            "total = sum(len(x) for x in factory.outputs); "
            "assert len(cands) == total, (cands, total); "
            "assert cands[0][1] is not None, cands; "
            "d0, o0 = cands[0]; "
            "out0 = factory.outputs[d0][o0]; "
            "meta0 = factory.output_metadata.get(out0.devicename); "
            "assert meta0 is None or bool(meta0[1]) or not any(m[1] for m in factory.output_metadata.values())"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_dxcam_capture_errors() -> None:
    fake_dxcam = _FakeDxcamModule(
        outputs=[[SimpleNamespace(devicename=r"\\.\DISPLAY1")]],
        output_metadata={r"\\.\DISPLAY1": ["GPU", True, []]},
    )
    fake_dxcam.behaviors[(0, 0)] = _DummyCOMError("headless")
    with pytest.raises(CaptureError):
        DxcamCapture(monitor_idx=0, dxcam_module=fake_dxcam)

    with pytest.raises(CaptureError):
        DxcamCapture(monitor_idx=3, dxcam_module=fake_dxcam)

    err_cam = _FakeCamera(grab_exc=_DummyCOMError("desktop lost"))
    fake_dxcam.behaviors[(0, 0)] = err_cam
    with DxcamCapture(monitor_idx=0, crop_to_window=False, dxcam_module=fake_dxcam) as src:
        with pytest.raises(CaptureError):
            src.read()
    assert err_cam.release_calls == 1


def test_mss_capture_monitor_mapping_and_errors() -> None:
    virtual = {"left": -1920, "top": 0, "width": 3840, "height": 1080}
    mon_primary = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    mon_second = {"left": -1920, "top": 0, "width": 1920, "height": 1080}

    bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
    bgra[:, :] = (12, 34, 56, 255)
    sct = _FakeMss(
        monitors=[virtual, mon_primary, mon_second],
        primary_monitor=mon_primary,
        shots=[bgra],
    )
    src = MssCapture(
        monitor_idx=0,
        crop_to_window=False,
        mss_factory=lambda: sct,
        clock_fn=_FakeClock(),
    )
    assert isinstance(src, FrameSource)
    assert src.monitor_rect == WindowRect(0, 0, 1920, 1080)

    pkt = src.read()
    assert pkt is not None
    assert pkt.frame_idx == 0
    assert pkt.image_bgr.shape == (1080, 1920, 3)
    assert pkt.image_bgr.flags["C_CONTIGUOUS"]
    assert tuple(pkt.image_bgr[0, 0]) == (12, 34, 56)
    # Must request primary monitor (not monitors[0] virtual box!)
    assert sct.grab_rects[0] == {"left": 0, "top": 0, "width": 1920, "height": 1080}

    src.close()
    src.close()
    assert sct.close_calls == 1
    assert src.read() is None

    # monitor_idx=1 selects monitors[2]
    sct2 = _FakeMss(monitors=[virtual, mon_primary, mon_second])
    src2 = MssCapture(monitor_idx=1, crop_to_window=False, mss_factory=lambda: sct2)
    assert src2.monitor_rect == WindowRect(-1920, 0, 0, 1080)
    src2.close()

    # Out of range monitor_idx
    sct3 = _FakeMss(monitors=[virtual, mon_primary])
    with pytest.raises(CaptureError):
        MssCapture(monitor_idx=1, mss_factory=lambda: sct3)
    assert sct3.close_calls == 1

    # ScreenShotError wrapped into CaptureError
    sct4 = _FakeMss(
        monitors=[virtual, mon_primary],
        grab_exc=_DummyScreenShotError("BitBlt failed"),
    )
    with MssCapture(monitor_idx=0, crop_to_window=False, mss_factory=lambda: sct4) as src4:
        with pytest.raises(CaptureError):
            src4.read()
    assert sct4.close_calls == 1


def test_mss_capture_non_primary_first_ordering() -> None:
    from vrc_smith_auto.capture.mss_source import _select_mss_monitor

    virtual = {"left": -1920, "top": 0, "width": 3840, "height": 1080}

    # Case 1: explicit is_primary flags, secondary precedes primary in EnumDisplayMonitors order
    mon_sec_flag = {"left": -1920, "top": 0, "width": 1920, "height": 1080, "is_primary": False}
    mon_pri_flag = {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True}
    with MssCapture(
        monitor_idx=0,
        crop_to_window=False,
        mss_factory=lambda: _FakeMss(
            monitors=[virtual, mon_sec_flag, mon_pri_flag],
            primary_monitor=mon_pri_flag,
        ),
    ) as m0:
        assert m0.monitor_rect == WindowRect(0, 0, 1920, 1080)
    with MssCapture(
        monitor_idx=1,
        crop_to_window=False,
        mss_factory=lambda: _FakeMss(
            monitors=[virtual, mon_sec_flag, mon_pri_flag],
            primary_monitor=mon_pri_flag,
        ),
    ) as m1:
        assert m1.monitor_rect == WindowRect(-1920, 0, 0, 1080)

    sct_oob_flag = _FakeMss(
        monitors=[virtual, mon_sec_flag, mon_pri_flag],
        primary_monitor=mon_pri_flag,
    )
    with pytest.raises(CaptureError):
        MssCapture(monitor_idx=2, crop_to_window=False, mss_factory=lambda: sct_oob_flag)
    assert sct_oob_flag.close_calls == 1

    # Case 2: is_primary absent across phys; fallback to primary_monitor (both same object and equal copy)
    mon_sec_plain = {"left": -1920, "top": 0, "width": 1920, "height": 1080}
    mon_pri_plain = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    for primary_ref in (mon_pri_plain, dict(mon_pri_plain)):
        with MssCapture(
            monitor_idx=0,
            crop_to_window=False,
            mss_factory=lambda ref=primary_ref: _FakeMss(
                monitors=[virtual, mon_sec_plain, mon_pri_plain],
                primary_monitor=ref,
            ),
        ) as m0_fb:
            assert m0_fb.monitor_rect == WindowRect(0, 0, 1920, 1080)
        with MssCapture(
            monitor_idx=1,
            crop_to_window=False,
            mss_factory=lambda ref=primary_ref: _FakeMss(
                monitors=[virtual, mon_sec_plain, mon_pri_plain],
                primary_monitor=ref,
            ),
        ) as m1_fb:
            assert m1_fb.monitor_rect == WindowRect(-1920, 0, 0, 1080)

        sct_oob_fb = _FakeMss(
            monitors=[virtual, mon_sec_plain, mon_pri_plain],
            primary_monitor=primary_ref,
        )
        with pytest.raises(CaptureError):
            MssCapture(monitor_idx=2, crop_to_window=False, mss_factory=lambda: sct_oob_fb)
        assert sct_oob_fb.close_calls == 1

    # Negative monitor_idx directly into _select_mss_monitor raises CaptureError
    with pytest.raises(CaptureError):
        _select_mss_monitor(
            _FakeMss(monitors=[virtual, mon_sec_flag, mon_pri_flag], primary_monitor=mon_pri_flag),
            -1,
        )


def test_gdi_capture_bitblt_cleanup_and_errors() -> None:
    api = _FakeGdiApi(
        monitors=[(WindowRect(10, 20, 110, 100), True)],
        pixel_bgra=(21, 42, 63, 255),
    )
    with GdiBitBltCapture(
        monitor_idx=0,
        crop_to_window=False,
        api=api,
        get_last_error_fn=lambda: api.last_error,
        clock_fn=_FakeClock(),
    ) as src:
        assert isinstance(src, FrameSource)
        pkt = src.read()
        assert pkt is not None
        assert pkt.image_bgr.shape == (80, 100, 3)
        assert pkt.image_bgr.flags["C_CONTIGUOUS"]
        assert tuple(pkt.image_bgr[0, 0]) == (21, 42, 63)
        assert api.captured_bmi == (100, -80, 32)

    # Verify SelectObject restore happened BEFORE GetDIBits and handles were cleaned up
    select_indices = [i for i, ev in enumerate(api.events) if ev[0] == "SelectObject"]
    getdibits_idx = next(i for i, ev in enumerate(api.events) if ev[0] == "GetDIBits")
    assert len(select_indices) == 2
    assert select_indices[1] < getdibits_idx
    assert api.events[select_indices[1]] == ("SelectObject", 2002, 4004)
    assert ("DeleteObject", 3003) in api.events
    assert ("DeleteDC", 2002) in api.events
    assert ("ReleaseDC", None, 1001) in api.events

    # BitBlt failure must still restore SelectObject(mem_dc, old_obj) in finally
    fail_api = _FakeGdiApi(bitblt_ok=False, last_error=6)
    with GdiBitBltCapture(
        monitor_idx=0,
        crop_to_window=False,
        api=fail_api,
        get_last_error_fn=lambda: fail_api.last_error,
    ) as fail_src:
        with pytest.raises(CaptureError, match="winerror=6"):
            fail_src.read()
    assert ("SelectObject", 2002, 4004) in fail_api.events
    assert ("DeleteObject", 3003) in fail_api.events
    assert ("DeleteDC", 2002) in fail_api.events
    assert ("ReleaseDC", None, 1001) in fail_api.events

    # CreateCompatibleDC failure must still ReleaseDC
    dc_fail_api = _FakeGdiApi(mem_dc=0, last_error=8)
    with pytest.raises(CaptureError):
        GdiBitBltCapture(
            monitor_idx=0,
            api=dc_fail_api,
            get_last_error_fn=lambda: dc_fail_api.last_error,
        )
    assert ("ReleaseDC", None, 1001) in dc_fail_api.events

    # Non-primary-first monitor ordering: monitor_idx=0 -> primary, monitor_idx=1 -> secondary
    sec_rect = WindowRect(-1920, 0, 0, 1080)
    pri_rect = WindowRect(0, 0, 1920, 1080)
    multi_api_0 = _FakeGdiApi(monitors=[(sec_rect, False), (pri_rect, True)])
    multi_api_1 = _FakeGdiApi(monitors=[(sec_rect, False), (pri_rect, True)])
    with GdiBitBltCapture(monitor_idx=0, crop_to_window=False, api=multi_api_0) as g0:
        assert g0.monitor_rect == pri_rect
    with GdiBitBltCapture(monitor_idx=1, crop_to_window=False, api=multi_api_1) as g1:
        assert g1.monitor_rect == sec_rect

    # Partial GetDIBits scanlines (0 < scanlines < height) must raise CaptureError
    partial_api = _FakeGdiApi(
        monitors=[(WindowRect(0, 0, 64, 40), True)],
        scanlines=20,
        last_error=0,
    )
    with GdiBitBltCapture(
        monitor_idx=0,
        crop_to_window=False,
        api=partial_api,
        get_last_error_fn=lambda: partial_api.last_error,
    ) as partial_src:
        with pytest.raises(CaptureError, match="scanlines=20, expected=40"):
            partial_src.read()

    if sys.platform == "win32":
        from vrc_smith_auto.capture.gdi_source import _GdiApi

        real_api = _GdiApi()
        assert real_api.gdi32.CreateCompatibleBitmap.restype == wintypes.HBITMAP
        assert real_api.gdi32.CreateCompatibleBitmap.argtypes == [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
        ]
        assert real_api.gdi32.GetDIBits.argtypes is not None
        assert real_api.gdi32.BitBlt.restype == wintypes.BOOL

        # Real mss.MSS() init must emit zero DeprecationWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            real_mss = MssCapture(monitor_idx=0, crop_to_window=False)
            assert real_mss.monitor_rect.is_valid
            real_mss.close()
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warnings == [], f"Unexpected DeprecationWarning: {dep_warnings}"


def test_create_live_capture_auto_fallback_and_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    dpi_calls: list[bool] = []
    monkeypatch.setattr(capture_base, "enable_dpi_awareness", lambda: dpi_calls.append(True) or True)

    class _StubSource:
        def __init__(self, name: str, *, read_val: Any = None, read_exc: BaseException | None = None) -> None:
            self.name = name
            self._read_val = read_val
            self._read_exc = read_exc
            self.closed = False

        def read(self) -> FramePacket | None:
            if self._read_exc is not None:
                raise self._read_exc
            return self._read_val

        def close(self) -> None:
            self.closed = True

    def _fake_dxcam_fail(**kwargs: Any) -> Any:
        calls.append("dxcam")
        raise CaptureError("dxgi unavailable")

    def _fake_mss_probe_fail(**kwargs: Any) -> Any:
        calls.append("mss")
        return _StubSource("mss", read_exc=CaptureError("bitblt failed"))

    gdi_stub = _StubSource(
        "gdi",
        read_val=FramePacket(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0),
    )

    def _fake_gdi_ok(**kwargs: Any) -> Any:
        calls.append("gdi")
        return gdi_stub

    import vrc_smith_auto.capture.dxcam_source as dxcam_mod
    import vrc_smith_auto.capture.mss_source as mss_mod
    import vrc_smith_auto.capture.gdi_source as gdi_mod

    monkeypatch.setattr(dxcam_mod, "DxcamCapture", _fake_dxcam_fail)
    monkeypatch.setattr(mss_mod, "MssCapture", _fake_mss_probe_fail)
    monkeypatch.setattr(gdi_mod, "GdiBitBltCapture", _fake_gdi_ok)

    src = create_live_capture(AppConfig(), probe=True)
    assert src is gdi_stub
    assert calls == ["dxcam", "mss", "gdi"]
    if sys.platform == "win32":
        assert len(dpi_calls) >= 1

    # Idle dxcam returning None on probe=True must be accepted (no fallback!)
    calls.clear()
    idle_dxcam = _StubSource("dxcam", read_val=None)
    monkeypatch.setattr(dxcam_mod, "DxcamCapture", lambda **kw: calls.append("dxcam") or idle_dxcam)
    assert create_live_capture(CaptureConfig(backend="auto"), probe=True) is idle_dxcam
    assert calls == ["dxcam"]

    # Explicit backend must NOT fall back
    calls.clear()
    monkeypatch.setattr(dxcam_mod, "DxcamCapture", _fake_dxcam_fail)
    with pytest.raises(CaptureError, match="dxgi unavailable"):
        create_live_capture(CaptureConfig(backend="dxcam"))
    assert calls == ["dxcam"]

    # All failing in auto raises combined CaptureError
    monkeypatch.setattr(gdi_mod, "GdiBitBltCapture", lambda **kw: (_ for _ in ()).throw(CaptureError("gdi dead")))
    with pytest.raises(CaptureError, match="All live capture backends failed"):
        create_live_capture(CaptureConfig(backend="auto"), probe=True)
