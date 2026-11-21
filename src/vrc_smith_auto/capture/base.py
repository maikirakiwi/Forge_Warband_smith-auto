from __future__ import annotations

import inspect
import logging
import sys
import time
from typing import Any, Callable

import cv2
import numpy as np

from vrc_smith_auto.config import AppConfig, CaptureConfig, WindowConfig
from vrc_smith_auto.protocols import FrameSource
from vrc_smith_auto.window import (
    WindowRect,
    enable_dpi_awareness,
    find_vrchat_window,
    get_client_rect,
)

logger = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """Raised when screen capture initialization, grabbing, or media decoding fails."""


def normalize_bgr(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale, BGR, or BGRA uint8 image arrays to contiguous (H, W, 3) BGR."""
    if not isinstance(image, np.ndarray):
        raise CaptureError(f"Expected numpy.ndarray frame, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise CaptureError(f"Expected uint8 frame dtype, got {image.dtype}")
    if image.size == 0:
        raise CaptureError("Frame array is empty")

    if image.ndim == 2:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise CaptureError(f"Invalid 2D frame shape: {image.shape}")
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    if image.ndim == 3:
        h, w, ch = image.shape
        if h <= 0 or w <= 0:
            raise CaptureError(f"Invalid 3D frame dimensions: {image.shape}")
        if ch == 1:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        if ch == 3:
            return np.ascontiguousarray(image)
        if ch == 4:
            bgr = image[:, :, :3]
            if bgr.flags["C_CONTIGUOUS"]:
                return bgr.copy()
            return np.ascontiguousarray(bgr)
        raise CaptureError(f"Unsupported channel count {ch}; expected 1, 3, or 4")

    raise CaptureError(f"Expected 2D or 3D image array, got {image.ndim}D shape {image.shape}")


def coerce_region(region: WindowRect | tuple[int, int, int, int] | None) -> WindowRect | None:
    """Convert an optional (left, top, right, bottom) tuple or WindowRect into a valid WindowRect."""
    if region is None:
        return None
    if isinstance(region, WindowRect):
        rect = region
    elif isinstance(region, tuple) and len(region) == 4:
        left, top, right, bottom = (int(v) for v in region)
        rect = WindowRect(left=left, top=top, right=right, bottom=bottom)
    else:
        raise CaptureError(f"Invalid capture region specification: {region!r}")

    if not rect.is_valid:
        raise CaptureError(f"Capture region must have positive width and height, got {rect}")
    return rect


class WindowRegionTracker:
    """Tracks target VRChat client-area bounds with periodic refresh."""

    def __init__(
        self,
        window_config: WindowConfig | None = None,
        crop_to_window: bool = True,
        static_region: WindowRect | tuple[int, int, int, int] | None = None,
        *,
        clock_fn: Callable[[], float] = time.perf_counter,
        find_window_fn: Callable[..., int | None] = find_vrchat_window,
        get_client_rect_fn: Callable[[int], WindowRect | None] = get_client_rect,
    ) -> None:
        self._window_config = window_config or WindowConfig()
        self._crop_to_window = bool(crop_to_window)
        self._static_region = coerce_region(static_region)
        self._clock_fn = clock_fn
        self._find_window_fn = find_window_fn
        self._get_client_rect_fn = get_client_rect_fn
        self._find_accepts_process_name = False
        try:
            sig = inspect.signature(find_window_fn)
            self._find_accepts_process_name = (
                "process_name" in sig.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            )
        except (ValueError, TypeError):
            self._find_accepts_process_name = False

        self._hwnd: int | None = None
        self._cached_rect: WindowRect | None = None
        self._last_refresh_s: float | None = None

    def _find_target_window(self) -> int | None:
        class_arg = self._window_config.class_name or None
        if self._find_accepts_process_name:
            proc_arg = "VRChat.exe" if self._window_config.title == "VRChat" else None
            return self._find_window_fn(
                self._window_config.title,
                class_arg,
                process_name=proc_arg,
            )
        return self._find_window_fn(self._window_config.title, class_arg)

    @property
    def hwnd(self) -> int | None:
        return self._hwnd

    @property
    def cached_rect(self) -> WindowRect | None:
        return self._cached_rect

    def invalidate(self) -> None:
        """Force re-discovery of target HWND and client rect on the next read."""
        self._hwnd = None
        self._cached_rect = None
        self._last_refresh_s = None

    def _refresh_if_due(self) -> None:
        now = float(self._clock_fn())
        if (
            self._last_refresh_s is not None
            and (now - self._last_refresh_s) < self._window_config.rect_refresh_s
        ):
            return

        self._last_refresh_s = now
        if self._hwnd is None:
            self._hwnd = self._find_target_window()

        rect: WindowRect | None = None
        if self._hwnd is not None:
            rect = self._get_client_rect_fn(self._hwnd)
            if rect is None:
                # Target window may have closed, minimized, or been recreated.
                self._hwnd = self._find_target_window()
                if self._hwnd is not None:
                    rect = self._get_client_rect_fn(self._hwnd)
                if rect is None:
                    self._hwnd = None

        self._cached_rect = rect if (rect is not None and rect.is_valid) else None

    def resolve_screen_rect(self, fallback_rect: WindowRect) -> WindowRect:
        """Resolve screen-space capture rectangle clamped to fallback_rect."""
        if self._static_region is not None:
            clamped = self._static_region.clamp_to_bounds(fallback_rect)
            if clamped is None:
                raise CaptureError(
                    f"Requested capture region {self._static_region.as_ltrb()} does not intersect "
                    f"monitor bounds {fallback_rect.as_ltrb()}"
                )
            return clamped

        if not self._crop_to_window:
            return fallback_rect

        self._refresh_if_due()
        if self._cached_rect is not None:
            clamped = self._cached_rect.clamp_to_bounds(fallback_rect)
            if clamped is not None:
                return clamped

        return fallback_rect


def _build_backend(
    name: str,
    cap_cfg: CaptureConfig,
    win_cfg: WindowConfig,
    region: WindowRect | tuple[int, int, int, int] | None,
) -> FrameSource:
    # Local imports avoid circular references; backend modules themselves are import-time pure.
    from vrc_smith_auto.capture.dxcam_source import DxcamCapture
    from vrc_smith_auto.capture.gdi_source import GdiBitBltCapture
    from vrc_smith_auto.capture.mss_source import MssCapture

    if name == "dxcam":
        return DxcamCapture(
            monitor_idx=cap_cfg.monitor_idx,
            crop_to_window=cap_cfg.crop_to_window,
            window_config=win_cfg,
            region=region,
        )
    if name == "mss":
        return MssCapture(
            monitor_idx=cap_cfg.monitor_idx,
            crop_to_window=cap_cfg.crop_to_window,
            window_config=win_cfg,
            region=region,
        )
    if name == "gdi":
        return GdiBitBltCapture(
            monitor_idx=cap_cfg.monitor_idx,
            crop_to_window=cap_cfg.crop_to_window,
            window_config=win_cfg,
            region=region,
        )
    raise CaptureError(f"Unsupported live capture backend: {name!r}")


def create_live_capture(
    config: AppConfig | CaptureConfig,
    *,
    window_config: WindowConfig | None = None,
    region: WindowRect | tuple[int, int, int, int] | None = None,
    probe: bool = False,
) -> FrameSource:
    """Create a live screen capture FrameSource from AppConfig or CaptureConfig."""
    if isinstance(config, AppConfig):
        cap_cfg = config.capture
        win_cfg = window_config or config.window
    elif isinstance(config, CaptureConfig):
        cap_cfg = config
        win_cfg = window_config or WindowConfig()
    else:
        raise TypeError(f"Expected AppConfig or CaptureConfig, got {type(config).__name__}")

    if sys.platform == "win32":
        enable_dpi_awareness()

    if cap_cfg.backend != "auto":
        source = _build_backend(cap_cfg.backend, cap_cfg, win_cfg, region)
        if probe:
            try:
                source.read()
            except Exception as exc:
                try:
                    source.close()
                except Exception:
                    pass
                if isinstance(exc, CaptureError):
                    raise
                raise CaptureError(f"Capture backend {cap_cfg.backend!r} probe failed: {exc}") from exc
        return source

    errors: list[str] = []
    for candidate in ("dxcam", "mss", "gdi"):
        source: FrameSource | None = None
        try:
            source = _build_backend(candidate, cap_cfg, win_cfg, region)
            if probe:
                # Note: dxcam.read() returning None indicates no new DWM frame (healthy idle state).
                source.read()
            return source
        except (CaptureError, ImportError, OSError, RuntimeError, ValueError) as exc:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            msg = f"{candidate}: {exc}"
            errors.append(msg)
            logger.debug("Live capture candidate %s failed: %s", candidate, exc)

    raise CaptureError("All live capture backends failed: " + "; ".join(errors))
