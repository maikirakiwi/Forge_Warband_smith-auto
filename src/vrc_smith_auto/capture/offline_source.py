from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vrc_smith_auto.capture.base import CaptureError, normalize_bgr
from vrc_smith_auto.protocols import FramePacket


def _validate_positive_fps(fps: float) -> float:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError(f"fps must be a finite number > 0, got {fps!r}")
    val = float(fps)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"fps must be > 0 and finite, got {fps!r}")
    return val


def _validate_optional_nonneg_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be None or an integer >= 0, got {value!r}")
    return value


class VideoFileFrameSource:
    """Deterministic video-file FrameSource backed by cv2.VideoCapture."""

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        preload: bool = False,
        max_frames: int | None = None,
        default_fps: float = 60.0,
    ) -> None:
        self._path = Path(path)
        self._loop = bool(loop)
        self._preload = bool(preload)
        self._max_frames = _validate_optional_nonneg_int(max_frames, "max_frames")
        fallback_fps = _validate_positive_fps(default_fps)

        if not self._path.is_file():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"Failed to open video file: {self._path}")

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = raw_fps if (math.isfinite(raw_fps) and raw_fps > 0.0) else fallback_fps
        raw_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._frame_count = max(0, raw_count)
        self._width = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        self._height = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        self._closed = False
        self._frame_idx = 0
        self._stream_pos = 0
        self._cursor = 0
        self._preloaded: list[np.ndarray] | None = None
        self._cap: cv2.VideoCapture | None = cap

        if self._preload:
            frames: list[np.ndarray] = []
            try:
                while self._max_frames is None or len(frames) < self._max_frames:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    frames.append(normalize_bgr(frame))
            finally:
                cap.release()
                self._cap = None

            self._preloaded = frames
            self._frame_count = len(frames)
            if frames:
                self._height, self._width = int(frames[0].shape[0]), int(frames[0].shape[1])

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def read(self) -> FramePacket | None:
        if self._closed:
            return None

        if self._preloaded is not None:
            if not self._preloaded:
                return None
            if self._cursor >= len(self._preloaded):
                if self._loop:
                    self._cursor = 0
                else:
                    return None
            image_bgr = self._preloaded[self._cursor].copy()
            self._cursor += 1
        else:
            if self._cap is None:
                return None
            if self._max_frames is not None and self._stream_pos >= self._max_frames:
                if self._loop and self._stream_pos > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._stream_pos = 0
                else:
                    return None

            ok, raw = self._cap.read()
            if not ok or raw is None:
                if self._loop and self._stream_pos > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._stream_pos = 0
                    ok, raw = self._cap.read()
                    if not ok or raw is None:
                        return None
                else:
                    return None
            self._stream_pos += 1
            image_bgr = normalize_bgr(raw)

        idx = self._frame_idx
        self._frame_idx += 1
        return FramePacket(
            image_bgr=image_bgr,
            timestamp_s=idx / self._fps,
            frame_idx=idx,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cap = self._cap
        self._cap = None
        if cap is not None:
            cap.release()
        if self._preloaded is not None:
            self._preloaded.clear()
            self._preloaded = None

    def __enter__(self) -> VideoFileFrameSource:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class ImageFrameSource:
    """Static image FrameSource supporting single-shot, finite repeat, or infinite playback."""

    def __init__(
        self,
        path: str | Path,
        *,
        repeat: int | None = 1,
        fps: float = 60.0,
    ) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")

        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise CaptureError(f"Failed to decode image file: {p}")

        self._init_from_normalized(normalize_bgr(raw), repeat=repeat, fps=fps, path=p)

    def _init_from_normalized(
        self,
        image_bgr: np.ndarray,
        *,
        repeat: int | None,
        fps: float,
        path: Path | None,
    ) -> None:
        self._repeat = _validate_optional_nonneg_int(repeat, "repeat")
        self._fps = _validate_positive_fps(fps)
        self._path = path
        self._image_bgr = image_bgr
        self._closed = False
        self._frame_idx = 0

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        *,
        repeat: int | None = 1,
        fps: float = 60.0,
    ) -> ImageFrameSource:
        obj = cls.__new__(cls)
        obj._init_from_normalized(normalize_bgr(image).copy(), repeat=repeat, fps=fps, path=None)
        return obj

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def repeat(self) -> int | None:
        return self._repeat

    @property
    def width(self) -> int:
        return int(self._image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self._image_bgr.shape[0])

    def read(self) -> FramePacket | None:
        if self._closed:
            return None
        if self._repeat is not None and self._frame_idx >= self._repeat:
            return None

        idx = self._frame_idx
        self._frame_idx += 1
        return FramePacket(
            image_bgr=self._image_bgr.copy(),
            timestamp_s=idx / self._fps,
            frame_idx=idx,
        )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> ImageFrameSource:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
