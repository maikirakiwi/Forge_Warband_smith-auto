from __future__ import annotations

from .config import (
    AppConfig,
    CaptureConfig,
    GeometryConfig,
    HSVConfig,
    HSVRange,
    TriggerConfig,
    WindowConfig,
)
from .protocols import FramePacket, FrameSource, MouseBackend
from .vision.models import DetectionResult, OrientedBox
from .window import WindowRect

__version__ = "0.1.0"

__all__ = [
    "__version__",
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
]
