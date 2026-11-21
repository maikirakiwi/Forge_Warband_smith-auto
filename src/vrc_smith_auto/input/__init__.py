from __future__ import annotations

from .spy_mouse import RecordedClick, SpyMouseBackend
from .win32_sendinput import (
    HARDWAREINPUT,
    INPUT,
    INPUT_HARDWARE,
    INPUT_KEYBOARD,
    INPUT_MOUSE,
    KEYBDINPUT,
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MOVE,
    MOUSEINPUT,
    Win32SendInputMouse,
    build_mouse_left_input,
)

__all__ = [
    "HARDWAREINPUT",
    "INPUT",
    "INPUT_HARDWARE",
    "INPUT_KEYBOARD",
    "INPUT_MOUSE",
    "KEYBDINPUT",
    "MOUSEEVENTF_ABSOLUTE",
    "MOUSEEVENTF_LEFTDOWN",
    "MOUSEEVENTF_LEFTUP",
    "MOUSEEVENTF_MOVE",
    "MOUSEINPUT",
    "RecordedClick",
    "SpyMouseBackend",
    "Win32SendInputMouse",
    "build_mouse_left_input",
]
