from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

VK_ESCAPE = 0x1B
VK_F8 = 0x77
KEY_DOWN_MASK = 0x8000


@dataclass(frozen=True, slots=True)
class HotkeyEvents:
    """Edge-triggered hotkey events returned by a single poll()."""

    toggle_armed: bool = False
    emergency_stop: bool = False


_GET_ASYNC_KEY_STATE: Callable[[int], int] | None = None


def _get_default_key_state_fn() -> Callable[[int], int]:
    global _GET_ASYNC_KEY_STATE
    if sys.platform != "win32":
        return lambda _vk: 0

    if _GET_ASYNC_KEY_STATE is None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = wintypes.SHORT
        fn = user32.GetAsyncKeyState
        _GET_ASYNC_KEY_STATE = lambda vk: int(fn(int(vk)))  # noqa: E731

    return _GET_ASYNC_KEY_STATE


class HotkeyController:
    """Edge-triggered polling controller for F8 arm/pause toggle and Esc emergency stop."""

    def __init__(
        self,
        *,
        toggle_vk: int = VK_F8,
        stop_vk: int = VK_ESCAPE,
        key_state_fn: Callable[[int], int] | None = None,
    ) -> None:
        self._toggle_vk = int(toggle_vk)
        self._stop_vk = int(stop_vk)
        self._key_state_fn = key_state_fn
        self._prev_toggle_down = False
        self._prev_stop_down = False

    def _read_key_state(self, vk: int) -> int:
        if self._key_state_fn is None:
            self._key_state_fn = _get_default_key_state_fn()
        return int(self._key_state_fn(vk))

    def poll(self) -> HotkeyEvents:
        toggle_down = (self._read_key_state(self._toggle_vk) & KEY_DOWN_MASK) != 0
        stop_down = (self._read_key_state(self._stop_vk) & KEY_DOWN_MASK) != 0

        toggle_edge = toggle_down and not self._prev_toggle_down
        stop_edge = stop_down and not self._prev_stop_down

        self._prev_toggle_down = toggle_down
        self._prev_stop_down = stop_down

        return HotkeyEvents(toggle_armed=toggle_edge, emergency_stop=stop_edge)

    def reset(self) -> None:
        self._prev_toggle_down = False
        self._prev_stop_down = False
