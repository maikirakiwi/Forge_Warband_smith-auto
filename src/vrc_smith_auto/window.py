from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
PROCESS_PER_MONITOR_DPI_AWARE = 2
MONITOR_DEFAULTTONEAREST = 2

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_INFO_CLASS = 20

ERROR_ACCESS_DENIED = 5
E_ACCESSDENIED_U32 = 0x80070005


@dataclass(frozen=True, slots=True)
class WindowRect:
    """Screen-space physical pixel rectangle [left, top, right, bottom)."""

    left: int
    top: int
    right: int
    bottom: int
    hwnd: int = 0

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def is_valid(self) -> bool:
        return self.right > self.left and self.bottom > self.top

    @classmethod
    def from_xywh(
        cls,
        left: int,
        top: int,
        width: int,
        height: int,
        hwnd: int = 0,
    ) -> WindowRect:
        return cls(
            left=int(left),
            top=int(top),
            right=int(left) + int(width),
            bottom=int(top) + int(height),
            hwnd=int(hwnd),
        )

    def as_ltrb(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def as_mss_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    def clamp_to_bounds(self, bounds: WindowRect) -> WindowRect | None:
        left = max(self.left, bounds.left)
        top = max(self.top, bounds.top)
        right = min(self.right, bounds.right)
        bottom = min(self.bottom, bounds.bottom)
        if right <= left or bottom <= top:
            return None
        return WindowRect(left=left, top=top, right=right, bottom=bottom, hwnd=self.hwnd)

    def to_monitor_relative(self, monitor_bounds: WindowRect) -> tuple[int, int, int, int] | None:
        clamped = self.clamp_to_bounds(monitor_bounds)
        if clamped is None:
            return None
        return (
            clamped.left - monitor_bounds.left,
            clamped.top - monitor_bounds.top,
            clamped.right - monitor_bounds.left,
            clamped.bottom - monitor_bounds.top,
        )


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


class _TOKEN_ELEVATION(ctypes.Structure):
    _fields_ = [
        ("TokenIsElevated", wintypes.DWORD),
    ]


class _Win32Api:
    """Lazy-initialized Win32 ctypes DLL bindings using use_last_error=True."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        try:
            self.shcore: Any = ctypes.WinDLL("shcore", use_last_error=True)
        except OSError:
            self.shcore = None
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._configure()

    def _configure(self) -> None:
        u = self.user32
        if hasattr(u, "SetProcessDpiAwarenessContext"):
            u.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            u.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if hasattr(u, "SetProcessDPIAware"):
            u.SetProcessDPIAware.argtypes = []
            u.SetProcessDPIAware.restype = wintypes.BOOL

        u.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        u.IsWindow.argtypes = [wintypes.HWND]
        u.IsWindow.restype = wintypes.BOOL
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.IsWindowVisible.restype = wintypes.BOOL
        u.IsIconic.argtypes = [wintypes.HWND]
        u.IsIconic.restype = wintypes.BOOL
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.restype = ctypes.c_int
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetClassNameW.restype = ctypes.c_int
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
        u.GetClientRect.restype = wintypes.BOOL
        u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(_POINT)]
        u.ClientToScreen.restype = wintypes.BOOL
        u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.MonitorFromWindow.restype = wintypes.HMONITOR
        u.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFOEXW)]
        u.GetMonitorInfoW.restype = wintypes.BOOL
        u.GetForegroundWindow.argtypes = []
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD

        if self.shcore is not None and hasattr(self.shcore, "SetProcessDpiAwareness"):
            self.shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
            self.shcore.SetProcessDpiAwareness.restype = ctypes.c_long

        k = self.kernel32
        k.GetCurrentProcess.argtypes = []
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL

        a = self.advapi32
        a.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        a.OpenProcessToken.restype = wintypes.BOOL
        a.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        a.GetTokenInformation.restype = wintypes.BOOL


_WIN32_API: _Win32Api | None = None
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM) if sys.platform == "win32" else None


def _get_win32() -> _Win32Api:
    global _WIN32_API
    if _WIN32_API is None:
        _WIN32_API = _Win32Api()
    return _WIN32_API


def enable_dpi_awareness() -> bool:
    """Enable Per-Monitor-V2 DPI awareness idempotently on Windows."""
    if sys.platform != "win32":
        return False

    api = _get_win32()
    user32 = api.user32

    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        try:
            ctypes.set_last_error(0)
            ok = user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            )
            if ok or ctypes.get_last_error() == ERROR_ACCESS_DENIED:
                return True
        except (OSError, AttributeError):
            pass

    if api.shcore is not None and hasattr(api.shcore, "SetProcessDpiAwareness"):
        try:
            hr = int(api.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE))
            if hr == 0 or (hr & 0xFFFFFFFF) == E_ACCESSDENIED_U32:
                return True
        except (OSError, AttributeError):
            pass

    if hasattr(user32, "SetProcessDPIAware"):
        try:
            return bool(user32.SetProcessDPIAware())
        except (OSError, AttributeError):
            return False

    return False


def _get_window_title(api: _Win32Api, hwnd: int) -> str:
    length = int(api.user32.GetWindowTextLengthW(hwnd))
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    copied = int(api.user32.GetWindowTextW(hwnd, buf, length + 1))
    if copied <= 0:
        return ""
    return buf.value


def _get_window_class(api: _Win32Api, hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    copied = int(api.user32.GetClassNameW(hwnd, buf, len(buf)))
    if copied <= 0:
        return ""
    return buf.value


def _get_process_basename(api: _Win32Api, hwnd: int) -> str | None:
    pid = wintypes.DWORD(0)
    api.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        return None
    hproc = api.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not hproc:
        return None
    try:
        buf_len = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(buf_len.value)
        if not api.kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(buf_len)):
            return None
        return PureWindowsPath(buf.value).name
    finally:
        api.kernel32.CloseHandle(hproc)


def find_vrchat_window(
    title_query: str = "VRChat",
    class_name: str | None = "UnityWndClass",
    *,
    process_name: str | None = "VRChat.exe",
    include_minimized: bool = False,
) -> int | None:
    """Locate the best matching visible VRChat top-level window HWND."""
    if sys.platform != "win32":
        return None

    query_norm = title_query.strip().lower()
    if not query_norm:
        return None

    api = _get_win32()
    user32 = api.user32
    hwnds: list[int] = []

    enum_cb_type = _WNDENUMPROC or ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _on_window(hwnd_val: Any, _lparam: int) -> bool:
        if hwnd_val:
            hwnds.append(int(hwnd_val))
        return True

    cb = enum_cb_type(_on_window)
    ctypes.set_last_error(0)
    ret = user32.EnumWindows(cb, 0)
    if ret == 0 and ctypes.get_last_error() != 0 and not hwnds:
        return None

    class_norm = class_name.lower() if class_name else None
    proc_norm = process_name.lower() if process_name is not None else None

    best_hwnd: int | None = None
    best_score: tuple[int, int, int] = (-1, -1, -1)

    for hwnd in hwnds:
        if not user32.IsWindowVisible(hwnd):
            continue
        if not include_minimized and user32.IsIconic(hwnd):
            continue

        title = _get_window_title(api, hwnd)
        if not title:
            continue
        title_lower = title.lower()
        if query_norm not in title_lower:
            continue

        cls = _get_window_class(api, hwnd)
        class_match = class_norm is not None and cls.lower() == class_norm
        exact_title = title.strip().lower() == query_norm

        proc_match = False
        if proc_norm is not None and (class_match or class_norm is None or exact_title):
            img_name = _get_process_basename(api, hwnd)
            proc_match = img_name is not None and img_name.lower() == proc_norm

        if class_norm is not None and not class_match:
            if not (proc_norm is not None and proc_match):
                continue
        elif class_norm is None and proc_norm is not None and not proc_match:
            continue

        score = (int(proc_match), int(exact_title), int(class_match))
        if score > best_score:
            best_score = score
            best_hwnd = hwnd

    return best_hwnd


def get_client_rect(hwnd: int) -> WindowRect | None:
    """Return physical screen-space client rectangle for a non-minimized window."""
    if sys.platform != "win32" or hwnd <= 0:
        return None

    api = _get_win32()
    user32 = api.user32
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None

    rc = _RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    width = int(rc.right - rc.left)
    height = int(rc.bottom - rc.top)
    if width <= 0 or height <= 0:
        return None

    pt = _POINT(rc.left, rc.top)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None

    return WindowRect(
        left=int(pt.x),
        top=int(pt.y),
        right=int(pt.x) + width,
        bottom=int(pt.y) + height,
        hwnd=int(hwnd),
    )


def get_monitor_rect(hwnd: int) -> WindowRect | None:
    """Return physical screen-space rectangle of the monitor nearest to hwnd."""
    if sys.platform != "win32" or hwnd <= 0:
        return None

    api = _get_win32()
    user32 = api.user32
    if not user32.IsWindow(hwnd):
        return None

    hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not hmon:
        return None

    mi = _MONITORINFOEXW()
    mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
    if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
        return None

    return WindowRect(
        left=int(mi.rcMonitor.left),
        top=int(mi.rcMonitor.top),
        right=int(mi.rcMonitor.right),
        bottom=int(mi.rcMonitor.bottom),
        hwnd=int(hwnd),
    )


def is_foreground_window(hwnd: int) -> bool:
    """Return True iff hwnd is non-zero and currently the foreground window."""
    if sys.platform != "win32" or hwnd <= 0:
        return False

    api = _get_win32()
    fg = api.user32.GetForegroundWindow()
    return bool(fg) and int(fg) == int(hwnd)


def is_current_process_elevated() -> bool | None:
    """Return True if current Python process has elevated token, False if not, None if unavailable."""
    if sys.platform != "win32":
        return None

    api = _get_win32()
    hproc = api.kernel32.GetCurrentProcess()
    htok = wintypes.HANDLE()
    if not api.advapi32.OpenProcessToken(hproc, TOKEN_QUERY, ctypes.byref(htok)):
        return None
    try:
        elev = _TOKEN_ELEVATION()
        ret_len = wintypes.DWORD(0)
        if not api.advapi32.GetTokenInformation(
            htok,
            TOKEN_ELEVATION_INFO_CLASS,
            ctypes.byref(elev),
            ctypes.sizeof(elev),
            ctypes.byref(ret_len),
        ):
            return None
        return bool(elev.TokenIsElevated)
    finally:
        api.kernel32.CloseHandle(htok)


def is_target_elevated(hwnd: int) -> bool | None:
    """Detect whether the process owning hwnd is elevated (UIPI check)."""
    if sys.platform != "win32" or hwnd <= 0:
        return None

    api = _get_win32()
    if not api.user32.IsWindow(hwnd):
        return None

    pid = wintypes.DWORD(0)
    api.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        return None

    ctypes.set_last_error(0)
    hproc = api.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not hproc:
        err = ctypes.get_last_error()
        return True if err == ERROR_ACCESS_DENIED else None

    try:
        htok = wintypes.HANDLE()
        ctypes.set_last_error(0)
        if not api.advapi32.OpenProcessToken(hproc, TOKEN_QUERY, ctypes.byref(htok)):
            err = ctypes.get_last_error()
            return True if err == ERROR_ACCESS_DENIED else None
        try:
            elev = _TOKEN_ELEVATION()
            ret_len = wintypes.DWORD(0)
            if not api.advapi32.GetTokenInformation(
                htok,
                TOKEN_ELEVATION_INFO_CLASS,
                ctypes.byref(elev),
                ctypes.sizeof(elev),
                ctypes.byref(ret_len),
            ):
                return None
            return bool(elev.TokenIsElevated)
        finally:
            api.kernel32.CloseHandle(htok)
    finally:
        api.kernel32.CloseHandle(hproc)
