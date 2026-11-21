from __future__ import annotations

import ctypes
import math
import subprocess
import sys
import threading

import pytest

from vrc_smith_auto.app import (
    KEY_DOWN_MASK,
    VK_ESCAPE,
    VK_F8,
    HotkeyController,
    HotkeyEvents,
)
from vrc_smith_auto.input import (
    HARDWAREINPUT,
    INPUT,
    INPUT_MOUSE,
    KEYBDINPUT,
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MOVE,
    MOUSEINPUT,
    RecordedClick,
    SpyMouseBackend,
    Win32SendInputMouse,
    build_mouse_left_input,
)
from vrc_smith_auto.protocols import MouseBackend


# ---------------------------------------------------------------------------
# Import-Time Win32 / DPI Purity
# ---------------------------------------------------------------------------


def test_import_time_win32_dpi_purity() -> None:
    script = (
        "import sys; "
        "import vrc_smith_auto; "
        "import vrc_smith_auto.capture; "
        "import vrc_smith_auto.input; "
        "import vrc_smith_auto.app; "
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


# ---------------------------------------------------------------------------
# Win32 SendInput Struct Layout & Driver
# ---------------------------------------------------------------------------


def test_win32_input_struct_layout_and_inplace_flags() -> None:
    is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(HARDWAREINPUT) == 8
    assert ctypes.sizeof(KEYBDINPUT) == (24 if is_64 else 16)
    assert ctypes.sizeof(MOUSEINPUT) == (32 if is_64 else 24)
    assert ctypes.sizeof(INPUT) == (40 if is_64 else 28)

    assert INPUT.type.offset == 0
    assert INPUT.mi.offset == (8 if is_64 else 4)
    assert MOUSEINPUT.dwExtraInfo.offset == (24 if is_64 else 20)

    down = build_mouse_left_input(MOUSEEVENTF_LEFTDOWN)
    up = build_mouse_left_input(MOUSEEVENTF_LEFTUP)
    for inp, expected_flag in ((down, MOUSEEVENTF_LEFTDOWN), (up, MOUSEEVENTF_LEFTUP)):
        assert inp.type == INPUT_MOUSE
        assert inp.mi.dx == 0
        assert inp.mi.dy == 0
        assert inp.mi.mouseData == 0
        assert inp.mi.time == 0
        assert inp.mi.dwExtraInfo == 0
        assert inp.mi.dwFlags == expected_flag
        assert (inp.mi.dwFlags & MOUSEEVENTF_MOVE) == 0
        assert (inp.mi.dwFlags & MOUSEEVENTF_ABSOLUTE) == 0


def test_win32_sendinput_mouse_lifecycle_and_context_manager() -> None:
    emitted: list[int] = []
    now = 10.0

    mouse = Win32SendInputMouse(
        require_foreground=True,
        target_hwnd=1234,
        check_uipi=True,
        send_input_fn=lambda inp: (emitted.append(int(inp.mi.dwFlags)), 1)[1],
        get_last_error_fn=lambda: 0,
        clock_fn=lambda: now,
        is_foreground_fn=lambda hwnd: hwnd == 1234,
        is_target_elevated_fn=lambda _hwnd: False,
        is_current_elevated_fn=lambda: False,
    )
    assert isinstance(mouse, MouseBackend)
    assert mouse.is_pressed is False

    # Press initiates LEFTDOWN immediately without blocking
    assert mouse.press_left(30.0, timestamp_s=1.0) is True
    assert mouse.is_pressed is True
    assert emitted == [MOUSEEVENTF_LEFTDOWN]

    # Overlap guard prevents second LEFTDOWN
    assert mouse.press_left(30.0, timestamp_s=1.01) is False
    assert mouse.last_blocked_reason == "already_pressed"
    assert emitted == [MOUSEEVENTF_LEFTDOWN]

    # Before release deadline
    mouse.tick(1.029)
    assert mouse.is_pressed is True
    assert emitted == [MOUSEEVENTF_LEFTDOWN]

    # At release deadline
    mouse.tick(1.030)
    assert mouse.is_pressed is False
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    # Invalid duration_ms rejected before touching state or OS
    for bad in (0.0, -5.0, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            mouse.press_left(bad)  # type: ignore[arg-type]

    # Invalid timestamp_s / tick(now_s) rejected before touching state or OS
    for bad_ts in (math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError):
            mouse.press_left(30.0, timestamp_s=bad_ts)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            mouse.tick(bad_ts)  # type: ignore[arg-type]
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    # Context manager guarantees release_all() on exception & idempotency
    emitted.clear()
    with pytest.raises(RuntimeError):
        with mouse:
            assert mouse.press_left(50.0, timestamp_s=2.0) is True
            raise RuntimeError("boom")
    assert mouse.is_pressed is False
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    mouse.release_all()
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]


def test_win32_sendinput_focus_and_uipi_guards() -> None:
    emitted: list[int] = []

    # 1. Missing target window when require_foreground=True
    m_no_hwnd = Win32SendInputMouse(
        require_foreground=True,
        target_hwnd=None,
        send_input_fn=lambda inp: (emitted.append(int(inp.mi.dwFlags)), 1)[1],
    )
    assert m_no_hwnd.press_left(30.0, timestamp_s=0.0) is False
    assert m_no_hwnd.last_blocked_reason == "no_target_window"
    assert emitted == []

    # 2. Target not foreground
    m_bg = Win32SendInputMouse(
        require_foreground=True,
        hwnd_provider=lambda: 777,
        send_input_fn=lambda inp: (emitted.append(int(inp.mi.dwFlags)), 1)[1],
        is_foreground_fn=lambda _hwnd: False,
    )
    assert m_bg.press_left(30.0, timestamp_s=0.0) is False
    assert m_bg.last_blocked_reason == "not_foreground"
    assert emitted == []

    # 3. require_foreground=False allows click even without hwnd
    m_nofg = Win32SendInputMouse(
        require_foreground=False,
        target_hwnd=None,
        send_input_fn=lambda inp: (emitted.append(int(inp.mi.dwFlags)), 1)[1],
    )
    assert m_nofg.press_left(25.0, timestamp_s=0.0) is True
    assert emitted == [MOUSEEVENTF_LEFTDOWN]
    m_nofg.release_all()
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    # 4. UIPI mismatch pre-check & recovery reset when target switches to un-elevated window
    emitted.clear()
    m_uipi = Win32SendInputMouse(
        require_foreground=True,
        target_hwnd=888,
        check_uipi=True,
        send_input_fn=lambda inp: (emitted.append(int(inp.mi.dwFlags)), 1)[1],
        is_foreground_fn=lambda _hwnd: True,
        is_target_elevated_fn=lambda hwnd: hwnd == 888,
        is_current_elevated_fn=lambda: False,
    )
    assert m_uipi.press_left(30.0, timestamp_s=0.0) is False
    assert m_uipi.uipi_blocked is True
    assert m_uipi.last_blocked_reason == "uipi_elevation_mismatch"
    assert emitted == []

    m_uipi.set_target_hwnd(889)
    assert m_uipi.press_left(30.0, timestamp_s=0.1) is True
    assert m_uipi.uipi_blocked is False
    assert m_uipi.last_blocked_reason is None
    assert emitted == [MOUSEEVENTF_LEFTDOWN]
    m_uipi.release_all()
    assert emitted == [MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP]

    # 5. SendInput returning 0 with ERROR_ACCESS_DENIED (5)
    m_err5 = Win32SendInputMouse(
        require_foreground=False,
        check_uipi=False,
        send_input_fn=lambda _inp: 0,
        get_last_error_fn=lambda: 5,
    )
    assert m_err5.press_left(30.0, timestamp_s=0.0) is False
    assert m_err5.is_pressed is False
    assert m_err5.uipi_blocked is True
    assert m_err5.last_error == 5
    assert m_err5.last_blocked_reason == "uipi_access_denied"

    # 5b. SendInput returning 0 with non-5 WinError (e.g. 87) & set_target_hwnd
    m_err87 = Win32SendInputMouse(
        require_foreground=True,
        target_hwnd=None,
        check_uipi=False,
        send_input_fn=lambda _inp: 0,
        get_last_error_fn=lambda: 87,
        is_foreground_fn=lambda hwnd: hwnd == 909,
    )
    assert m_err87.press_left(30.0, timestamp_s=0.0) is False
    assert m_err87.last_blocked_reason == "no_target_window"
    m_err87.set_target_hwnd(909)
    assert m_err87.press_left(30.0, timestamp_s=0.0) is False
    assert m_err87.uipi_blocked is False
    assert m_err87.last_error == 87
    assert m_err87.last_blocked_reason == "send_input_failed_87"

    # 6. SendInput failing on LEFTUP still clears is_pressed so state never wedges
    results = [1, 0]
    m_up_fail = Win32SendInputMouse(
        require_foreground=False,
        check_uipi=False,
        send_input_fn=lambda _inp: results.pop(0),
        get_last_error_fn=lambda: 5,
    )
    assert m_up_fail.press_left(20.0, timestamp_s=1.0) is True
    assert m_up_fail.is_pressed is True
    m_up_fail.tick(1.020)
    assert m_up_fail.is_pressed is False
    assert m_up_fail.last_error == 5
    assert m_up_fail.uipi_blocked is True


def test_win32_sendinput_concurrent_release_all_thread_safety() -> None:
    downs = 0
    ups = 0
    entered_down = threading.Event()
    allow_down_return = threading.Event()

    def _send(inp: INPUT) -> int:
        nonlocal downs, ups
        if int(inp.mi.dwFlags) == MOUSEEVENTF_LEFTDOWN:
            downs += 1
            entered_down.set()
            allow_down_return.wait(timeout=1.0)
        elif int(inp.mi.dwFlags) == MOUSEEVENTF_LEFTUP:
            ups += 1
        return 1

    mouse = Win32SendInputMouse(
        require_foreground=False,
        check_uipi=False,
        send_input_fn=_send,
    )
    t_press = threading.Thread(target=lambda: mouse.press_left(50.0, timestamp_s=1.0))
    t_press.start()
    assert entered_down.wait(timeout=1.0)

    t_rel = threading.Thread(target=mouse.release_all)
    t_rel.start()
    allow_down_return.set()
    t_press.join(timeout=1.0)
    t_rel.join(timeout=1.0)

    assert downs == 1
    assert ups == 1
    assert mouse.is_pressed is False


# ---------------------------------------------------------------------------
# SpyMouseBackend
# ---------------------------------------------------------------------------


def test_spy_mouse_backend_recording_and_lifecycle() -> None:
    clock_val = 5.0
    spy = SpyMouseBackend(clock_fn=lambda: clock_val)
    assert isinstance(spy, MouseBackend)
    assert spy.is_pressed is False
    assert spy.click_count == 0
    assert spy.clicks == ()

    spy.set_frame_idx(42)
    assert spy.press_left(30.0, timestamp_s=0.7) is True
    assert spy.is_pressed is True
    assert spy.click_count == 1
    assert spy.clicks[0] == RecordedClick(
        timestamp_s=0.7,
        duration_ms=30.0,
        release_timestamp_s=None,
        released=False,
        frame_idx=42,
    )

    # Overlap returns False
    assert spy.press_left(30.0, timestamp_s=0.71) is False
    assert spy.click_count == 1

    spy.tick(0.72)
    assert spy.is_pressed is True
    spy.tick(0.73)
    assert spy.is_pressed is False
    assert spy.clicks[0].released is True
    assert spy.clicks[0].release_timestamp_s == pytest.approx(0.73)

    # Context manager releases active hold
    with spy:
        spy.set_frame_idx(99)
        assert spy.press_left(40.0) is True
    assert spy.is_pressed is False
    assert spy.click_count == 2
    assert spy.clicks[1].released is True
    assert spy.clicks[1].release_timestamp_s == pytest.approx(5.0)
    assert spy.clicks[1].frame_idx == 99

    # Idempotent release_all and clear
    spy.release_all()
    spy.clear()
    assert spy.click_count == 0
    assert spy.clicks == ()

    with pytest.raises(ValueError):
        spy.press_left(0.0)
    for bad_ts in (math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError):
            spy.press_left(30.0, timestamp_s=bad_ts)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            spy.tick(bad_ts)  # type: ignore[arg-type]

    # Out-of-order earlier tick(now_s) must keep _last_now_s monotonic so release_timestamp_s >= timestamp_s
    spy.clear()
    assert spy.press_left(30.0, timestamp_s=1.0) is True
    spy.tick(0.5)
    assert spy.is_pressed is True
    spy.release_all()
    assert spy.is_pressed is False
    assert spy.clicks[0].release_timestamp_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HotkeyController
# ---------------------------------------------------------------------------


def test_hotkey_controller_rising_edge_and_bitmask() -> None:
    states: dict[int, int] = {VK_F8: 0, VK_ESCAPE: 0}
    ctrl = HotkeyController(key_state_fn=lambda vk: states.get(vk, 0))

    assert KEY_DOWN_MASK == 0x8000
    assert ctrl.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=False)

    # Bit 0x0001 ("was pressed since last call") alone MUST NOT trigger
    states[VK_F8] = 0x0001
    states[VK_ESCAPE] = 0x0001
    assert ctrl.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=False)

    # Rising edge on 0x8000 triggers once; holding across polls does not re-trigger
    states[VK_F8] = 0x8000
    assert ctrl.poll() == HotkeyEvents(toggle_armed=True, emergency_stop=False)
    states[VK_F8] = -32768  # signed SHORT representation of 0x8000
    assert ctrl.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=False)

    # Release then re-press F8, simultaneous with Esc (0x8001)
    states[VK_F8] = 0
    assert ctrl.poll() == HotkeyEvents(toggle_armed=False, emergency_stop=False)
    states[VK_F8] = 0x8001
    states[VK_ESCAPE] = -32768
    assert ctrl.poll() == HotkeyEvents(toggle_armed=True, emergency_stop=True)

    # reset() clears previous held state so still-held key fires again cleanly
    ctrl.reset()
    assert ctrl.poll() == HotkeyEvents(toggle_armed=True, emergency_stop=True)
