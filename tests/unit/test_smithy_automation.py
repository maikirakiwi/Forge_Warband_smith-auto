from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vrc_smith_auto.app.runner import Runner
from vrc_smith_auto.app.smithy import SmithyAutomation
from vrc_smith_auto.config import AppConfig
from vrc_smith_auto.input import SpyMouseBackend
from vrc_smith_auto.input import win32_sendinput
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.smithy import SmithyVision, _load
from vrc_smith_auto.window import WindowRect


def scene(size, *placements):
    width, height = size
    frame = np.full((height, width, 3), 38, np.uint8)
    for reference, x, y in placements:
        image = _load(reference) if isinstance(reference, str) else reference
        scaled = cv2.resize(
            image, (round(image.shape[1] * width / 1920), round(image.shape[0] * height / 1080)),
            interpolation=cv2.INTER_AREA if width < 1920 else cv2.INTER_LINEAR,
        )
        frame[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    return frame


class RecordingInput(SpyMouseBackend):
    def __init__(self):
        super().__init__()
        self.events = []
        self.cursor = None

    def move_cursor(self, rect, x, y):
        self.cursor = x, y
        return True

    def press_left(self, duration_ms, timestamp_s=None):
        self.events.append(self.cursor)
        return super().press_left(duration_ms, timestamp_s)

    def press_tab(self, rect):
        self.events.append("Tab")
        return True


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1024, 768), (800, 450), (2560, 1440)])
def test_scaled_templates_distinguish_button_states_and_storage(size):
    vision = SmithyVision(auto_sell=True)
    for name, other in [("hit", "hit_unavailable"), ("hit_unavailable", "hit"),
                        ("fire_up", "fire_up_unavailable"), ("fire_up_unavailable", "fire_up")]:
        vision.prepare(scene(size, (name, 231, 117)))
        assert vision.button(name) is not None
        assert vision.button(other) is None

    vision.prepare(scene(size, ("empty_storage", 231, 117)))
    assert vision.storage() == []
    vision.prepare(scene(size, ("full_storage", 231, 117)))
    assert len(vision.storage()) == 12
    vision.prepare(scene(size))
    assert vision.storage() is None


def test_runner_sells_mixed_storage_then_resumes_gold(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    size = (1280, 720)
    empty = _load("empty_storage")
    mixed = empty.copy()
    mixed[46:76, 29:57] = (185, 29, 101)
    mixed[102:132, 84:112] = (19, 161, 61)
    one_left = mixed.copy()
    one_left[46:76, 29:57] = empty[46:76, 29:57]
    forge = scene(size, ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    smithy = scene(size, ("smithy", 231, 117))
    two_items = scene(size, (mixed, 231, 117))
    hit = scene(size, ("hit", 231, 117))
    replay = [
        (0.0, forge), (0.499, hit), (0.6, hit), (0.9, smithy),
        (1.2, two_items), (1.5, two_items), (1.8, scene(size, (one_left, 231, 117))),
        (2.1, scene(size, (empty, 231, 117))), (2.4, hit),
    ]
    frames = iter(FramePacket(image, ts, i) for i, (ts, image) in enumerate(replay))
    mouse = RecordingInput()
    seen = []
    runner = Runner(
        source=SimpleNamespace(read=lambda: next(frames, None)),
        mouse=mouse,
        config=AppConfig.from_dict({"trigger": {"auto_fire_up": True, "auto_sell": True}}),
        stop_on_eof=True,
        detector=SimpleNamespace(
            reset=lambda: None,
            detect_packet=lambda packet: seen.append(packet.frame_idx) or SimpleNamespace(gauge_present=True),
        ),
        state_machine=SimpleNamespace(
            reset=lambda: None,
            update=lambda *args, **kwargs: SimpleNamespace(should_click=True, hold_ms=1, release_delay_s=0),
        ),
    )
    result = runner.run()
    assert seen == [8]
    assert result.click_frames == (8,)
    assert mouse.events == [
        (268, 243), "Tab", (325, 140),
        (260, 158), (260, 158), (296, 195), (296, 195), "Tab", (295, 140),
    ]


@pytest.mark.parametrize("auto_sell", [False, True])
def test_fire_up_resumes_when_selling_is_disabled_or_fire_up_is_unavailable(monkeypatch, auto_sell):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    mouse = RecordingInput()
    automation = SmithyAutomation(mouse, auto_sell)
    size = (1280, 720)
    forge = scene(size, ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    ready = scene(size, ("hit", 231, 117), ("fire_up_unavailable", 231, 225))
    assert not automation.allows_gauge(FramePacket(forge, 0.0, 0))
    assert automation.allows_gauge(FramePacket(ready, 0.6, 1))
    assert mouse.events == [(268, 243)]
    assert mouse.cursor == (295, 140)


def test_fire_up_waits_for_enabled_button_and_accepts_height_scaled_ui(monkeypatch):
    monkeypatch.setattr("vrc_smith_auto.app.smithy.time", SimpleNamespace(sleep=lambda _: None, perf_counter=lambda: 0))
    mouse = RecordingInput()
    automation = SmithyAutomation(mouse, auto_sell=True)
    size = (1024, 768)
    disabled = scene(size, ("hit_unavailable", 231, 117), ("fire_up_unavailable", 231, 225))
    enabled = scene(size, ("hit_unavailable", 231, 117))
    image = _load("fire_up")
    scaled = cv2.resize(image, None, fx=768 / 1080, fy=768 / 1080, interpolation=cv2.INTER_AREA)
    h, w = scaled.shape[:2]
    enabled[225:225 + h, 231:231 + w] = scaled

    assert not automation.allows_gauge(FramePacket(disabled, 0.0, 0))
    assert not automation.allows_gauge(FramePacket(disabled, 9.0, 1))
    assert mouse.events == []
    assert not automation.allows_gauge(FramePacket(enabled, 9.2, 2))
    assert mouse.events == [(231 + w // 2, 225 + h // 2)]


def test_fire_up_timeout_reports_when_input_is_blocked():
    mouse = RecordingInput()
    mouse.move_cursor = lambda *args: False
    mouse.last_blocked_reason = "not_foreground"
    automation = SmithyAutomation(mouse, auto_sell=True)
    forge = scene((1280, 720), ("hit_unavailable", 231, 117), ("fire_up", 231, 225))
    assert not automation.allows_gauge(FramePacket(forge, 0.0, 0))
    with pytest.raises(RuntimeError, match="input was blocked.*not_foreground"):
        automation.allows_gauge(FramePacket(forge, 8.1, 1))


def test_ui_input_uses_client_origin_and_requires_current_foreground_window(monkeypatch):
    rect = WindowRect.from_xywh(-1260, 90, 1280, 720, hwnd=42)
    foreground = False
    sent = []
    moves = []
    monkeypatch.setattr(win32_sendinput, "get_client_rect", lambda _: rect)
    monkeypatch.setattr(win32_sendinput, "time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr(
        win32_sendinput.ctypes, "windll",
        SimpleNamespace(user32=SimpleNamespace(SetCursorPos=lambda x, y: moves.append((x, y)) or 1)),
        raising=False,
    )
    mouse = win32_sendinput.Win32SendInputMouse(
        target_hwnd=42,
        check_uipi=False,
        is_foreground_fn=lambda _: foreground,
        send_input_fn=lambda event: sent.append((event.type, event.ki.wVk, event.ki.dwFlags)) or 1,
        get_last_error_fn=lambda: 0,
    )
    assert not mouse.move_cursor(rect, 300, 120)
    assert not mouse.press_tab(rect)
    foreground = True
    assert not mouse.move_cursor(WindowRect.from_xywh(-1261, 90, 1280, 720, hwnd=42), 300, 120)
    assert mouse.move_cursor(rect, 300, 120)
    assert mouse.press_tab(rect)
    assert moves == [(-960, 210)]
    assert sent == [(1, 9, 0), (1, 9, 2)]
