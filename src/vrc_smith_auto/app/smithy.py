from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable

from vrc_smith_auto.protocols import FramePacket, SmithyInput
from vrc_smith_auto.vision.smithy import SmithyVision
from vrc_smith_auto.window import WindowRect


class SmithyAutomation:
    """Advance repair and selling using fresh frames, pausing the gauge meanwhile."""

    def __init__(self, mouse: SmithyInput, auto_sell: bool) -> None:
        self._mouse = mouse
        self._auto_sell = auto_sell
        self._vision = SmithyVision(auto_sell)
        self._state = "gauge"
        self._next_probe = 0.0
        self._deadline = 0.0
        self._previous_count = 13

    def _click(self, rect: WindowRect | None, point: tuple[int, int], *, double: bool = False) -> bool:
        for index in range(2 if double else 1):
            if index:
                time.sleep(0.045)
            if not self._mouse.move_cursor(rect, *point):
                return False
            now = time.perf_counter()
            if not self._mouse.press_left(35, timestamp_s=now):
                return False
            try:
                time.sleep(0.035)
            finally:
                self._mouse.tick(now + 0.035)
        return True

    def _enter(self, state: str, now: float, delay: float = 0.12) -> None:
        self._state = state
        self._next_probe = now + delay
        self._deadline = now + 8.0

    def _client_frame(self, packet: FramePacket, client: WindowRect | None) -> FramePacket | None:
        # Offline sources supply image coordinates. Live frames can include the
        # whole monitor, but actions require the whole VRChat client to be visible.
        if packet.screen_rect is None:
            return packet
        captured = packet.screen_rect
        if client is None or not client.is_valid:
            return None
        x, y = client.left - captured.left, client.top - captured.top
        if x < 0 or y < 0 or x + client.width > packet.width or y + client.height > packet.height:
            return None
        return replace(packet, image_bgr=packet.image_bgr[y:y + client.height, x:x + client.width],
                       screen_rect=client)

    def allows_gauge(
        self, packet: FramePacket, client_rect: Callable[[], WindowRect | None] | None = None
    ) -> bool:
        now = packet.timestamp_s
        if now < self._next_probe:
            return self._state == "gauge"
        probe_started = time.perf_counter()
        self._next_probe = now + 0.12
        if self._state != "gauge" and now >= self._deadline:
            raise RuntimeError(f"Smithy automation timed out during {self._state.replace('_', ' ')}")

        client = client_rect() if client_rect is not None else packet.screen_rect
        packet = self._client_frame(packet, client)
        if packet is None:
            return self._state == "gauge"
        rect = packet.screen_rect
        self._vision.prepare(packet.image_bgr)
        find = self._vision.button

        if self._state == "gauge":
            if find("hit_unavailable") is None:
                return True
            self._enter("fire_up", now)
        if self._state == "fire_up":
            point = find("fire_up")
            if point is not None and self._click(rect, point):
                elapsed = time.perf_counter() - probe_started
                self._enter("after_fire_up", now + elapsed, 0.5 if self._auto_sell else 0.12)
            return False
        elif self._state == "after_fire_up":
            if self._auto_sell and find("fire_up_unavailable") is None:
                self._enter("open_menu", now, 0.0)
            else:
                self._enter("return_to_hit", now, 0.0)
        if self._state == "open_menu":
            if self._mouse.press_tab(rect):
                self._enter("open_smithy", now, 0.25)
        elif self._state == "open_smithy":
            point = find("smithy")
            if point is not None and self._click(rect, point):
                self._previous_count = 13
                self._enter("sell", now, 0.25)
        elif self._state == "sell":
            occupied = self._vision.storage()
            if occupied == []:
                if self._mouse.press_tab(rect):
                    self._enter("after_sell", now, 0.25)
            elif occupied is not None and len(occupied) < self._previous_count:
                if self._click(rect, occupied[0], double=True):
                    self._previous_count = len(occupied)
                    self._enter("sell", now, 0.25)
        elif self._state in ("return_to_hit", "after_sell"):
            point = find("hit")
            if point is not None and self._mouse.move_cursor(rect, *point):
                self._enter("gauge", now)
                return True
            if self._state == "after_sell" and find("hit_unavailable") is not None:
                self._enter("fire_up", now)
        return False
