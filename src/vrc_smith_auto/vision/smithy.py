from pathlib import Path

import cv2
import numpy as np

_IMAGES = Path(__file__).resolve().parents[1] / "images"
# Slot centers measured in each reference. The title and footer locate the panel
# without relying on the count, item icons, or item colors.
_PANELS = (
    ("empty_storage", (43, 98, 154, 209), (61, 117, 172), (13, 4, 123, 27), (49, 203, 208, 238)),
    ("full_storage", (35, 91, 146, 201), (57, 112, 166), (5, 3, 115, 23), (48, 198, 200, 233)),
)


def _load(name: str) -> np.ndarray:
    image = cv2.imread(str(_IMAGES / f"{name}.png"))
    if image is None:
        raise ValueError(f"Could not read Smithy reference image: {name}.png")
    return image


def _match(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    if not template.size or any(length < size for length, size in zip(image.shape[:2], template.shape[:2])):
        return 1.0, (0, 0)
    score, _, origin, _ = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED))
    return score, origin


class SmithyVision:
    def __init__(self, auto_sell: bool) -> None:
        names = ["hit", "hit_unavailable", "fire_up"]
        if auto_sell:
            names += ["fire_up_unavailable", "smithy"]
        self._buttons = {name: cv2.cvtColor(_load(name), cv2.COLOR_BGR2GRAY) for name in names}
        self._panels = {name: _load(name) for name, *_ in _PANELS} if auto_sell else {}
        self._background = self._panels["empty_storage"][61, 43] if auto_sell else None
        self._size = (0, 0)

    def prepare(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self._frame = frame
        self._gray = None
        self._coarse = None
        if (width, height) != self._size:
            self._size = (width, height)
            self._sx, self._sy = width / 1920, height / 1080
            self._scaled_buttons = {
                name: self._resize(image, self._sx, self._sy)
                for name, image in self._buttons.items()
            }
            self._coarse_hit = self._resize(self._buttons["hit_unavailable"], self._sx / 2, self._sy / 2)
            self._scaled_panels = {
                name: self._resize(image, self._sx, self._sy) for name, image in self._panels.items()
            }

    @staticmethod
    def _resize(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(image, (max(1, round(w * sx)), max(1, round(h * sy))),
                          interpolation=cv2.INTER_AREA if max(sx, sy) <= 1 else cv2.INTER_LINEAR)

    def button(self, name: str) -> tuple[int, int] | None:
        template = self._scaled_buttons[name]
        width, height = self._size
        h, w = template.shape
        x0 = y0 = 0
        if name == "hit_unavailable":
            if self._coarse is None:
                small = cv2.resize(self._frame, (max(1, width // 2), max(1, height // 2)),
                                   interpolation=cv2.INTER_AREA)
                self._coarse = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            coarse_score, (cx, cy) = _match(self._coarse, self._coarse_hit)
            if coarse_score > 0.17:
                return None
            x0, y0 = max(0, 2 * cx - 6), max(0, 2 * cy - 6)
            patch = self._frame[y0:2 * cy + h + 6, x0:2 * cx + w + 6]
            image = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            if self._gray is None:
                self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
            image = self._gray
        score, (x, y) = _match(image, template)
        if score > 0.10:
            return None
        return x0 + x + w // 2, y0 + y + h // 2

    def storage(self) -> list[tuple[int, int]] | None:
        """Return occupied slot centers, or None if the storage panel is unconfirmed."""
        matches = []
        for name, xs, ys, title, footer in _PANELS:
            panel = self._scaled_panels[name]
            h, w = panel.shape[:2]
            tx0, ty0, tx1, ty1 = self._scaled_bounds(title)
            fx0, fy0, fx1, fy1 = self._scaled_bounds(footer)
            score, (x, y) = _match(self._frame, panel[ty0:ty1, tx0:tx1])
            x, y = x - tx0, y - ty0
            if score > 0.10 or x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
                continue
            footer_template = panel[fy0:fy1, fx0:fx1]
            # Independently confirm the footer near the location predicted by the title.
            margin = 3
            footer_image = self._frame[max(0, y + fy0 - margin):y + fy1 + margin,
                                       max(0, x + fx0 - margin):x + fx1 + margin]
            footer_score, _ = _match(footer_image, footer_template)
            if footer_score <= 0.12:
                matches.append((score + footer_score, x, y, xs, ys))
        if not matches:
            return None

        _, x, y, xs, ys = min(matches)
        occupied = []
        for cy in ys:
            for cx in xs:
                x0, y0, x1, y1 = self._scaled_bounds((cx - 13, cy - 13, cx + 14, cy + 14))
                patch = self._frame[y + y0:y + y1, x + x0:x + x1]
                if not patch.size:
                    return None
                difference = np.max(np.abs(patch.astype(np.int16) - self._background), axis=2)
                if np.mean(difference > 17) > 0.07:
                    occupied.append((x + round(cx * self._sx), y + round(cy * self._sy)))
        return occupied

    def _scaled_bounds(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        return round(x0 * self._sx), round(y0 * self._sy), round(x1 * self._sx), round(y1 * self._sy)
