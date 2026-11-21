from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ALL_DISTRACTORS: tuple[str, ...] = (
    "pct_labels",
    "hammer_btn",
    "progress_boxes",
    "master_squares",
    "medals",
)


@dataclass(frozen=True)
class GaugePalette:
    """Calibrated BGR palette matching VRChat blacksmith minigame media."""

    bg_slate_bgr: tuple[int, int, int] = (24, 24, 28)
    black_track_bgr: tuple[int, int, int] = (12, 11, 18)
    dark_gray_bgr: tuple[int, int, int] = (32, 34, 41)
    silver_bgr: tuple[int, int, int] = (184, 177, 174)
    gold_bgr: tuple[int, int, int] = (95, 207, 249)
    red_bgr: tuple[int, int, int] = (74, 72, 239)
    distractor_hammer_bgr: tuple[int, int, int] = (70, 140, 243)
    distractor_pct_bgr: tuple[int, int, int] = (72, 91, 255)
    distractor_medal_bgr: tuple[int, int, int] = (95, 207, 249)
    distractor_progress_bgr: tuple[int, int, int] = (60, 190, 250)


VIDEO_PALETTE = GaugePalette()

SCREENSHOT_PALETTE = GaugePalette(
    bg_slate_bgr=(22, 24, 28),
    black_track_bgr=(14, 14, 20),
    dark_gray_bgr=(42, 50, 59),
    silver_bgr=(185, 178, 174),
    gold_bgr=(90, 207, 255),
    red_bgr=(73, 90, 255),
    distractor_hammer_bgr=(70, 140, 243),
    distractor_pct_bgr=(73, 90, 255),
    distractor_medal_bgr=(90, 207, 255),
    distractor_progress_bgr=(60, 190, 250),
)


@dataclass(frozen=True)
class GaugeSpec:
    """Geometry and rendering specification for a synthetic 5-band gauge frame."""

    width: int = 1920
    height: int = 1080
    cx: float = 850.0
    cy: float = 410.0
    tilt_deg: float = 0.0
    scale: float = 1.0
    bar_width: float = 38.0
    track_len: float = 184.0
    colored_len: float = 150.0
    silver_pad: float = 28.0
    gold_offset: float = 0.0
    gold_len: float = 30.0
    red_pos: float | None = 0.0
    red_width: float = 62.0
    red_thickness: float = 7.0
    show_gauge: bool = True
    distractors: bool | tuple[str, ...] = False
    palette: GaugePalette = VIDEO_PALETTE
    seed: int | None = None


@dataclass(frozen=True)
class StrikeSpec:
    """Single forging strike segment for synthetic multi-strike MP4 generation."""

    frames: int
    cx: float = 850.0
    cy: float = 410.0
    tilt_deg: float = 0.0
    gold_offset: float = 0.0
    gold_len: float = 30.0
    red_start: float | None = None
    red_velocity_px_per_frame: float = 3.5
    bounce_bounds: tuple[float, float] = (-72.0, 72.0)
    scale: float = 1.0
    distractors: bool | tuple[str, ...] = False
    show_gauge: bool = True
    bar_width: float = 38.0
    red_width: float = 62.0
    red_thickness: float = 7.0


def expected_norm_pos(spec: GaugeSpec) -> float | None:
    """Return ground-truth normalized position s_norm inside Gold [-1.0, 1.0]."""
    if not spec.show_gauge or spec.red_pos is None or spec.gold_len <= 0.0:
        return None
    return (spec.red_pos - spec.gold_offset) / (spec.gold_len / 2.0)


def red_pos_for_norm(spec: GaugeSpec, norm_pos: float) -> float:
    """Compute signed red_pos along gauge axis u corresponding to norm_pos."""
    return spec.gold_offset + norm_pos * (spec.gold_len / 2.0)


def _axes_from_tilt(tilt_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    rad = math.radians(tilt_deg)
    u = (math.sin(rad), math.cos(rad))
    v = (math.cos(rad), -math.sin(rad))
    return u, v


def _oriented_rect_pts(
    center_xy: tuple[float, float],
    u: tuple[float, float],
    v: tuple[float, float],
    length_u: float,
    width_v: float,
) -> Any:
    import numpy as np

    hu = length_u * 0.5
    hw = width_v * 0.5
    cx, cy = center_xy
    ux, uy = u
    vx, vy = v
    corners = [
        (cx - hu * ux - hw * vx, cy - hu * uy - hw * vy),
        (cx - hu * ux + hw * vx, cy - hu * uy + hw * vy),
        (cx + hu * ux + hw * vx, cy + hu * uy + hw * vy),
        (cx + hu * ux - hw * vx, cy + hu * uy - hw * vy),
    ]
    return np.array([[int(round(px)), int(round(py))] for px, py in corners], dtype=np.int32)


def _shift_point(
    center_xy: tuple[float, float],
    u: tuple[float, float],
    offset_u: float,
) -> tuple[float, float]:
    return (center_xy[0] + offset_u * u[0], center_xy[1] + offset_u * u[1])


def _resolve_distractors(spec_distractors: bool | tuple[str, ...]) -> tuple[str, ...]:
    if spec_distractors is False:
        return ()
    if spec_distractors is True:
        return ALL_DISTRACTORS
    return tuple(spec_distractors)


def _overlaps_box(
    ax1: int,
    ay1: int,
    ax2: int,
    ay2: int,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
) -> bool:
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def _pick_non_overlapping_pos(
    candidates: Sequence[tuple[int, int]],
    box_w: int,
    box_h: int,
    excl_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int]:
    for x, y in candidates:
        x_clamped = max(8, min(width - box_w - 8, x))
        y_clamped = max(8, min(height - box_h - 8, y))
        if excl_box is None:
            return (x_clamped, y_clamped)
        if not _overlaps_box(
            x_clamped,
            y_clamped,
            x_clamped + box_w,
            y_clamped + box_h,
            *excl_box,
        ):
            return (x_clamped, y_clamped)
    # Fallback corner furthest from exclusion center
    if excl_box is not None:
        ecx = (excl_box[0] + excl_box[2]) * 0.5
        ecy = (excl_box[1] + excl_box[3]) * 0.5
        tx = 16 if ecx > width * 0.5 else max(16, width - box_w - 16)
        ty = 16 if ecy > height * 0.5 else max(16, height - box_h - 16)
        return (tx, ty)
    return (16, 16)


def _draw_distractors(
    img: Any,
    spec: GaugeSpec,
    enabled: tuple[str, ...],
) -> None:
    import cv2

    if not enabled:
        return

    s = spec.scale
    w = spec.width
    h = spec.height
    pal = spec.palette

    if spec.show_gauge:
        excl_half = int(math.ceil(max(spec.track_len, spec.red_width) * s * 0.65 + 48 * s))
        excl_box: tuple[int, int, int, int] | None = (
            int(spec.cx) - excl_half,
            int(spec.cy) - excl_half,
            int(spec.cx) + excl_half,
            int(spec.cy) + excl_half,
        )
    else:
        excl_box = None

    if "pct_labels" in enabled:
        bw, bh = int(120 * s), int(56 * s)
        px, py = _pick_non_overlapping_pos(
            [
                (int(spec.cx + 180 * s), int(spec.cy - 40 * s)),
                (int(w * 0.72), int(h * 0.36)),
                (int(w * 0.12), int(h * 0.36)),
            ],
            bw,
            bh,
            excl_box,
            w,
            h,
        )
        cv2.putText(
            img,
            "0%",
            (px, py + int(42 * s)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, 1.35 * s),
            pal.distractor_pct_bgr,
            max(2, int(3 * s)),
            cv2.LINE_8,
        )

    if "hammer_btn" in enabled:
        bw, bh = int(130 * s), int(50 * s)
        px, py = _pick_non_overlapping_pos(
            [
                (int(w * 0.45), int(h * 0.78)),
                (int(w * 0.20), int(h * 0.80)),
                (int(w * 0.75), int(h * 0.80)),
            ],
            bw,
            bh,
            excl_box,
            w,
            h,
        )
        cv2.rectangle(img, (px, py), (px + bw, py + bh), pal.distractor_hammer_bgr, -1, lineType=cv2.LINE_8)
        cv2.putText(
            img,
            "HIT!",
            (px + int(18 * s), py + int(35 * s)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, 0.9 * s),
            (25, 25, 30),
            max(1, int(2 * s)),
            cv2.LINE_8,
        )

    if "progress_boxes" in enabled:
        bw, bh = int(340 * s), int(38 * s)
        px, py = _pick_non_overlapping_pos(
            [
                (int(w * 0.36), int(24 * s)),
                (int(w * 0.36), int(h - bh - 24 * s)),
                (int(24 * s), int(24 * s)),
            ],
            bw,
            bh,
            excl_box,
            w,
            h,
        )
        step = max(20, int(32 * s))
        box_sz = max(14, int(24 * s))
        for idx in range(10):
            bx = px + idx * step
            color = pal.distractor_progress_bgr if idx < 5 else (95, 95, 105)
            cv2.rectangle(img, (bx, py), (bx + box_sz, py + box_sz), color, -1, lineType=cv2.LINE_8)

    if "master_squares" in enabled:
        bw, bh = int(170 * s), int(56 * s)
        px, py = _pick_non_overlapping_pos(
            [
                (int(24 * s), int(h - bh - 28 * s)),
                (int(w - bw - 24 * s), int(28 * s)),
            ],
            bw,
            bh,
            excl_box,
            w,
            h,
        )
        sq = max(18, int(36 * s))
        for idx in range(3):
            sx = px + idx * int(48 * s)
            cv2.rectangle(img, (sx, py), (sx + sq, py + sq), pal.distractor_progress_bgr, -1, lineType=cv2.LINE_8)

    if "medals" in enabled:
        bw, bh = int(150 * s), int(64 * s)
        px, py = _pick_non_overlapping_pos(
            [
                (int(w - bw - 32 * s), int(h - bh - 32 * s)),
                (int(32 * s), int(32 * s)),
            ],
            bw,
            bh,
            excl_box,
            w,
            h,
        )
        r = max(10, int(20 * s))
        for idx in range(2):
            mcx = px + int(30 * s) + idx * int(62 * s)
            mcy = py + int(32 * s)
            cv2.circle(img, (mcx, mcy), r, pal.distractor_medal_bgr, -1, lineType=cv2.LINE_8)


def render_gauge_frame(spec: GaugeSpec) -> Any:
    """Render a deterministic BGR image according to GaugeSpec."""
    import cv2
    import numpy as np

    img = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    img[:, :] = spec.palette.bg_slate_bgr

    if spec.seed is not None:
        rng = np.random.default_rng(spec.seed)
        step = max(32, int(64 * spec.scale))
        shade = tuple(int(max(8, min(55, c + int(rng.integers(4, 12))))) for c in spec.palette.bg_slate_bgr)
        for gx in range(0, spec.width, step):
            cv2.line(img, (gx, 0), (gx, spec.height - 1), shade, 1, lineType=cv2.LINE_8)
        for gy in range(0, spec.height, step):
            cv2.line(img, (0, gy), (spec.width - 1, gy), shade, 1, lineType=cv2.LINE_8)

    if spec.show_gauge:
        s = spec.scale
        center = (spec.cx, spec.cy)
        u, v = _axes_from_tilt(spec.tilt_deg)
        pal = spec.palette
        bar_w = spec.bar_width * s

        # 1. Black track (outer)
        pts_black = _oriented_rect_pts(center, u, v, spec.track_len * s, bar_w)
        cv2.fillConvexPoly(img, pts_black, pal.black_track_bgr, lineType=cv2.LINE_8)

        # 2. Dark Gray strip
        pts_gray = _oriented_rect_pts(center, u, v, spec.colored_len * s, bar_w)
        cv2.fillConvexPoly(img, pts_gray, pal.dark_gray_bgr, lineType=cv2.LINE_8)

        # 3. Silver band (bilateral around Gold, clamped within colored_len)
        half_col = spec.colored_len * 0.5
        half_gold = spec.gold_len * 0.5
        s_top = max(-half_col, spec.gold_offset - half_gold - spec.silver_pad)
        s_bot = min(half_col, spec.gold_offset + half_gold + spec.silver_pad)
        if s_bot > s_top:
            silver_mid_u = 0.5 * (s_top + s_bot) * s
            silver_len_u = (s_bot - s_top) * s
            pts_silver = _oriented_rect_pts(
                _shift_point(center, u, silver_mid_u),
                u,
                v,
                silver_len_u,
                bar_w,
            )
            cv2.fillConvexPoly(img, pts_silver, pal.silver_bgr, lineType=cv2.LINE_8)

        # 4. Gold zone (perfect)
        g_top = max(-half_col, spec.gold_offset - half_gold)
        g_bot = min(half_col, spec.gold_offset + half_gold)
        if g_bot > g_top:
            gold_mid_u = 0.5 * (g_top + g_bot) * s
            gold_len_u = (g_bot - g_top) * s
            pts_gold = _oriented_rect_pts(
                _shift_point(center, u, gold_mid_u),
                u,
                v,
                gold_len_u,
                bar_w,
            )
            cv2.fillConvexPoly(img, pts_gold, pal.gold_bgr, lineType=cv2.LINE_8)

        # 5. Red indicator bar (drawn last so it physically bisects Gold)
        if spec.red_pos is not None:
            pts_red = _oriented_rect_pts(
                _shift_point(center, u, spec.red_pos * s),
                u,
                v,
                spec.red_thickness * s,
                spec.red_width * s,
            )
            cv2.fillConvexPoly(img, pts_red, pal.red_bgr, lineType=cv2.LINE_8)

    enabled_distractors = _resolve_distractors(spec.distractors)
    if enabled_distractors:
        _draw_distractors(img, spec, enabled_distractors)

    return img


def write_png(path: Path, frame_bgr: Any) -> Path:
    """Write a lossless PNG image to disk and verify write success."""
    import cv2

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write PNG image to {out_path}")
    return out_path


def write_synthetic_mp4(
    path: Path,
    frames: Iterable[Any],
    fps: float = 60.0,
) -> Path:
    """Encode BGR frames into an MP4 file using mp4v codec."""
    import cv2

    frame_iter = iter(frames)
    first = next(frame_iter, None)
    if first is None:
        raise ValueError("Cannot write synthetic MP4 from an empty frame sequence")

    h, w = int(first.shape[0]), int(first.shape[1])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {out_path}")

    try:
        writer.write(first)
        for frame in frame_iter:
            writer.write(frame)
    finally:
        writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Synthetic MP4 file missing or empty: {out_path}")
    return out_path


def generate_sweep_sequence(
    base_spec: GaugeSpec,
    red_positions: Sequence[float | None],
    *,
    tilt_fn: Callable[[int, float | None], float] | None = None,
    center_fn: Callable[[int, float | None], tuple[float, float]] | None = None,
    distractors_fn: Callable[[int, float | None], bool | tuple[str, ...]] | None = None,
) -> list[Any]:
    """Generate a deterministic list of BGR frames sweeping across red_positions."""
    frames: list[Any] = []
    for idx, rpos in enumerate(red_positions):
        tilt = tilt_fn(idx, rpos) if tilt_fn is not None else base_spec.tilt_deg
        cx, cy = center_fn(idx, rpos) if center_fn is not None else (base_spec.cx, base_spec.cy)
        dist = distractors_fn(idx, rpos) if distractors_fn is not None else base_spec.distractors
        spec = GaugeSpec(
            width=base_spec.width,
            height=base_spec.height,
            cx=cx,
            cy=cy,
            tilt_deg=tilt,
            scale=base_spec.scale,
            bar_width=base_spec.bar_width,
            track_len=base_spec.track_len,
            colored_len=base_spec.colored_len,
            silver_pad=base_spec.silver_pad,
            gold_offset=base_spec.gold_offset,
            gold_len=base_spec.gold_len,
            red_pos=rpos,
            red_width=base_spec.red_width,
            red_thickness=base_spec.red_thickness,
            show_gauge=base_spec.show_gauge,
            distractors=dist,
            palette=base_spec.palette,
            seed=base_spec.seed,
        )
        frames.append(render_gauge_frame(spec))
    return frames


def generate_multi_strike_sequence(
    strikes: Sequence[StrikeSpec],
    *,
    width: int = 1920,
    height: int = 1080,
    palette: GaugePalette = VIDEO_PALETTE,
    cx_fn: Callable[[int, int, StrikeSpec], float] | None = None,
    cy_fn: Callable[[int, int, StrikeSpec], float] | None = None,
    tilt_fn: Callable[[int, int, StrikeSpec], float] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Simulate a continuous multi-strike video sequence with ground truth per frame."""
    frames: list[Any] = []
    truth: list[dict[str, Any]] = []

    current_pos: float | None = None
    direction = 1.0
    global_idx = 0

    for strike_idx, strike in enumerate(strikes):
        lo, hi = strike.bounce_bounds
        if strike.red_start is not None:
            current_pos = float(strike.red_start)
            direction = 1.0 if strike.red_velocity_px_per_frame >= 0.0 else -1.0
        elif current_pos is None:
            current_pos = float(lo)
            direction = 1.0 if strike.red_velocity_px_per_frame >= 0.0 else -1.0
        else:
            # Clamp carried-over position inside current strike's bounce bounds
            current_pos = max(lo, min(hi, current_pos))
            if strike.red_velocity_px_per_frame < 0.0:
                direction = -1.0

        speed = abs(float(strike.red_velocity_px_per_frame))

        for local_idx in range(strike.frames):
            cx = cx_fn(global_idx, local_idx, strike) if cx_fn is not None else strike.cx
            cy = cy_fn(global_idx, local_idx, strike) if cy_fn is not None else strike.cy
            tilt = tilt_fn(global_idx, local_idx, strike) if tilt_fn is not None else strike.tilt_deg

            rpos = current_pos if strike.show_gauge else None
            spec = GaugeSpec(
                width=width,
                height=height,
                cx=cx,
                cy=cy,
                tilt_deg=tilt,
                scale=strike.scale,
                bar_width=strike.bar_width,
                gold_offset=strike.gold_offset,
                gold_len=strike.gold_len,
                red_pos=rpos,
                red_width=strike.red_width,
                red_thickness=strike.red_thickness,
                show_gauge=strike.show_gauge,
                distractors=strike.distractors,
                palette=palette,
            )
            frames.append(render_gauge_frame(spec))

            norm = expected_norm_pos(spec)
            in_gold = bool(norm is not None and abs(norm) <= 1.0)
            if strike.show_gauge and rpos is not None:
                g_top = strike.gold_offset - strike.gold_len * 0.5
                g_bot = strike.gold_offset + strike.gold_len * 0.5
                r_top = rpos - strike.red_thickness * 0.5
                r_bot = rpos + strike.red_thickness * 0.5
                any_overlap = (r_bot >= g_top) and (r_top <= g_bot)
            else:
                any_overlap = False

            truth.append(
                {
                    "frame_idx": global_idx,
                    "strike_idx": strike_idx,
                    "local_idx": local_idx,
                    "show_gauge": strike.show_gauge,
                    "cx": cx,
                    "cy": cy,
                    "tilt_deg": tilt,
                    "gold_offset": strike.gold_offset,
                    "gold_len": strike.gold_len,
                    "red_pos": rpos,
                    "expected_norm_pos": norm,
                    "in_gold": in_gold,
                    "any_overlap": any_overlap,
                }
            )

            # Advance triangle-wave position for next frame
            if strike.show_gauge and speed > 0.0:
                next_pos = current_pos + direction * speed
                while next_pos > hi or next_pos < lo:
                    if next_pos > hi:
                        next_pos = hi - (next_pos - hi)
                        direction = -1.0
                    elif next_pos < lo:
                        next_pos = lo + (lo - next_pos)
                        direction = 1.0
                current_pos = next_pos

            global_idx += 1

    return frames, truth
