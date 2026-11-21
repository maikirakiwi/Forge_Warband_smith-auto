from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from vrc_smith_auto.config import AppConfig, GeometryConfig, HSVConfig
from vrc_smith_auto.protocols import FramePacket
from vrc_smith_auto.vision.models import DetectionResult, OrientedBox

_MAX_TILT_DEG = 65.0


def _canon_u(ux: float, uy: float) -> tuple[float, float]:
    """Canonicalize unit axis vector top-to-bottom (uy >= 0; if horizontal, ux >= 0)."""
    norm = math.hypot(ux, uy)
    if norm < 1e-9:
        return (0.0, 1.0)
    ux /= norm
    uy /= norm
    if uy < 0.0 or (abs(uy) < 1e-6 and ux < 0.0):
        ux = -ux
        uy = -uy
    return (ux, uy)


def _rect_axes(rect: cv2.typing.RotatedRect) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Return (major_u, minor_u, max_dim, min_dim) from cv2.minAreaRect (pixel-inclusive)."""
    box = cv2.boxPoints(rect)
    e0 = box[1] - box[0]
    e1 = box[2] - box[1]
    l0 = float(math.hypot(float(e0[0]), float(e0[1])))
    l1 = float(math.hypot(float(e1[0]), float(e1[1])))
    d0 = float(rect[1][0]) + 1.0
    d1 = float(rect[1][1]) + 1.0
    if l0 >= l1:
        major_u = _canon_u(float(e0[0]), float(e0[1]))
        minor_u = _canon_u(float(e1[0]), float(e1[1]))
    else:
        major_u = _canon_u(float(e1[0]), float(e1[1]))
        minor_u = _canon_u(float(e0[0]), float(e0[1]))
    return major_u, minor_u, max(d0, d1), min(d0, d1)


@dataclass(slots=True)
class _CandidateEval:
    u: tuple[float, float]
    v: tuple[float, float]
    tilt_deg: float
    g_umin: float
    g_umax: float
    g_vmin: float
    g_vmax: float
    gold_l: float
    gauge_w: float
    above_ratio: float
    below_ratio: float
    fill_ratio: float
    score: float


class GaugeDetector:
    """Coarse-to-fine, tilt- and shift-invariant timing gauge detector."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        hsv: HSVConfig | None = None,
        geometry: GeometryConfig | None = None,
    ) -> None:
        base = config if config is not None else AppConfig()
        self._hsv: HSVConfig = hsv if hsv is not None else base.hsv
        self._geometry: GeometryConfig = geometry if geometry is not None else base.geometry
        self._config: AppConfig = (
            base
            if (hsv is None and geometry is None)
            else AppConfig(
                capture=base.capture,
                window=base.window,
                trigger=base.trigger,
                hsv=self._hsv,
                geometry=self._geometry,
            )
        )

        g_lo, g_hi = self._hsv.gold.to_bounds()
        s_lo, s_hi = self._hsv.silver.to_bounds()
        r1_lo, r1_hi = self._hsv.red_low.to_bounds()
        r2_lo, r2_hi = self._hsv.red_high.to_bounds()
        self._gold_lo = np.array(g_lo, dtype=np.uint8)
        self._gold_hi = np.array(g_hi, dtype=np.uint8)
        self._silver_lo = np.array(s_lo, dtype=np.uint8)
        self._silver_hi = np.array(s_hi, dtype=np.uint8)
        self._red1_lo = np.array(r1_lo, dtype=np.uint8)
        self._red1_hi = np.array(r1_hi, dtype=np.uint8)
        self._red2_lo = np.array(r2_lo, dtype=np.uint8)
        self._red2_hi = np.array(r2_hi, dtype=np.uint8)

        self._kernel_cache: dict[int, np.ndarray] = {}

        # Tracking and 2D edge-anchor state
        self._locked_center: tuple[float, float] | None = None
        self._last_roi: tuple[int, int, int, int] | None = None
        self._last_gauge_w: float | None = None
        self._lost_frames: int = 0
        self._ref_top_2d: np.ndarray | None = None
        self._ref_bot_2d: np.ndarray | None = None
        self._ref_center_2d: np.ndarray | None = None
        self._ref_gold_len: float | None = None

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def is_locked(self) -> bool:
        return self._locked_center is not None

    @property
    def last_roi(self) -> tuple[int, int, int, int] | None:
        return self._last_roi

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    def reset(self) -> None:
        self._locked_center = None
        self._last_roi = None
        self._last_gauge_w = None
        self._lost_frames = 0
        self._clear_edge_ref()

    def _clear_2d_pos_ref(self) -> None:
        self._ref_top_2d = None
        self._ref_bot_2d = None
        self._ref_center_2d = None

    def _clear_edge_ref(self) -> None:
        self._clear_2d_pos_ref()
        self._ref_gold_len = None

    def _get_ellipse_kernel(self, k_sz: int) -> np.ndarray:
        kern = self._kernel_cache.get(k_sz)
        if kern is None:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
            self._kernel_cache[k_sz] = kern
        return kern

    def detect_packet(self, packet: FramePacket) -> DetectionResult:
        return self.detect(packet.image_bgr, timestamp_s=packet.timestamp_s)

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float | None = None,
        *,
        update_state: bool = True,
    ) -> DetectionResult:
        del timestamp_s  # Reserved for optional temporal diagnostics
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.size == 0
            or frame_bgr.dtype != np.uint8
        ):
            return DetectionResult.empty()

        fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        frame_scale = 1.0 if fh < 480 else fh / float(self._geometry.reference_height)
        base_pad = int(round(self._geometry.roi_padding_px * frame_scale))
        pad = (
            max(base_pad, int(round(5.5 * self._last_gauge_w)))
            if self._last_gauge_w is not None
            else base_pad
        )

        # 1. Try locked full-res ROI first
        if self._locked_center is not None:
            cx_lock, cy_lock = self._locked_center
            x0 = max(0, int(round(cx_lock)) - pad)
            y0 = max(0, int(round(cy_lock)) - pad)
            x1 = min(fw, int(round(cx_lock)) + pad)
            y1 = min(fh, int(round(cy_lock)) + pad)
            if x1 > x0 and y1 > y0:
                res = self._detect_in_region(
                    frame_bgr,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    ds=1.0,
                    frame_scale=frame_scale,
                    strict=True,
                    update_ref=update_state,
                )
                if res is not None:
                    if update_state and res.gold_box is not None:
                        self._locked_center = res.gold_box.center
                        self._last_roi = res.roi_used
                        self._last_gauge_w = res.gold_box.size[0]
                        self._lost_frames = 0
                    return res

            if update_state:
                self._lost_frames += 1
                if self._lost_frames >= self._geometry.max_lost_frames:
                    self._locked_center = None
                    self._last_roi = None
                    self._last_gauge_w = None
                    self._clear_edge_ref()

        # 2. Full-frame acquisition (downscaled when frame is larger than 960x540 and acquire_downscale < 1.0)
        ds = 1.0 if (fh <= 540 and fw <= 960) else self._geometry.acquire_downscale
        if ds < 0.999:
            res_final = self._detect_in_region(
                frame_bgr,
                x0=0,
                y0=0,
                x1=fw,
                y1=fh,
                ds=ds,
                frame_scale=frame_scale,
                strict=False,
                update_ref=False,
                refine_fn=lambda cands: self._refine_coarse_candidates(
                    frame_bgr,
                    coarse_cands=cands,
                    fw=fw,
                    fh=fh,
                    pad=pad,
                    frame_scale=frame_scale,
                    update_state=update_state,
                ),
            )
            if res_final is None:
                return DetectionResult.empty(roi_used=(0, 0, fw, fh))
        else:
            res_final = self._detect_in_region(
                frame_bgr,
                x0=0,
                y0=0,
                x1=fw,
                y1=fh,
                ds=1.0,
                frame_scale=frame_scale,
                strict=True,
                update_ref=update_state,
            )
            if res_final is None:
                return DetectionResult.empty(roi_used=(0, 0, fw, fh))

        if update_state and res_final.gold_box is not None:
            self._locked_center = res_final.gold_box.center
            self._last_roi = res_final.roi_used
            self._last_gauge_w = res_final.gold_box.size[0]
            self._lost_frames = 0
        return res_final

    def _refine_coarse_candidates(
        self,
        frame_bgr: np.ndarray,
        *,
        coarse_cands: list[DetectionResult],
        fw: int,
        fh: int,
        pad: int,
        frame_scale: float,
        update_state: bool,
    ) -> DetectionResult | None:
        for coarse in coarse_cands:
            if coarse.gold_box is None:
                continue
            cgx, cgy = coarse.gold_box.center
            ref_pad = max(
                pad,
                int(round(120.0 * frame_scale)),
                int(round(5.5 * coarse.gold_box.size[0])),
            )
            rx0 = max(0, int(round(cgx)) - ref_pad)
            ry0 = max(0, int(round(cgy)) - ref_pad)
            rx1 = min(fw, int(round(cgx)) + ref_pad)
            ry1 = min(fh, int(round(cgy)) + ref_pad)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            refined = self._detect_in_region(
                frame_bgr,
                x0=rx0,
                y0=ry0,
                x1=rx1,
                y1=ry1,
                ds=1.0,
                frame_scale=frame_scale,
                strict=True,
                update_ref=update_state,
            )
            if refined is not None:
                return refined
        return None

    def _evaluate_gold_axis(
        self,
        *,
        u_cand: tuple[float, float],
        g_xs: np.ndarray,
        g_ys: np.ndarray,
        s_xs: np.ndarray,
        s_ys: np.ndarray,
        s: float,
        min_w: float,
        max_w: float,
        min_h: float,
        max_h: float,
        fill_ratio: float,
        allow_one_sided_silver: bool = False,
        narrow_min_h: float | None = None,
        out_can_abut: list[bool] | None = None,
    ) -> _CandidateEval | None:
        ux, uy = u_cand
        tilt_deg = math.degrees(math.atan2(ux, uy))
        if abs(tilt_deg) > _MAX_TILT_DEG:
            return None

        vx, vy = uy, -ux
        g_u = g_xs * ux + g_ys * uy
        g_v = g_xs * vx + g_ys * vy
        g_umin = float(g_u.min()) - 0.5
        g_umax = float(g_u.max()) + 0.5
        g_vmin = float(g_v.min()) - 0.5
        g_vmax = float(g_v.max()) + 0.5
        gold_l = g_umax - g_umin
        gauge_w = g_vmax - g_vmin

        if not (min_w <= gauge_w <= max_w and gold_l <= max_h):
            return None
        in_primary_h = gold_l >= min_h
        check_abut = (
            not in_primary_h
            and out_can_abut is not None
            and not out_can_abut
            and narrow_min_h is not None
            and gold_l >= narrow_min_h
        )
        if not in_primary_h and not check_abut:
            return None

        gc_v = 0.5 * (g_vmin + g_vmax)
        s_u = s_xs * ux + s_ys * uy
        s_v = s_xs * vx + s_ys * vy

        lat_mask = np.abs(s_v - gc_v) <= 0.40 * gauge_w
        band_l = max(22.0 * s, 0.65 * gold_l)
        band_area = max(1.0, band_l * (0.80 * gauge_w))
        above_cnt = int(np.count_nonzero(lat_mask & (s_u >= g_umin - band_l) & (s_u <= g_umin + 1.0)))
        below_cnt = int(np.count_nonzero(lat_mask & (s_u >= g_umax - 1.0) & (s_u <= g_umax + band_l)))
        above_ratio = above_cnt / band_area
        below_ratio = below_cnt / band_area

        max_side = max(above_ratio, below_ratio)
        if (
            out_can_abut is not None
            and not out_can_abut
            and narrow_min_h is not None
            and gold_l >= narrow_min_h
            and max_side >= self._geometry.min_silver_neighbor_ratio
        ):
            out_can_abut.append(True)

        if not in_primary_h:
            return None

        silver_support = max_side if allow_one_sided_silver else min(above_ratio, below_ratio)
        if silver_support < self._geometry.min_silver_neighbor_ratio:
            return None

        score = silver_support + 0.25 * min(1.0, fill_ratio)
        return _CandidateEval(
            u=(ux, uy),
            v=(vx, vy),
            tilt_deg=tilt_deg,
            g_umin=g_umin,
            g_umax=g_umax,
            g_vmin=g_vmin,
            g_vmax=g_vmax,
            gold_l=gold_l,
            gauge_w=gauge_w,
            above_ratio=above_ratio,
            below_ratio=below_ratio,
            fill_ratio=fill_ratio,
            score=score,
        )

    def _verify_red_abutment_for_candidate(
        self,
        *,
        cand: _CandidateEval,
        hsv_img: np.ndarray,
        red_mask_full: np.ndarray | None,
        s_xs: np.ndarray,
        s_ys: np.ndarray,
        ww: int,
        wh: int,
        min_scale: float,
        frame_scale: float,
        ref_gold_len: float | None,
    ) -> _CandidateEval | None:
        if max(cand.above_ratio, cand.below_ratio) < self._geometry.min_silver_neighbor_ratio:
            return None

        ux, uy = cand.u
        vx, vy = cand.v
        gc_u_w = 0.5 * (cand.g_umin + cand.g_umax)
        gc_v_w = 0.5 * (cand.g_vmin + cand.g_vmax)
        gcx_w = gc_u_w * ux + gc_v_w * vx
        gcy_w = gc_u_w * uy + gc_v_w * vy

        hu_strip = max(120.0 * min_scale, 3.5 * cand.gauge_w)
        hv_strip = 1.3 * cand.gauge_w
        corners_x = (
            gcx_w - hu_strip * ux - hv_strip * vx,
            gcx_w - hu_strip * ux + hv_strip * vx,
            gcx_w + hu_strip * ux + hv_strip * vx,
            gcx_w + hu_strip * ux - hv_strip * vx,
        )
        corners_y = (
            gcy_w - hu_strip * uy - hv_strip * vy,
            gcy_w - hu_strip * uy + hv_strip * vy,
            gcy_w + hu_strip * uy + hv_strip * vy,
            gcy_w + hu_strip * uy - hv_strip * vy,
        )
        sx0 = max(0, int(math.floor(min(corners_x))))
        sy0 = max(0, int(math.floor(min(corners_y))))
        sx1 = min(ww, int(math.ceil(max(corners_x))) + 1)
        sy1 = min(wh, int(math.ceil(max(corners_y))) + 1)
        if sx1 <= sx0 or sy1 <= sy0:
            return None

        if red_mask_full is not None:
            red_mask = red_mask_full[sy0:sy1, sx0:sx1]
        else:
            strip_hsv = hsv_img[sy0:sy1, sx0:sx1]
            red_mask = cv2.bitwise_or(
                cv2.inRange(strip_hsv, self._red1_lo, self._red1_hi),
                cv2.inRange(strip_hsv, self._red2_lo, self._red2_hi),
            )
        if cv2.countNonZero(red_mask) < 3:
            return None

        n_red, lbl_red, st_red, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        s_u = s_xs * ux + s_ys * uy
        s_v = s_xs * vx + s_ys * vy
        lat_mask = np.abs(s_v - gc_v_w) <= 0.40 * cand.gauge_w
        band_l = max(22.0 * min_scale, 0.65 * cand.gold_l)
        band_area = max(1.0, band_l * (0.80 * cand.gauge_w))
        tol_touch = 2.0 * frame_scale
        eff_gl = max(cand.gold_l, ref_gold_len) if ref_gold_len is not None else cand.gold_l

        for rid in range(1, n_red):
            if int(st_red[rid, cv2.CC_STAT_AREA]) < 3:
                continue
            rx_b = int(st_red[rid, cv2.CC_STAT_LEFT])
            ry_b = int(st_red[rid, cv2.CC_STAT_TOP])
            rw_b = int(st_red[rid, cv2.CC_STAT_WIDTH])
            rh_b = int(st_red[rid, cv2.CC_STAT_HEIGHT])
            sub_r = lbl_red[ry_b : ry_b + rh_b, rx_b : rx_b + rw_b] == rid
            ry_loc, rx_loc = np.nonzero(sub_r)
            r_xs = (rx_loc + sx0 + rx_b).astype(np.float64)
            r_ys = (ry_loc + sy0 + ry_b).astype(np.float64)

            r_u = r_xs * ux + r_ys * uy
            r_v = r_xs * vx + r_ys * vy
            ru_min = float(r_u.min()) - 0.5
            ru_max = float(r_u.max()) + 0.5
            rv_min = float(r_v.min()) - 0.5
            rv_max = float(r_v.max()) + 0.5
            r_width = rv_max - rv_min
            r_vc = 0.5 * (rv_min + rv_max)

            if r_width < self._geometry.red_min_width_ratio * cand.gauge_w or r_width > 2.35 * cand.gauge_w:
                continue
            if abs(r_vc - gc_v_w) > self._geometry.red_max_axis_offset_ratio * cand.gauge_w:
                continue

            r_pts_f32 = np.column_stack((r_xs, r_ys)).astype(np.float32)
            r_rect = cv2.minAreaRect(r_pts_f32)
            r_thick = min(float(r_rect[1][0]) + 1.0, float(r_rect[1][1]) + 1.0)
            if r_thick > self._geometry.red_max_thickness_ratio * max(eff_gl, cand.gauge_w):
                continue

            r_center_u = float(r_u.mean())
            abuts_top = (
                ru_min <= cand.g_umin + tol_touch
                and ru_max >= cand.g_umin - tol_touch
                and r_center_u < gc_u_w
            )
            abuts_bot = (
                ru_max >= cand.g_umax - tol_touch
                and ru_min <= cand.g_umax + tol_touch
                and r_center_u > gc_u_w
            )

            if abuts_top and cand.below_ratio >= self._geometry.min_silver_neighbor_ratio:
                opp_cnt = int(np.count_nonzero(lat_mask & (s_u >= ru_min - band_l) & (s_u <= ru_min + 1.0)))
                near_cnt = int(
                    np.count_nonzero(lat_mask & (s_u >= ru_min - 4.0 * min_scale) & (s_u <= ru_min + 1.0))
                )
                opp_ratio = opp_cnt / band_area
                if opp_ratio >= self._geometry.min_silver_neighbor_ratio and near_cnt >= 2:
                    new_above = max(cand.above_ratio, opp_ratio)
                    return _CandidateEval(
                        u=cand.u,
                        v=cand.v,
                        tilt_deg=cand.tilt_deg,
                        g_umin=cand.g_umin,
                        g_umax=cand.g_umax,
                        g_vmin=cand.g_vmin,
                        g_vmax=cand.g_vmax,
                        gold_l=cand.gold_l,
                        gauge_w=cand.gauge_w,
                        above_ratio=new_above,
                        below_ratio=cand.below_ratio,
                        fill_ratio=cand.fill_ratio,
                        score=min(new_above, cand.below_ratio) + 0.25 * min(1.0, cand.fill_ratio),
                    )

            if abuts_bot and cand.above_ratio >= self._geometry.min_silver_neighbor_ratio:
                opp_cnt = int(np.count_nonzero(lat_mask & (s_u >= ru_max - 1.0) & (s_u <= ru_max + band_l)))
                near_cnt = int(
                    np.count_nonzero(lat_mask & (s_u >= ru_max - 1.0) & (s_u <= ru_max + 4.0 * min_scale))
                )
                opp_ratio = opp_cnt / band_area
                if opp_ratio >= self._geometry.min_silver_neighbor_ratio and near_cnt >= 2:
                    new_below = max(cand.below_ratio, opp_ratio)
                    return _CandidateEval(
                        u=cand.u,
                        v=cand.v,
                        tilt_deg=cand.tilt_deg,
                        g_umin=cand.g_umin,
                        g_umax=cand.g_umax,
                        g_vmin=cand.g_vmin,
                        g_vmax=cand.g_vmax,
                        gold_l=cand.gold_l,
                        gauge_w=cand.gauge_w,
                        above_ratio=cand.above_ratio,
                        below_ratio=new_below,
                        fill_ratio=cand.fill_ratio,
                        score=min(cand.above_ratio, new_below) + 0.25 * min(1.0, cand.fill_ratio),
                    )

        return None

    def _evaluate_gold_candidate(
        self,
        *,
        g_xs: np.ndarray,
        g_ys: np.ndarray,
        filled_area: int,
        actual_gold_px: int,
        hsv_img: np.ndarray,
        silver_mask: np.ndarray,
        labels: np.ndarray,
        kernel: np.ndarray,
        ww: int,
        wh: int,
        min_w: float,
        max_w: float,
        min_h: float,
        max_h: float,
        min_side_bound: float,
        max_side_bound: float,
        min_fill: float,
        min_silver_px: int,
        min_scale: float,
        max_scale: float,
        k_scale: float,
        cids: tuple[int, ...],
        sub_closed: np.ndarray | None = None,
        sub_box: tuple[int, int, int, int] | None = None,
        bridge_crop: np.ndarray | None = None,
        bridge_box: tuple[int, int, int, int] | None = None,
        allow_red_abutting: bool = False,
        frame_scale: float = 1.0,
        ref_gold_len: float | None = None,
        narrow_min_h: float | None = None,
        out_can_abut: list[bool] | None = None,
    ) -> _CandidateEval | None:
        if actual_gold_px < max(3, int(0.15 * filled_area)):
            return None

        pts_f32 = np.column_stack((g_xs, g_ys)).astype(np.float32)
        rect = cv2.minAreaRect(pts_f32)
        g_major_u, g_minor_u, g_max_d, g_min_d = _rect_axes(rect)
        if g_min_d < min_side_bound - 0.5 or g_max_d > max_side_bound + 1.0:
            return None

        fill_ratio = min(1.0, float(filled_area) / max(1.0, g_max_d * g_min_d))
        if fill_ratio < min_fill:
            return None

        gcx = 0.5 * (float(g_xs.min()) + float(g_xs.max()))
        gcy = 0.5 * (float(g_ys.min()) + float(g_ys.max()))
        base_patch_r = 85.0 if k_scale >= 0.999 else 44.0
        patch_r = int(round(max(base_patch_r * k_scale, 0.5 * g_max_d + max(28.0 * k_scale, 0.70 * g_max_d))))
        px0 = max(0, int(round(gcx)) - patch_r)
        py0 = max(0, int(round(gcy)) - patch_r)
        px1 = min(ww, int(round(gcx)) + patch_r)
        py1 = min(wh, int(round(gcy)) + patch_r)

        loc_s = silver_mask[py0:py1, px0:px1]
        if cv2.countNonZero(loc_s) < max(6, min_silver_px):
            return None

        if len(cids) == 1 and sub_closed is not None and sub_box is not None:
            bx, by, bw, bh = sub_box
            ix0 = max(px0, bx)
            iy0 = max(py0, by)
            ix1 = min(px1, bx + bw)
            iy1 = min(py1, by + bh)
            loc_g_this = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
            if ix1 > ix0 and iy1 > iy0:
                loc_g_this[iy0 - py0 : iy1 - py0, ix0 - px0 : ix1 - px0] = (
                    sub_closed[iy0 - by : iy1 - by, ix0 - bx : ix1 - bx].astype(np.uint8) * 255
                )
        else:
            sub_lbl_patch = labels[py0:py1, px0:px1]
            if len(cids) == 1:
                loc_g_mask = sub_lbl_patch == cids[0]
            elif len(cids) == 2:
                loc_g_mask = (sub_lbl_patch == cids[0]) | (sub_lbl_patch == cids[1])
            else:
                loc_g_mask = np.isin(sub_lbl_patch, cids)
            loc_g_this = (loc_g_mask.astype(np.uint8)) * 255
        if bridge_crop is not None and bridge_box is not None:
            bx0, by0, bx1, by1 = bridge_box
            ix0 = max(px0, bx0)
            iy0 = max(py0, by0)
            ix1 = min(px1, bx1)
            iy1 = min(py1, by1)
            if ix1 > ix0 and iy1 > iy0:
                loc_g_this[iy0 - py0 : iy1 - py0, ix0 - px0 : ix1 - px0] = cv2.bitwise_or(
                    loc_g_this[iy0 - py0 : iy1 - py0, ix0 - px0 : ix1 - px0],
                    bridge_crop[iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0],
                )

        loc_sg = cv2.morphologyEx(cv2.bitwise_or(loc_g_this, loc_s), cv2.MORPH_CLOSE, kernel)

        contours_sg, _ = cv2.findContours(loc_sg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_sg:
            return None

        gcx_loc = gcx - px0
        gcy_loc = gcy - py0
        best_cnt = None
        best_cnt_area = -1.0
        for cnt in contours_sg:
            c_area = cv2.contourArea(cnt)
            if c_area < 6.0:
                continue
            if cv2.pointPolygonTest(cnt, (gcx_loc, gcy_loc), False) >= 0:
                best_cnt = cnt
                break
            if c_area > best_cnt_area:
                best_cnt_area = c_area
                best_cnt = cnt
        if best_cnt is None:
            return None

        sg_rect = cv2.minAreaRect(best_cnt)
        sg_major_u, sg_minor_u, sg_max_d, sg_min_d = _rect_axes(sg_rect)
        asp_sg = sg_max_d / max(1.0, sg_min_d)
        asp_g = g_max_d / max(1.0, g_min_d)

        s_ys_loc, s_xs_loc = np.nonzero(loc_s)
        s_xs = (s_xs_loc + px0).astype(np.float64)
        s_ys = (s_ys_loc + py0).astype(np.float64)

        cand_axes: list[tuple[float, float]] = []
        if asp_sg >= 1.55:
            cand_axes.append(sg_major_u)
        else:
            if asp_g >= 1.25:
                cand_axes.append(g_minor_u)
                cand_axes.append(g_major_u)
            cand_axes.append(sg_major_u)
            cand_axes.append(sg_minor_u)

        local_best: _CandidateEval | None = None
        for u_cand in cand_axes:
            ev = self._evaluate_gold_axis(
                u_cand=u_cand,
                g_xs=g_xs,
                g_ys=g_ys,
                s_xs=s_xs,
                s_ys=s_ys,
                s=min_scale,
                min_w=min_w,
                max_w=max_w,
                min_h=min_h,
                max_h=max_h,
                fill_ratio=fill_ratio,
                allow_one_sided_silver=allow_red_abutting,
                narrow_min_h=narrow_min_h,
                out_can_abut=out_can_abut if not allow_red_abutting else None,
            )
            if ev is not None and (local_best is None or ev.score > local_best.score):
                local_best = ev

        if local_best is None:
            return None

        # Refine axis using Silver top/bottom centroids when S+G is near-square
        if asp_sg < 1.55:
            ux, uy = local_best.u
            vx, vy = local_best.v
            gc_u_tmp = 0.5 * (local_best.g_umin + local_best.g_umax)
            gc_v_tmp = 0.5 * (local_best.g_vmin + local_best.g_vmax)
            s_u = s_xs * ux + s_ys * uy
            s_v = s_xs * vx + s_ys * vy
            near_lat = np.abs(s_v - gc_v_tmp) <= 0.40 * local_best.gauge_w
            near_ax = np.abs(s_u - gc_u_tmp) <= 50.0 * max_scale
            top_mask = near_lat & near_ax & (s_u < local_best.g_umin + 1.0)
            bot_mask = near_lat & near_ax & (s_u > local_best.g_umax - 1.0)
            if int(np.count_nonzero(top_mask)) >= 4 and int(np.count_nonzero(bot_mask)) >= 4:
                dx_s = float(s_xs[bot_mask].mean() - s_xs[top_mask].mean())
                dy_s = float(s_ys[bot_mask].mean() - s_ys[top_mask].mean())
                u_ref = _canon_u(dx_s, dy_s)
                ev_ref = self._evaluate_gold_axis(
                    u_cand=u_ref,
                    g_xs=g_xs,
                    g_ys=g_ys,
                    s_xs=s_xs,
                    s_ys=s_ys,
                    s=min_scale,
                    min_w=min_w,
                    max_w=max_w,
                    min_h=min_h,
                    max_h=max_h,
                    fill_ratio=fill_ratio,
                    allow_one_sided_silver=allow_red_abutting,
                )
                if ev_ref is not None:
                    local_best = ev_ref

        if allow_red_abutting:
            local_best = self._verify_red_abutment_for_candidate(
                cand=local_best,
                hsv_img=hsv_img,
                red_mask_full=None,
                s_xs=s_xs,
                s_ys=s_ys,
                ww=ww,
                wh=wh,
                min_scale=min_scale,
                frame_scale=frame_scale,
                ref_gold_len=ref_gold_len,
            )

        return local_best

    def _detect_in_region(
        self,
        frame_bgr: np.ndarray,
        *,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        ds: float,
        frame_scale: float,
        strict: bool,
        update_ref: bool,
        refine_fn: Callable[[list[DetectionResult]], DetectionResult | None] | None = None,
    ) -> DetectionResult | None:
        roi_bgr = frame_bgr[y0:y1, x0:x1]
        min_scale = min(1.0, frame_scale) * ds
        max_scale = max(1.0, frame_scale) * ds
        k_scale = max(frame_scale, 1.0) * ds
        s = min_scale

        if ds < 0.999 and abs(ds - 0.5) < 1e-6 and roi_bgr.shape[0] >= 8 and roi_bgr.shape[1] >= 8:
            rh = roi_bgr.shape[0] // 2
            rw = roi_bgr.shape[1] // 2
            wh, ww = int(rh), int(rw)
            if wh < 4 or ww < 4:
                return None
            hsv_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            gm_full = cv2.inRange(hsv_full, self._gold_lo, self._gold_hi)
            gm_sub = gm_full[: 2 * rh, : 2 * rw]
            # Exact 2x2 OR pool via SIMD INTER_AREA (>0 iff >=1 pixel in 2x2 block is 255)
            _, gold_mask = cv2.threshold(
                cv2.resize(gm_sub, (ww, wh), interpolation=cv2.INTER_AREA),
                0,
                255,
                cv2.THRESH_BINARY,
            )
            if cv2.countNonZero(gold_mask) < 4:
                return None
            hsv_img = cv2.resize(hsv_full[: 2 * rh, : 2 * rw], (ww, wh), interpolation=cv2.INTER_NEAREST)
        else:
            if ds < 0.999:
                rw = max(1, int(round(roi_bgr.shape[1] * ds)))
                rh = max(1, int(round(roi_bgr.shape[0] * ds)))
                work_bgr = cv2.resize(roi_bgr, (rw, rh), interpolation=cv2.INTER_NEAREST)
            else:
                work_bgr = roi_bgr

            wh, ww = int(work_bgr.shape[0]), int(work_bgr.shape[1])
            if wh < 4 or ww < 4:
                return None

            hsv_img = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
            gold_mask = cv2.inRange(hsv_img, self._gold_lo, self._gold_hi)
            if cv2.countNonZero(gold_mask) < 4:
                return None

        k_sz = max(3, int(round(self._geometry.morph_bridge_kernel_h * k_scale)))
        if k_sz % 2 == 0:
            k_sz += 1
        kernel = self._get_ellipse_kernel(k_sz)
        gold_closed = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gold_closed, connectivity=8)
        if num_labels <= 1:
            return None

        min_w = self._geometry.min_gold_width_px * min_scale
        max_w = self._geometry.max_gold_width_px * max_scale
        min_h = self._geometry.min_gold_height_px * min_scale
        max_h = self._geometry.max_gold_height_px * max_scale
        if not strict:
            min_w = max(2.0, min_w - 2.0)
            min_h = max(1.5, min_h - 2.5)
            max_w = max_w + 3.0
            max_h = max_h + 3.0
            min_fill = self._geometry.min_gold_fill_ratio * 0.75
        else:
            min_fill = self._geometry.min_gold_fill_ratio

        min_side_bound = min(min_w, min_h)
        max_side_bound = max(max_w, max_h)
        min_cc_area = max(
            6,
            int(0.35 * (self._geometry.min_gold_width_px * min_scale) * (self._geometry.min_gold_height_px * min_scale)),
        )
        min_cc_span = max(3.0, (self._geometry.min_gold_width_px * min_scale) * 0.65)
        max_cc_span = max_side_bound * 1.45 + 4.0
        min_silver_px = (
            max(6, int(1.2 * self._geometry.min_silver_neighbor_ratio * 22.0 * min_scale * 0.80 * min_w))
            if self._geometry.min_silver_neighbor_ratio > 0.0
            else 0
        )

        areas = stats[1:, cv2.CC_STAT_AREA]
        bws = stats[1:, cv2.CC_STAT_WIDTH]
        bhs = stats[1:, cv2.CC_STAT_HEIGHT]
        max_b = np.maximum(bws, bhs)

        silver_mask = cv2.inRange(hsv_img, self._silver_lo, self._silver_hi)
        req_silver = max(6, min_silver_px)
        if cv2.countNonZero(silver_mask) < req_silver:
            return None

        silver_near_cache: dict[int, bool] = {}

        def _has_silver_near(cid: int, bx: int, by: int, bw: int, bh: int) -> bool:
            cached = silver_near_cache.get(cid)
            if cached is not None:
                return cached
            ub_d = math.hypot(bw, bh)
            pr = int(round(max(44.0 * k_scale, 0.5 * ub_d + max(28.0 * k_scale, 0.70 * ub_d))))
            cx_b = int(round(bx + 0.5 * (bw - 1)))
            cy_b = int(round(by + 0.5 * (bh - 1)))
            px0 = max(0, cx_b - pr)
            py0 = max(0, cy_b - pr)
            px1 = min(ww, cx_b + pr + 1)
            py1 = min(wh, cy_b + pr + 1)
            ok_silver = int(cv2.countNonZero(silver_mask[py0:py1, px0:px1])) >= req_silver
            silver_near_cache[cid] = ok_silver
            return ok_silver

        narrow_min_h = max(1.8 * min_scale, 0.9 if ds < 0.999 else 1.5)
        valid_cids = (
            np.nonzero(
                (areas >= min_cc_area)
                & (max_b >= min_cc_span)
                & (max_b <= max_cc_span)
            )[0]
            + 1
        )

        best_eval: _CandidateEval | None = None
        best_g_xs: np.ndarray | None = None
        best_g_ys: np.ndarray | None = None
        coarse_evals: list[_CandidateEval] = []
        skip_narrow_cids: set[int] = set()

        for cid_np in valid_cids:
            cid = int(cid_np)
            bx = int(stats[cid, cv2.CC_STAT_LEFT])
            by = int(stats[cid, cv2.CC_STAT_TOP])
            bw = int(stats[cid, cv2.CC_STAT_WIDTH])
            bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
            area = int(stats[cid, cv2.CC_STAT_AREA])

            if math.hypot(bw, bh) < min_side_bound * 0.85:
                continue
            if not _has_silver_near(cid, bx, by, bw, bh):
                skip_narrow_cids.add(cid)
                continue

            sub_lbl = labels[by : by + bh, bx : bx + bw]
            sub_closed = sub_lbl == cid
            actual_gold_px = int(np.count_nonzero(gold_mask[by : by + bh, bx : bx + bw][sub_closed]))
            if actual_gold_px < max(3, int(0.15 * area)):
                skip_narrow_cids.add(cid)
                continue

            ys_loc, xs_loc = np.nonzero(sub_closed)
            g_xs = (xs_loc + bx).astype(np.float64)
            g_ys = (ys_loc + by).astype(np.float64)

            can_abut_flag: list[bool] = []
            local_best = self._evaluate_gold_candidate(
                g_xs=g_xs,
                g_ys=g_ys,
                filled_area=area,
                actual_gold_px=actual_gold_px,
                hsv_img=hsv_img,
                silver_mask=silver_mask,
                labels=labels,
                kernel=kernel,
                ww=ww,
                wh=wh,
                min_w=min_w,
                max_w=max_w,
                min_h=min_h,
                max_h=max_h,
                min_side_bound=min_side_bound,
                max_side_bound=max_side_bound,
                min_fill=min_fill,
                min_silver_px=min_silver_px,
                min_scale=min_scale,
                max_scale=max_scale,
                k_scale=k_scale,
                cids=(cid,),
                sub_closed=sub_closed,
                sub_box=(bx, by, bw, bh),
                narrow_min_h=narrow_min_h,
                out_can_abut=can_abut_flag,
            )
            if local_best is None and not can_abut_flag and min(bw, bh) >= min_h:
                skip_narrow_cids.add(cid)
            if local_best is not None:
                if refine_fn is not None:
                    coarse_evals.append(local_best)
                if best_eval is None or local_best.score > best_eval.score:
                    best_eval = local_best
                    best_g_xs = g_xs
                    best_g_ys = g_ys

        dedup_r = max(20.0 * frame_scale, 16.0)

        def _build_coarse_cands(
            evals: list[_CandidateEval],
            exclude_centers: list[tuple[float, float]],
        ) -> list[DetectionResult]:
            evals.sort(key=lambda ev: ev.score, reverse=True)
            kept_centers: list[tuple[float, float]] = list(exclude_centers)
            cands: list[DetectionResult] = []
            for ev in evals:
                e_ux, e_uy = ev.u
                e_vx, e_vy = ev.v
                e_gu = 0.5 * (ev.g_umin + ev.g_umax)
                e_gv = 0.5 * (ev.g_vmin + ev.g_vmax)
                gcx_w = e_gu * e_ux + e_gv * e_vx
                gcy_w = e_gu * e_uy + e_gv * e_vy
                gcx_ff = x0 + gcx_w / ds
                gcy_ff = y0 + gcy_w / ds
                if any(math.hypot(gcx_ff - kx, gcy_ff - ky) < dedup_r for kx, ky in kept_centers):
                    continue
                kept_centers.append((gcx_ff, gcy_ff))
                cands.append(
                    DetectionResult(
                        gauge_present=True,
                        gold_box=OrientedBox(
                            center=(float(gcx_ff), float(gcy_ff)),
                            size=(float(ev.gauge_w / ds), float(ev.gold_l / ds)),
                            angle_deg=float(ev.tilt_deg),
                        ),
                        axis_unit=ev.u,
                        tilt_deg=float(ev.tilt_deg),
                        confidence=0.5,
                        roi_used=(x0, y0, x1 - x0, y1 - y0),
                    )
                )
                if len(cands) >= 6:
                    break
            return cands

        tried_centers: list[tuple[float, float]] = []
        run_fallbacks = best_eval is None
        if refine_fn is not None and coarse_evals:
            primary_cands = _build_coarse_cands(coarse_evals, tried_centers)
            refined = refine_fn(primary_cands)
            if refined is not None:
                return refined
            # Primary coarse candidates were coarse-only distractors; run split/narrow fallbacks on existing state
            tried_centers.extend(c.gold_box.center for c in primary_cands if c.gold_box is not None)
            coarse_evals.clear()
            run_fallbacks = True

        # Lazy per-pair Red-guided split-occlusion fallback
        if run_fallbacks and num_labels >= 3:
            pair_cids = [
                int(c)
                for c in (
                    np.nonzero(
                        (areas >= max(3, min_cc_area // 4))
                        & (max_b >= min_cc_span * 0.65)
                        & (max_b <= max_cc_span)
                    )[0]
                    + 1
                )
                if _has_silver_near(
                    int(c),
                    int(stats[int(c), cv2.CC_STAT_LEFT]),
                    int(stats[int(c), cv2.CC_STAT_TOP]),
                    int(stats[int(c), cv2.CC_STAT_WIDTH]),
                    int(stats[int(c), cv2.CC_STAT_HEIGHT]),
                )
            ]
            if len(pair_cids) > 16:
                pair_cids.sort(key=lambda c: int(stats[c, cv2.CC_STAT_AREA]), reverse=True)
                pair_cids = pair_cids[:16]

            max_pair_gap = 45.0 * max_scale
            for idx_a in range(len(pair_cids)):
                ca = pair_cids[idx_a]
                ax0 = int(stats[ca, cv2.CC_STAT_LEFT])
                ay0 = int(stats[ca, cv2.CC_STAT_TOP])
                ax1 = ax0 + int(stats[ca, cv2.CC_STAT_WIDTH])
                ay1 = ay0 + int(stats[ca, cv2.CC_STAT_HEIGHT])
                for idx_b in range(idx_a + 1, len(pair_cids)):
                    cb = pair_cids[idx_b]
                    bx0 = int(stats[cb, cv2.CC_STAT_LEFT])
                    by0 = int(stats[cb, cv2.CC_STAT_TOP])
                    bx1 = bx0 + int(stats[cb, cv2.CC_STAT_WIDTH])
                    by1 = by0 + int(stats[cb, cv2.CC_STAT_HEIGHT])

                    gx = max(0, max(ax0, bx0) - min(ax1, bx1))
                    gy = max(0, max(ay0, by0) - min(ay1, by1))
                    if math.hypot(gx, gy) > max_pair_gap:
                        continue

                    ux0 = min(ax0, bx0)
                    uy0 = min(ay0, by0)
                    ux1 = max(ax1, bx1)
                    uy1 = max(ay1, by1)
                    uw = ux1 - ux0
                    uh = uy1 - uy0
                    if max(uw, uh) > max_cc_span or math.hypot(uw, uh) < min_side_bound * 0.85:
                        continue

                    crop_hsv = hsv_img[uy0:uy1, ux0:ux1]
                    sub_red = cv2.bitwise_or(
                        cv2.inRange(crop_hsv, self._red1_lo, self._red1_hi),
                        cv2.inRange(crop_hsv, self._red2_lo, self._red2_hi),
                    )
                    if cv2.countNonZero(sub_red) < 3:
                        continue

                    sub_lbl = labels[uy0:uy1, ux0:ux1]
                    pair_gold = (sub_lbl == ca) | (sub_lbl == cb)
                    ys_loc, xs_loc = np.nonzero(pair_gold)
                    if ys_loc.size < 4:
                        continue
                    g_xs = (xs_loc + ux0).astype(np.float64)
                    g_ys = (ys_loc + uy0).astype(np.float64)

                    hull_pts = cv2.convexHull(np.column_stack((xs_loc, ys_loc)).astype(np.int32))
                    hull_mask = np.zeros((uh, uw), dtype=np.uint8)
                    cv2.fillConvexPoly(hull_mask, hull_pts, 255)
                    pair_bridge = cv2.bitwise_and(sub_red, hull_mask)
                    if cv2.countNonZero(pair_bridge) < 2:
                        continue

                    filled_area = int(np.count_nonzero(pair_gold | (pair_bridge > 0)))
                    actual_gold_px = int(np.count_nonzero(gold_mask[uy0:uy1, ux0:ux1][pair_gold]))
                    local_best = self._evaluate_gold_candidate(
                        g_xs=g_xs,
                        g_ys=g_ys,
                        filled_area=filled_area,
                        actual_gold_px=actual_gold_px,
                        hsv_img=hsv_img,
                        silver_mask=silver_mask,
                        labels=labels,
                        kernel=kernel,
                        ww=ww,
                        wh=wh,
                        min_w=min_w,
                        max_w=max_w,
                        min_h=min_h,
                        max_h=max_h,
                        min_side_bound=min_side_bound,
                        max_side_bound=max_side_bound,
                        min_fill=min_fill,
                        min_silver_px=min_silver_px,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        k_scale=k_scale,
                        cids=(ca, cb),
                        bridge_crop=pair_bridge,
                        bridge_box=(ux0, uy0, ux1, uy1),
                    )
                    if local_best is not None:
                        if refine_fn is not None:
                            coarse_evals.append(local_best)
                        if best_eval is None or local_best.score > best_eval.score:
                            best_eval = local_best
                            best_g_xs = g_xs
                            best_g_ys = g_ys

        # Narrow-Gold Red-abutting fallback (runs in both full-res and coarse acquisition after normal paths fail)
        if run_fallbacks:
            narrow_margin = max(2, int(round(3.0 * max_scale)))
            narrow_min_side = min(min_w, narrow_min_h)
            narrow_min_area = max(6, int(0.15 * min_w * narrow_min_h))
            narrow_cids = [
                int(c)
                for c in (
                    np.nonzero(
                        (areas >= narrow_min_area)
                        & (max_b >= min_cc_span)
                        & (max_b <= max_cc_span)
                    )[0]
                    + 1
                )
                if int(c) not in skip_narrow_cids
                and _has_silver_near(
                    int(c),
                    int(stats[int(c), cv2.CC_STAT_LEFT]),
                    int(stats[int(c), cv2.CC_STAT_TOP]),
                    int(stats[int(c), cv2.CC_STAT_WIDTH]),
                    int(stats[int(c), cv2.CC_STAT_HEIGHT]),
                )
            ]
            if len(narrow_cids) > 16:
                narrow_cids.sort(key=lambda c: int(stats[c, cv2.CC_STAT_AREA]), reverse=True)
                narrow_cids = narrow_cids[:16]

            for cid in narrow_cids:
                bx = int(stats[cid, cv2.CC_STAT_LEFT])
                by = int(stats[cid, cv2.CC_STAT_TOP])
                bw = int(stats[cid, cv2.CC_STAT_WIDTH])
                bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if math.hypot(bw, bh) < narrow_min_side * 0.85:
                    continue

                rx0 = max(0, bx - narrow_margin)
                ry0 = max(0, by - narrow_margin)
                rx1 = min(ww, bx + bw + narrow_margin)
                ry1 = min(wh, by + bh + narrow_margin)
                loc_r_hsv = hsv_img[ry0:ry1, rx0:rx1]
                loc_r = cv2.bitwise_or(
                    cv2.inRange(loc_r_hsv, self._red1_lo, self._red1_hi),
                    cv2.inRange(loc_r_hsv, self._red2_lo, self._red2_hi),
                )
                if cv2.countNonZero(loc_r) < 3:
                    continue

                sub_lbl = labels[by : by + bh, bx : bx + bw]
                sub_closed = sub_lbl == cid
                actual_gold_px = int(np.count_nonzero(gold_mask[by : by + bh, bx : bx + bw][sub_closed]))
                if actual_gold_px < max(3, int(0.15 * area)):
                    continue

                ys_loc, xs_loc = np.nonzero(sub_closed)
                g_xs = (xs_loc + bx).astype(np.float64)
                g_ys = (ys_loc + by).astype(np.float64)

                local_best = self._evaluate_gold_candidate(
                    g_xs=g_xs,
                    g_ys=g_ys,
                    filled_area=area,
                    actual_gold_px=actual_gold_px,
                    hsv_img=hsv_img,
                    silver_mask=silver_mask,
                    labels=labels,
                    kernel=kernel,
                    ww=ww,
                    wh=wh,
                    min_w=min_w,
                    max_w=max_w,
                    min_h=narrow_min_h,
                    max_h=max_h,
                    min_side_bound=narrow_min_side,
                    max_side_bound=max_side_bound,
                    min_fill=min_fill,
                    min_silver_px=min_silver_px,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    k_scale=k_scale,
                    cids=(cid,),
                    sub_closed=sub_closed,
                    sub_box=(bx, by, bw, bh),
                    allow_red_abutting=True,
                    frame_scale=frame_scale,
                    ref_gold_len=self._ref_gold_len if ds >= 0.999 else None,
                )
                if local_best is not None:
                    if refine_fn is not None:
                        coarse_evals.append(local_best)
                    if best_eval is None or local_best.score > best_eval.score:
                        best_eval = local_best
                        best_g_xs = g_xs
                        best_g_ys = g_ys

        if refine_fn is not None:
            if not coarse_evals:
                return None
            fb_cands = _build_coarse_cands(coarse_evals, tried_centers)
            if not fb_cands:
                return None
            return refine_fn(fb_cands)

        if best_eval is None or best_g_xs is None or best_g_ys is None:
            return None

        ux, uy = best_eval.u
        vx, vy = best_eval.v
        gc_u_w = 0.5 * (best_eval.g_umin + best_eval.g_umax)
        gc_v_w = 0.5 * (best_eval.g_vmin + best_eval.g_vmax)

        # Coarse pass returns candidate center for full-res refinement
        if ds < 0.999:
            gcx_w = gc_u_w * ux + gc_v_w * vx
            gcy_w = gc_u_w * uy + gc_v_w * vy
            gcx_ff = x0 + gcx_w / ds
            gcy_ff = y0 + gcy_w / ds
            return DetectionResult(
                gauge_present=True,
                gold_box=OrientedBox(
                    center=(float(gcx_ff), float(gcy_ff)),
                    size=(float(best_eval.gauge_w / ds), float(best_eval.gold_l / ds)),
                    angle_deg=float(best_eval.tilt_deg),
                ),
                axis_unit=best_eval.u,
                tilt_deg=float(best_eval.tilt_deg),
                confidence=0.5,
                roi_used=(x0, y0, x1 - x0, y1 - y0),
            )

        # Full-resolution (ds == 1.0): compute full-frame projections & strip-scoped Red localization
        g_umin_ff = best_eval.g_umin + x0 * ux + y0 * uy
        g_umax_ff = best_eval.g_umax + x0 * ux + y0 * uy
        gc_v_ff = gc_v_w + x0 * vx + y0 * vy
        gold_l = best_eval.gold_l
        gauge_w = best_eval.gauge_w

        u_vec = np.array([ux, uy], dtype=np.float64)
        v_vec = np.array([vx, vy], dtype=np.float64)
        curr_top_2d = g_umin_ff * u_vec + gc_v_ff * v_vec
        curr_bot_2d = g_umax_ff * u_vec + gc_v_ff * v_vec
        curr_center_2d = 0.5 * (curr_top_2d + curr_bot_2d)

        # Maintain 2D positional anchor separately from intrinsic _ref_gold_len
        ref_pos_valid = False
        if (
            self._ref_center_2d is not None
            and self._ref_gold_len is not None
            and self._ref_top_2d is not None
            and self._ref_bot_2d is not None
        ):
            shift_dist = float(np.hypot(*(curr_center_2d - self._ref_center_2d)))
            if shift_dist <= max(32.0 * frame_scale, 1.8 * self._ref_gold_len):
                ref_pos_valid = True
            elif update_ref:
                self._clear_2d_pos_ref()

        ref_gold_len = self._ref_gold_len
        if (
            ref_gold_len is not None
            and self._last_gauge_w is not None
            and abs(gauge_w - self._last_gauge_w) > 0.35 * max(gauge_w, self._last_gauge_w)
        ):
            ref_gold_len = None
            if update_ref:
                self._ref_gold_len = None

        # Strip-scoped Red localization in work (= ROI) coords
        gcx_w = gc_u_w * ux + gc_v_w * vx
        gcy_w = gc_u_w * uy + gc_v_w * vy
        hu_strip = max(200.0 * s, 4.6 * gauge_w)
        hv_strip = 1.3 * gauge_w
        corners_x = (
            gcx_w - hu_strip * ux - hv_strip * vx,
            gcx_w - hu_strip * ux + hv_strip * vx,
            gcx_w + hu_strip * ux + hv_strip * vx,
            gcx_w + hu_strip * ux - hv_strip * vx,
        )
        corners_y = (
            gcy_w - hu_strip * uy - hv_strip * vy,
            gcy_w - hu_strip * uy + hv_strip * vy,
            gcy_w + hu_strip * uy + hv_strip * vy,
            gcy_w + hu_strip * uy - hv_strip * vy,
        )
        sx0 = max(0, int(math.floor(min(corners_x))))
        sy0 = max(0, int(math.floor(min(corners_y))))
        sx1 = min(ww, int(math.ceil(max(corners_x))) + 1)
        sy1 = min(wh, int(math.ceil(max(corners_y))) + 1)

        red_box: OrientedBox | None = None
        red_proj: float | None = None
        r_umin_ff: float | None = None
        r_umax_ff: float | None = None

        if sx1 > sx0 and sy1 > sy0:
            strip_hsv = hsv_img[sy0:sy1, sx0:sx1]
            red_mask = cv2.bitwise_or(
                cv2.inRange(strip_hsv, self._red1_lo, self._red1_hi),
                cv2.inRange(strip_hsv, self._red2_lo, self._red2_hi),
            )
            if cv2.countNonZero(red_mask) >= 3:
                n_red, lbl_red, st_red, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
                best_red_dist = float("inf")
                best_red_area = -1
                best_red_data: tuple[float, float, float, float, float, float, float] | None = None
                effective_gold_l_for_red = max(gold_l, ref_gold_len) if ref_gold_len is not None else gold_l
                g_center_u_ff = 0.5 * (g_umin_ff + g_umax_ff)
                for rid in range(1, n_red):
                    r_area = int(st_red[rid, cv2.CC_STAT_AREA])
                    if r_area < 3:
                        continue
                    rx_b = int(st_red[rid, cv2.CC_STAT_LEFT])
                    ry_b = int(st_red[rid, cv2.CC_STAT_TOP])
                    rw_b = int(st_red[rid, cv2.CC_STAT_WIDTH])
                    rh_b = int(st_red[rid, cv2.CC_STAT_HEIGHT])
                    sub_r = lbl_red[ry_b : ry_b + rh_b, rx_b : rx_b + rw_b] == rid
                    ry_loc, rx_loc = np.nonzero(sub_r)
                    r_xs = (rx_loc + sx0 + rx_b + x0).astype(np.float64)
                    r_ys = (ry_loc + sy0 + ry_b + y0).astype(np.float64)

                    r_u = r_xs * ux + r_ys * uy
                    r_v = r_xs * vx + r_ys * vy
                    ru_min = float(r_u.min()) - 0.5
                    ru_max = float(r_u.max()) + 0.5
                    rv_min = float(r_v.min()) - 0.5
                    rv_max = float(r_v.max()) + 0.5
                    r_width = rv_max - rv_min
                    r_vc = 0.5 * (rv_min + rv_max)

                    if r_width < self._geometry.red_min_width_ratio * gauge_w or r_width > 2.35 * gauge_w:
                        continue
                    if abs(r_vc - gc_v_ff) > self._geometry.red_max_axis_offset_ratio * gauge_w:
                        continue

                    r_pts_f32 = np.column_stack((r_xs, r_ys)).astype(np.float32)
                    r_rect = cv2.minAreaRect(r_pts_f32)
                    r_thick = min(float(r_rect[1][0]) + 1.0, float(r_rect[1][1]) + 1.0)
                    if r_thick > self._geometry.red_max_thickness_ratio * max(effective_gold_l_for_red, gauge_w):
                        continue

                    r_cx = float(r_xs.mean())
                    r_cy = float(r_ys.mean())
                    r_p = r_cx * ux + r_cy * uy
                    r_dist = abs(r_p - g_center_u_ff)
                    if r_dist < best_red_dist - 1.0 or (abs(r_dist - best_red_dist) <= 1.0 and r_area > best_red_area):
                        best_red_dist = r_dist
                        best_red_area = r_area
                        best_red_data = (r_cx, r_cy, r_width, r_thick, r_p, ru_min, ru_max)

                if best_red_data is not None:
                    r_cx, r_cy, r_width, r_thick, r_p, ru_min, ru_max = best_red_data
                    red_box = OrientedBox(
                        center=(r_cx, r_cy),
                        size=(r_width, r_thick),
                        angle_deg=float(best_eval.tilt_deg),
                    )
                    red_proj = float(r_p)
                    r_umin_ff = ru_min
                    r_umax_ff = ru_max

        # 2D Edge-Anchor maintenance & single-edge occlusion recovery
        tol_touch = 1.0 * frame_scale
        touches_top = (
            r_umin_ff is not None
            and r_umax_ff is not None
            and r_umin_ff <= g_umin_ff + tol_touch
            and r_umax_ff >= g_umin_ff - tol_touch
        )
        touches_bot = (
            r_umin_ff is not None
            and r_umax_ff is not None
            and r_umax_ff >= g_umax_ff - tol_touch
            and r_umin_ff <= g_umax_ff + tol_touch
        )

        if not touches_top and not touches_bot:
            if update_ref and (
                not ref_pos_valid
                or ref_gold_len is None
                or red_box is not None
                or gold_l >= 0.85 * ref_gold_len
            ):
                self._ref_top_2d = curr_top_2d
                self._ref_bot_2d = curr_bot_2d
                self._ref_center_2d = curr_center_2d
                self._ref_gold_len = gold_l
        elif ref_gold_len is not None:
            if gold_l < ref_gold_len - 1.5 * frame_scale:
                if touches_top and not touches_bot and r_umin_ff is not None:
                    if (g_umax_ff - ref_gold_len) < (r_umin_ff - 1.5 * frame_scale):
                        # Stale _ref_gold_len from a wider prior strike (Silver visible beyond Red)
                        ref_gold_len = None
                        if update_ref:
                            self._ref_gold_len = None
                    else:
                        g_umin_ff = min(g_umin_ff, max(g_umax_ff - ref_gold_len, r_umin_ff))
                elif touches_bot and not touches_top and r_umax_ff is not None:
                    if (g_umin_ff + ref_gold_len) > (r_umax_ff + 1.5 * frame_scale):
                        ref_gold_len = None
                        if update_ref:
                            self._ref_gold_len = None
                    else:
                        g_umax_ff = max(g_umax_ff, min(g_umin_ff + ref_gold_len, r_umax_ff))
        elif gold_l < min_h and r_umin_ff is not None and r_umax_ff is not None:
            est_len = max(min_h, gold_l + 0.5 * (r_umax_ff - r_umin_ff))
            if touches_top and not touches_bot:
                g_umin_ff = min(g_umin_ff, max(g_umax_ff - est_len, r_umin_ff))
            elif touches_bot and not touches_top:
                g_umax_ff = max(g_umax_ff, min(g_umin_ff + est_len, r_umax_ff))

        final_gold_l = max(1e-6, g_umax_ff - g_umin_ff)
        g_center_u = 0.5 * (g_umin_ff + g_umax_ff)
        half_len = 0.5 * final_gold_l
        g_cx_ff = g_center_u * ux + gc_v_ff * vx
        g_cy_ff = g_center_u * uy + gc_v_ff * vy

        norm_pos: float | None = None
        if red_proj is not None:
            norm_pos = (red_proj - g_center_u) / half_len

        min_silver = min(best_eval.above_ratio, best_eval.below_ratio)
        silver_score = min(1.0, min_silver / max(0.30, self._geometry.min_silver_neighbor_ratio * 1.5))
        fill_score = min(1.0, best_eval.fill_ratio / 0.85)
        confidence = max(0.1, min(1.0, 0.5 * silver_score + 0.5 * fill_score))

        return DetectionResult(
            gauge_present=True,
            gold_box=OrientedBox(
                center=(float(g_cx_ff), float(g_cy_ff)),
                size=(float(gauge_w), float(final_gold_l)),
                angle_deg=float(best_eval.tilt_deg),
            ),
            red_box=red_box,
            axis_unit=(float(ux), float(uy)),
            gold_span=(float(g_umin_ff), float(g_umax_ff)),
            red_proj=red_proj,
            norm_pos=norm_pos,
            tilt_deg=float(best_eval.tilt_deg),
            confidence=float(confidence),
            roi_used=(x0, y0, x1 - x0, y1 - y0),
        )
