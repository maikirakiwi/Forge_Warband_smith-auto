from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import pytest


# Ground-truth partition for example/2026-11-20 15-32-46.mp4 (1570 frames @ 60.0 fps)
INACTIVE_SEGMENTS: tuple[tuple[int, int], ...] = (
    (0, 117),
    (631, 724),
    (1339, 1569),
)

MANDATORY_FULL_EPISODES: tuple[tuple[str, int, int], ...] = (
    ("E04", 363, 374),
    ("E06", 465, 471),
    ("E08", 574, 579),
    ("E09", 621, 624),
    ("E11", 823, 837),
    ("E14", 1013, 1022),
    ("E15", 1038, 1047),
    ("E16", 1058, 1065),
    ("E19", 1155, 1161),
    ("E20", 1203, 1211),
    ("E21", 1236, 1240),
    ("E22_Seg2_H9a", 1285, 1287),
    ("E23_Seg2_H9b", 1303, 1306),
)

MANDATORY_TRUNC_EPISODES: tuple[tuple[str, int, int], ...] = (
    ("E01", 167, 188),
    ("E02", 232, 248),
    ("E03", 317, 333),
    ("E05", 434, 447),
    ("E07", 537, 543),
    ("E10", 786, 803),
    ("E12", 943, 954),
    ("E13", 993, 1002),
    ("E17", 1106, 1111),
)

COOLDOWN_AMBIG_EPISODES: tuple[tuple[str, int, int], ...] = (
    ("E18_Seg2_H6a", 1112, 1118),
)

ALL_23_EPISODES: tuple[tuple[str, int, int], ...] = (
    MANDATORY_FULL_EPISODES + MANDATORY_TRUNC_EPISODES + COOLDOWN_AMBIG_EPISODES
)


def _assert_video_click_frames_meet_scenario2(
    click_frames: list[int],
    *,
    judge_delay_frames: int = 0,
    cooldown_ms: float = 200.0,
) -> None:
    """Validate clicks at the frame the *game* judges them.

    The click is decided from a captured frame, but the game processes the
    SendInput event on a later frame (detection + dispatch + one game tick).
    ``judge_delay_frames=2`` models the measured worst case; a zero-latency
    assertion would accept a click that is already outside the Gold by the time
    the game sees it.
    """

    def _judged(fidx: int) -> int:
        return fidx + judge_delay_frames

    def _in_win(fidx: int, ep: tuple[str, int, int]) -> bool:
        return ep[1] <= _judged(fidx) <= ep[2]

    # 1. Zero inactive false positives
    inactive_hits = [
        f for f in click_frames if any(lo <= _judged(f) <= hi for lo, hi in INACTIVE_SEGMENTS)
    ]
    assert inactive_hits == [], f"Clicks judged during inactive segments: {inactive_hits}"

    # 2. 100% Gold precision (every judged click falls in one of the 23 Any-Overlap episodes)
    outside_hits = [
        f for f in click_frames if not any(_in_win(f, ep) for ep in ALL_23_EPISODES)
    ]
    assert outside_hits == [], f"Clicks judged outside 23 gold episodes (false positives): {outside_hits}"

    # 3. Zero double-clicks per episode & cooldown spacing
    for name, start, end in ALL_23_EPISODES:
        ep_hits = [f for f in click_frames if _in_win(f, (name, start, end))]
        assert len(ep_hits) <= 1, f"Double-click in episode {name} ({start}..{end}): {ep_hits}"

    min_gap = math.floor(cooldown_ms / (1000.0 / 60.0) + 1e-6)
    if len(click_frames) >= 2:
        gaps = [b - a for a, b in zip(click_frames, click_frames[1:])]
        assert min(gaps) >= min_gap, f"Cooldown violated; frame gaps={gaps}, min_gap={min_gap}"

    # 4. Recall thresholds
    full_hits = sum(
        1 for ep in MANDATORY_FULL_EPISODES if sum(1 for f in click_frames if _in_win(f, ep)) == 1
    )
    trunc_hits = sum(
        1 for ep in MANDATORY_TRUNC_EPISODES if sum(1 for f in click_frames if _in_win(f, ep)) == 1
    )
    main_hits = full_hits + trunc_hits

    assert full_hits >= 12, f"Expected >= 12/13 mandatory full-pass hits, got {full_hits}"
    assert main_hits >= 20, f"Expected >= 20/22 mandatory main-pass hits, got {main_hits}"

    # 5. Explicit late-round Seg2-H9 gate
    h9a = next(ep for ep in MANDATORY_FULL_EPISODES if ep[0] == "E22_Seg2_H9a")
    h9b = next(ep for ep in MANDATORY_FULL_EPISODES if ep[0] == "E23_Seg2_H9b")
    assert sum(1 for f in click_frames if _in_win(f, h9a)) == 1, f"Missed Seg2-H9a {h9a}: {click_frames}"
    assert sum(1 for f in click_frames if _in_win(f, h9b)) == 1, f"Missed Seg2-H9b {h9b}: {click_frames}"


def test_t4_s1_exact_reference_screenshot_silver_zero_clicks(
    e2e: Any,
    reference_screenshot_path: Path,
) -> None:
    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    src = e2e.make_image_source(reference_screenshot_path)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    try:
        pkt = src.read()
        assert pkt is not None
        assert pkt.image_bgr.shape == (223, 325, 3)
        assert str(pkt.image_bgr.dtype) == "uint8"
        assert pkt.frame_idx == 0
        assert src.read() is None

        res = e2e.detect_gauge(det, pkt)
        assert res.gauge_present is True
        assert res.gold_box is not None
        assert res.red_box is not None
        assert res.axis_unit is not None and res.axis_unit[1] > 0.90
        assert res.gold_box.center[0] < 80.0
        assert res.norm_pos is not None
        assert abs(res.norm_pos) > 1.0
        assert -2.5 <= res.norm_pos <= -1.05

        e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
        assert e2e.get_spy_click_count(spy) == 0
    finally:
        src.close()

    proc = e2e.run_cli(
        ["verify-image", "--image", str(reference_screenshot_path), "--expect-clicks", "0", "--json"]
    )
    assert proc.returncode == 0, proc.stderr
    c_cli, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc))
    assert c_cli == 0


def test_t4_s2_full_1570_frame_real_gameplay_video_replay(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    src = e2e.make_video_source(reference_video_path)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    frame_count = 0
    click_frames: list[int] = []
    try:
        while True:
            pkt = src.read()
            if pkt is None:
                break
            frame_count += 1
            spy.tick(pkt.timestamp_s)
            prev = e2e.get_spy_click_count(spy)
            res = e2e.detect_gauge(det, pkt)
            e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
            if e2e.get_spy_click_count(spy) > prev:
                click_frames.append(pkt.frame_idx)
    finally:
        src.close()

    assert frame_count == 1570
    _assert_video_click_frames_meet_scenario2(
        click_frames,
        judge_delay_frames=2,
        cooldown_ms=cfg.cooldown_ms,
    )


def test_t4_s3_synthetic_10_strike_sway_tilt_distractor_video(
    e2e: Any,
    synthetic_gauge_video_factory: Any,
) -> None:
    import synthetic_fixtures as sf

    # 10 strikes: Gold shrinks 38 -> 12 px; speed rises 145 -> 360 px/s (2.42 -> 6.00 px/frame @ 60fps)
    gold_lengths = [38.0, 35.0, 32.0, 29.0, 26.0, 23.0, 20.0, 17.0, 14.0, 12.0]
    speeds_px_per_s = [145.0, 168.0, 192.0, 216.0, 240.0, 264.0, 288.0, 312.0, 336.0, 360.0]
    gold_offsets = [-24.0, 0.0, 22.0, -18.0, 10.0, -12.0, 20.0, 0.0, -16.0, 14.0]
    tilts = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 6.0, -6.0, 12.0]

    strikes: list[sf.StrikeSpec] = [
        sf.StrikeSpec(frames=20, cx=960.0, cy=540.0, show_gauge=False, distractors=True)
    ]
    for idx in range(10):
        v_frame = speeds_px_per_s[idx] / 60.0
        # Roughly one full sweep from -72 to +72 (or +72 to -72) per strike
        n_frames = max(24, int(round(144.0 / v_frame)))
        strikes.append(
            sf.StrikeSpec(
                frames=n_frames,
                cx=900.0,
                cy=520.0,
                tilt_deg=tilts[idx],
                gold_offset=gold_offsets[idx],
                gold_len=gold_lengths[idx],
                red_start=-72.0 if idx == 0 else None,
                red_velocity_px_per_frame=v_frame,
                bounce_bounds=(-72.0, 72.0),
                distractors=True,
                red_thickness=5.0 if gold_lengths[idx] <= 14.0 else 7.0,
            )
        )
    strikes.append(
        sf.StrikeSpec(frames=20, cx=960.0, cy=540.0, show_gauge=False, distractors=True)
    )

    def _cx_sway(gidx: int, _lidx: int, st: sf.StrikeSpec) -> float:
        return st.cx + 80.0 * math.sin(gidx * 0.05)

    def _cy_sway(gidx: int, _lidx: int, st: sf.StrikeSpec) -> float:
        return st.cy + 40.0 * math.cos(gidx * 0.04)

    frames, truth = sf.generate_multi_strike_sequence(
        strikes,
        cx_fn=_cx_sway,
        cy_fn=_cy_sway,
    )
    mp4_path = synthetic_gauge_video_factory(frames, filename="t4_s3_10strike.mp4")

    # Build ground-truth Any-Overlap episodes from generator metadata
    episodes: list[tuple[int, int, int, float]] = []  # (start, end, strike_idx, tilt_deg)
    in_ep = False
    ep_start = 0
    for item in truth:
        if item["any_overlap"] and not in_ep:
            in_ep = True
            ep_start = item["frame_idx"]
        elif not item["any_overlap"] and in_ep:
            in_ep = False
            prev = truth[item["frame_idx"] - 1]
            episodes.append((ep_start, item["frame_idx"] - 1, prev["strike_idx"], prev["tilt_deg"]))
    if in_ep:
        last = truth[-1]
        episodes.append((ep_start, last["frame_idx"], last["strike_idx"], last["tilt_deg"]))

    assert len(episodes) >= 10, f"Expected >= 10 gold crossing episodes, got {len(episodes)}"

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    src = e2e.make_video_source(mp4_path)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    click_frames: list[int] = []
    try:
        while True:
            pkt = src.read()
            if pkt is None:
                break
            spy.tick(pkt.timestamp_s)
            prev = e2e.get_spy_click_count(spy)
            res = e2e.detect_gauge(det, pkt)
            e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
            if e2e.get_spy_click_count(spy) > prev:
                click_frames.append(pkt.frame_idx)
    finally:
        src.close()

    # The game judges the click ~2 frames after the capture frame (detection +
    # dispatch + one game tick).  Score the judged frame, not the decision frame.
    judge_delay_frames = 2
    judged = [f + judge_delay_frames for f in click_frames]

    # 1. Zero clicks in inactive prelude/epilogue
    for fidx in judged:
        assert 0 <= fidx < len(truth), f"Judged frame {fidx} out of range"
        assert truth[fidx]["show_gauge"] is True, f"Click judged on inactive frame {fidx}"

    # 2. 100% precision within Any-Overlap episodes
    for fidx in judged:
        assert any(s <= fidx <= e for s, e, _, _ in episodes), (
            f"Click judged at frame {fidx} outside all ground-truth Any-Overlap episodes {episodes}"
        )

    # 3. Zero double-clicks per episode
    hit_episodes: list[tuple[int, int, int, float]] = []
    for ep in episodes:
        ep_hits = [f for f in judged if ep[0] <= f <= ep[1]]
        assert len(ep_hits) <= 1, f"Double-click in synthetic episode {ep}: {ep_hits}"
        if len(ep_hits) == 1:
            hit_episodes.append(ep)

    # 4. Recall >= 90%, mandatory H10 (strike_idx == 10) and both tilt extremes (<= -10° and >= +10°)
    recall = len(hit_episodes) / len(episodes)
    assert recall >= 0.90, f"10-strike recall {recall:.2%} < 90% ({len(hit_episodes)}/{len(episodes)})"
    assert any(ep[2] == 10 for ep in hit_episodes), "Missed H10 (12 px @ 360 px/s)"
    assert any(ep[3] <= -10.0 for ep in hit_episodes), "Missed negative tilt extreme <= -10 deg"
    assert any(ep[3] >= 10.0 for ep in hit_episodes), "Missed positive tilt extreme >= +10 deg"


def test_t4_s4_reproducible_local_latency_benchmark_sla(
    e2e: Any,
    reference_video_path: Path,
) -> None:
    cv2, _ = e2e.import_cv2_np()
    FramePacket = e2e.import_attr("vrc_smith_auto.protocols", "FramePacket")

    # Preload 220 real 1080p frames covering inactive + active segments (first 10 as warmup)
    cap = cv2.VideoCapture(str(reference_video_path))
    assert cap.isOpened()
    packets: list[Any] = []
    try:
        for idx in range(220):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            packets.append(FramePacket(image_bgr=frame, timestamp_s=idx / 60.0, frame_idx=idx))
    finally:
        cap.release()
    assert len(packets) >= 210

    cfg = e2e.make_app_config(dry_run=True, require_foreground=False)
    det = e2e.make_gauge_detector(cfg)
    spy = e2e.make_spy_mouse()
    sm = e2e.make_trigger_state_machine(cfg, mouse=spy)

    # Warmup
    for pkt in packets[:10]:
        res = e2e.detect_gauge(det, pkt)
        e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)

    timings_ms: list[float] = []
    for pkt in packets[10:]:
        t0 = time.perf_counter()
        res = e2e.detect_gauge(det, pkt)
        e2e.step_state_machine(sm, res, pkt.timestamp_s, mouse=spy, frame_idx=pkt.frame_idx)
        spy.tick(pkt.timestamp_s)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_sorted = sorted(timings_ms)
    mean_ms = sum(timings_ms) / len(timings_ms)
    p95_idx = min(len(timings_sorted) - 1, int(math.ceil(0.95 * len(timings_sorted))) - 1)
    p95_ms = timings_sorted[p95_idx]

    assert mean_ms < 12.0, f"Direct pipeline mean latency {mean_ms:.2f} ms >= 12.0 ms"
    assert p95_ms < 16.7, f"Direct pipeline p95 latency {p95_ms:.2f} ms >= 16.7 ms"

    proc = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(reference_video_path),
            "--iterations",
            "200",
            "--max-p95-ms",
            "16.7",
            "--json",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    data = e2e.parse_cli_json(proc)
    cli_mean = float(data.get("mean_ms", data.get("avg_ms")))
    cli_p95 = float(data["p95_ms"])
    assert cli_mean < 12.0, f"CLI benchmark mean_ms {cli_mean:.2f} >= 12.0"
    assert cli_p95 < 16.7, f"CLI benchmark p95_ms {cli_p95:.2f} >= 16.7"


def test_t4_s5_end_to_end_cli_json_and_exit_code_contracts(
    e2e: Any,
    reference_screenshot_path: Path,
    reference_video_path: Path,
    tmp_path: Path,
) -> None:
    # 1. --help for root and all 4 subcommands
    for args in (["--help"], ["run", "--help"], ["verify-image", "--help"], ["verify-video", "--help"], ["benchmark", "--help"]):
        proc = e2e.run_cli(args)
        assert proc.returncode == 0, f"CLI {args} failed: {proc.stderr}"

    # 2. verify-image pass, expectation mismatch failure, and missing file failure
    proc_img_ok = e2e.run_cli(
        ["verify-image", "--image", str(reference_screenshot_path), "--expect-clicks", "0", "--json"]
    )
    assert proc_img_ok.returncode == 0, proc_img_ok.stderr
    c_img, _ = e2e.extract_cli_clicks(e2e.parse_cli_json(proc_img_ok))
    assert c_img == 0

    proc_img_mismatch = e2e.run_cli(
        ["verify-image", "--image", str(reference_screenshot_path), "--expect-clicks", "1", "--json"]
    )
    assert proc_img_mismatch.returncode != 0

    proc_img_missing = e2e.run_cli(
        ["verify-image", "--image", str(tmp_path / "nonexistent.png"), "--json"]
    )
    assert proc_img_missing.returncode != 0

    # 3. verify-video on real 1570-frame MP4 satisfies Scenario 2 metrics
    proc_vid = e2e.run_cli(
        ["verify-video", "--video", str(reference_video_path), "--dry-run", "--json"],
        timeout=180.0,
    )
    assert proc_vid.returncode == 0, proc_vid.stderr
    vid_data = e2e.parse_cli_json(proc_vid)
    cli_clicks, cli_frames = e2e.extract_cli_clicks(vid_data)
    assert cli_clicks >= 20
    assert len(cli_frames) == cli_clicks
    _assert_video_click_frames_meet_scenario2(cli_frames, judge_delay_frames=2)

    # 4. benchmark pass & threshold breach failure
    proc_bench_ok = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(reference_video_path),
            "--iterations",
            "30",
            "--max-p95-ms",
            "16.7",
            "--json",
        ]
    )
    assert proc_bench_ok.returncode == 0, proc_bench_ok.stderr
    e2e.parse_cli_json(proc_bench_ok)

    proc_bench_fail = e2e.run_cli(
        [
            "benchmark",
            "--video",
            str(reference_video_path),
            "--iterations",
            "15",
            "--max-p95-ms",
            "0.001",
            "--json",
        ]
    )
    assert proc_bench_fail.returncode != 0
