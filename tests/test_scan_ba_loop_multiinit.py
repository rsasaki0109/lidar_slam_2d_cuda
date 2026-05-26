"""Detection-side loop robustness (P-loop2): multi-init align + geometric verification.

The single LM align from `cur` only finds the basin nearest the current odometry pose,
so a rotated revisit is missed and a self-similar place can lock onto a low-rms but wrong
match the rms/inlier gates cannot catch. These cover the two new pieces:
  - _loop_match_ambiguous: the pure rival-basin decision (deterministic, no LM).
  - _align_loop_candidate: a yaw sweep recovers a rotated revisit a single init misses,
    and the default (0.0,) offset stays bit-identical to a single gated align.
"""
from __future__ import annotations

import math

import numpy as np

from slamx.core.scan_ba import ScanBaEngine, ScanBaEngineConfig
from slamx.core.scan_ba.align import align_scan_to_tsdf
from slamx.core.scan_ba.engine import _loop_match_ambiguous
from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.types import LaserScan, Pose2


# --- pure ambiguity decision -------------------------------------------------

def _sol(rms: float, x: float, y: float) -> tuple[float, Pose2]:
    return (rms, Pose2(x, y, 0.0))


def test_ambiguous_when_distinct_basin_aligns_nearly_as_well():
    # winner at (0,0) rms 0.10, a rival 3 m away at rms 0.11 -> within 30% margin -> ambiguous
    sols = [_sol(0.10, 0.0, 0.0), _sol(0.11, 3.0, 0.0)]
    assert _loop_match_ambiguous(sols, sep_m=0.5, margin=0.3) is True


def test_not_ambiguous_when_rival_clearly_worse():
    # rival is 3 m away but rms 0.20 vs 0.10 -> 100% worse, outside a 30% margin -> clear winner
    sols = [_sol(0.10, 0.0, 0.0), _sol(0.20, 3.0, 0.0)]
    assert _loop_match_ambiguous(sols, sep_m=0.5, margin=0.3) is False


def test_not_ambiguous_when_runners_up_are_the_same_basin():
    # the near-equal second solution is < sep_m away: agreement, not a rival
    sols = [_sol(0.10, 0.0, 0.0), _sol(0.10, 0.2, 0.0)]
    assert _loop_match_ambiguous(sols, sep_m=0.5, margin=0.3) is False


def test_margin_zero_disables_verification():
    sols = [_sol(0.10, 0.0, 0.0), _sol(0.10, 3.0, 0.0)]
    assert _loop_match_ambiguous(sols, sep_m=0.5, margin=0.0) is False


# --- multi-init align integration --------------------------------------------

def _segment_laserscan(pose: Pose2, segs, n_beams: int = 360) -> LaserScan:
    """Ray-cast a set of wall segments [(x1,y1,x2,y2), ...] from `pose`."""
    angle_min = -math.pi
    inc = 2.0 * math.pi / n_beams
    b = angle_min + np.arange(n_beams) * inc
    rx = np.cos(b + pose.theta)
    ry = np.sin(b + pose.theta)
    best = np.full(n_beams, np.inf)
    for x1, y1, x2, y2 in segs:
        sx, sy = x2 - x1, y2 - y1
        rxs = rx * sy - ry * sx  # r x s
        ax, ay = x1 - pose.x, y1 - pose.y
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * sy - ay * sx) / rxs  # along the ray
            u = (ax * ry - ay * rx) / rxs  # along the segment
        ok = np.isfinite(t) & (t > 0.05) & (u >= -1e-9) & (u <= 1 + 1e-9)
        best = np.where(ok & (t < best), t, best)
    ranges = np.where(np.isfinite(best) & (best <= 30.0), best, float("inf"))
    return LaserScan(
        stamp_ns=None,
        frame_id="laser",
        angle_min=angle_min,
        angle_max=angle_min + (n_beams - 1) * inc,
        angle_increment=inc,
        ranges=ranges,
        range_min=0.05,
        range_max=30.0,
    )


# L-shaped room: a 10x10 box with the top-right 6x6 corner removed. No rotational
# symmetry, so the alignment of a scan to this room has a single global optimum --
# the right setting to show multi-init recovering a rotated revisit deterministically.
_L_ROOM = [
    (0.0, 0.0, 10.0, 0.0),
    (10.0, 0.0, 10.0, 4.0),
    (10.0, 4.0, 4.0, 4.0),
    (4.0, 4.0, 4.0, 10.0),
    (4.0, 10.0, 0.0, 10.0),
    (0.0, 10.0, 0.0, 0.0),
]


def _candidate_engine(offsets, margin: float = 0.0) -> ScanBaEngine:
    cfg = ScanBaEngineConfig(
        tsdf=Tsdf2DConfig(
            resolution_m=0.05,
            origin_x_m=-2.0,
            origin_y_m=-2.0,
            size_x_m=14.0,
            size_y_m=14.0,
            truncation_m=0.6,
        ),
        loop_accept_inlier_ratio=0.4,
        loop_accept_rms_m=0.3,
        loop_max_correction_m=3.0,
        loop_init_yaw_offsets_rad=tuple(offsets),
        loop_ambiguity_margin=margin,
    )
    return ScanBaEngine(cfg=cfg)


def _verify_tsdf(eng: ScanBaEngine, pose: Pose2, pts: np.ndarray) -> Tsdf2D:
    tsdf = Tsdf2D.zeros(eng.cfg.tsdf)
    update_tsdf_from_scan(tsdf, pose_map=pose, points_sensor=pts, weight_inc=1.0, weight_max=100.0)
    return tsdf


def test_multi_init_recovers_rotated_revisit_single_init_misses():
    true_pose = Pose2(2.0, 2.0, 0.0)
    pts = _segment_laserscan(true_pose, _L_ROOM).points_xy()
    n_pts = pts.shape[0]
    # the verify map is built at the true pose; `cur` carries a large heading error
    # (2.5 rad ~ 143 deg) as if odometry yaw drifted before the revisit.
    cur = Pose2(2.0, 2.0, true_pose.theta + 2.5)

    eng_single = _candidate_engine(offsets=(0.0,))
    tsdf = _verify_tsdf(eng_single, true_pose, pts)
    single = eng_single._align_loop_candidate(tsdf, pts, cur, n_pts)
    # from 143 deg off, the single LM align cannot reach the true basin -> rejected
    assert single is None

    eng_multi = _candidate_engine(offsets=(0.0, -1.25, -2.5, 1.25, 2.5))
    multi = eng_multi._align_loop_candidate(tsdf, pts, cur, n_pts)
    assert multi is not None, "yaw sweep should recover the rotated revisit"
    pose, _inl, _rms = multi
    assert math.hypot(pose.x - true_pose.x, pose.y - true_pose.y) < 0.25
    dth = (pose.theta - true_pose.theta + math.pi) % (2 * math.pi) - math.pi
    assert abs(dth) < 0.15


def test_default_offsets_match_single_gated_align():
    # default (0.0,) + margin 0 must equal a plain single align + inlier/rms/corr gate
    true_pose = Pose2(3.0, 3.0, 0.4)
    pts = _segment_laserscan(true_pose, _L_ROOM).points_xy()
    n_pts = pts.shape[0]
    cur = Pose2(3.05, 2.97, 0.42)  # small odom offset, well inside the basin

    eng = _candidate_engine(offsets=(0.0,))
    tsdf = _verify_tsdf(eng, true_pose, pts)
    got = eng._align_loop_candidate(tsdf, pts, cur, n_pts)

    res = align_scan_to_tsdf(
        tsdf=tsdf, scan_xy=pts, pose_init=cur,
        max_iters=eng.cfg.optimize_max_iters, huber_delta_m=eng.cfg.huber_delta_m,
    )
    inl = res.num_inliers / n_pts
    rms = float(np.sqrt(2.0 * res.final_cost / max(1, res.num_inliers)))
    corr = float(np.hypot(res.pose.x - cur.x, res.pose.y - cur.y))
    expect_ok = (
        inl >= eng.cfg.loop_accept_inlier_ratio
        and rms <= eng.cfg.loop_accept_rms_m
        and corr <= eng.cfg.loop_max_correction_m
    )
    assert (got is not None) == expect_ok
    if got is not None:
        assert got[0].x == res.pose.x and got[0].y == res.pose.y and got[0].theta == res.pose.theta


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
