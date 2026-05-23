from __future__ import annotations

import math

import numpy as np

from slamx.core.scan_ba import ScanBaEngine, ScanBaEngineConfig
from slamx.core.scan_ba.tsdf import Tsdf2DConfig
from slamx.core.types import LaserScan, Pose2


def _box_laserscan(pose: Pose2, n_beams: int = 360, lo: float = 0.0, hi: float = 10.0) -> LaserScan:
    """Ray-cast a closed box room (walls x=lo,hi and y=lo,hi) from `pose`."""
    angle_min = -math.pi
    inc = 2.0 * math.pi / n_beams
    ang = angle_min + np.arange(n_beams) * inc
    ca = np.cos(ang + pose.theta)
    sa = np.sin(ang + pose.theta)
    ts = np.full(n_beams, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # vertical walls x = lo, hi: t = (x - px)/ca, hit y must be on segment
        for wx in (lo, hi):
            t = (wx - pose.x) / ca
            hy = pose.y + t * sa
            ok = (t > 0.05) & np.isfinite(t) & (hy >= lo - 1e-6) & (hy <= hi + 1e-6)
            ts = np.where(ok & (t < ts), t, ts)
        # horizontal walls y = lo, hi: t = (y - py)/sa, hit x must be on segment
        for wy in (lo, hi):
            t = (wy - pose.y) / sa
            hx = pose.x + t * ca
            ok = (t > 0.05) & np.isfinite(t) & (hx >= lo - 1e-6) & (hx <= hi + 1e-6)
            ts = np.where(ok & (t < ts), t, ts)
    ranges = np.where(np.isfinite(ts) & (ts <= 30.0), ts, float("inf"))
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


def _loop_trajectory(n: int = 60, cx: float = 5.0, cy: float = 5.0, r: float = 2.0) -> list[Pose2]:
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        th = a + math.pi / 2.0  # tangent heading
        out.append(Pose2(x, y, th))
    return out


def _engine(loop: bool) -> ScanBaEngine:
    cfg = ScanBaEngineConfig(
        tsdf=Tsdf2DConfig(
            resolution_m=0.05,
            origin_x_m=-2.0,
            origin_y_m=-2.0,
            size_x_m=14.0,
            size_y_m=14.0,
            truncation_m=0.6,
        ),
        window_size=8,
        seed_scans=3,
        map_window=15,
        prediction_mode="constant_velocity",
        loop_closure_enabled=loop,
        loop_detect_every_n=3,
        loop_dist_m=1.0,
        loop_min_gap=25,
        loop_max_candidates=2,
        loop_submap_window=6,
        loop_accept_inlier_ratio=0.4,
        loop_accept_cost=0.2,
        loop_max_correction_m=2.0,
    )
    return ScanBaEngine(cfg=cfg)


def test_loop_closure_adds_edge_when_revisiting_start():
    gt = _loop_trajectory(n=60)
    # repeat the first few poses so the robot clearly revisits the start
    gt = gt + gt[:8]
    eng = _engine(loop=True)
    for p in gt:
        eng.handle_scan(_box_laserscan(p))

    assert len(eng._loop_edges) >= 1, "expected at least one loop closure edge"
    # the pose graph should hold odom edges (N-1) plus loop edges
    n = len(eng.graph.poses)
    assert len(eng.graph.edges) >= (n - 1) + 1


def test_loop_closure_disabled_adds_no_loop_edges():
    gt = _loop_trajectory(n=60) + _loop_trajectory(n=60)[:8]
    eng = _engine(loop=False)
    for p in gt:
        eng.handle_scan(_box_laserscan(p))
    assert len(eng._loop_edges) == 0
    # only sequential odom edges
    assert len(eng.graph.edges) == len(eng.graph.poses) - 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
