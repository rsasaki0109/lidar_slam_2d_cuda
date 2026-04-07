from __future__ import annotations

import numpy as np

from slamx.core.backend.pose_graph import Edge, PoseGraph, PoseGraphConfig
from slamx.core.types import Pose2


def test_full_graph_reduces_odometry_residual() -> None:
    g = PoseGraph(cfg=PoseGraphConfig(optimization_window=None, max_iterations=40))
    g.add_pose(Pose2(0.0, 0.0, 0.0))
    g.add_pose(Pose2(1.0, 0.0, 0.0))
    g.add_pose(Pose2(2.0, 0.1, 0.0))  # small lateral error; chain wants y≈0
    g.add_edge(Edge(0, 1, Pose2(1.0, 0.0, 0.0)))
    g.add_edge(Edge(1, 2, Pose2(1.0, 0.0, 0.0)))
    out = g.optimize()
    assert out["success"]
    assert out["residual_rms_after"] <= out["residual_rms_before"] + 1e-6
    assert abs(g.poses[2].y) < 0.06


def test_window_preserves_frozen_prefix() -> None:
    cfg = PoseGraphConfig(optimization_window=2, max_iterations=80, max_nfev_cap=50000)
    g = PoseGraph(cfg=cfg)
    g.add_pose(Pose2(0.0, 0.0, 0.0))
    g.add_pose(Pose2(1.0, 0.0, 0.0))
    g.add_pose(Pose2(2.0, 0.0, 0.0))
    g.add_pose(Pose2(100.0, 0.0, 0.0))
    g.add_edge(Edge(0, 1, Pose2(1.0, 0.0, 0.0)))
    g.add_edge(Edge(1, 2, Pose2(1.0, 0.0, 0.0)))
    g.add_edge(Edge(2, 3, Pose2(1.0, 0.0, 0.0)))
    snap0 = np.array([g.poses[0].x, g.poses[0].y, g.poses[0].theta])
    snap1 = np.array([g.poses[1].x, g.poses[1].y, g.poses[1].theta])
    out = g.optimize()
    assert out.get("optimization_window_start") == 2
    np.testing.assert_allclose(
        [g.poses[0].x, g.poses[0].y, g.poses[0].theta], snap0, atol=1e-9
    )
    np.testing.assert_allclose(
        [g.poses[1].x, g.poses[1].y, g.poses[1].theta], snap1, atol=1e-9
    )
    assert abs(g.poses[3].x - 3.0) < 0.2
