from __future__ import annotations

import math

from slamx.core.backend.pose_graph import Edge, PoseGraph, PoseGraphConfig
from slamx.core.types import Pose2


def _rel(a: Pose2, b: Pose2) -> Pose2:
    return a.inverse().compose(b)


def test_pose_graph_optimize_keeps_anchor_fixed_and_closes_loop() -> None:
    truth = [
        Pose2(1.5, -0.5, 0.2),
        Pose2(2.7, 0.1, 0.3),
        Pose2(3.0, 1.4, 0.8),
    ]
    graph = PoseGraph(
        poses=[
            truth[0],
            Pose2(2.9, -0.2, 0.05),
            Pose2(2.5, 1.9, 1.1),
        ],
        edges=[
            Edge(i=0, j=1, rel=_rel(truth[0], truth[1])),
            Edge(i=1, j=2, rel=_rel(truth[1], truth[2])),
            Edge(i=0, j=2, rel=_rel(truth[0], truth[2])),
        ],
        cfg=PoseGraphConfig(max_iterations=20, max_nfev_cap=200),
    )

    anchor = graph.poses[0]
    rep = graph.optimize()

    assert rep["success"]
    assert graph.poses[0] == anchor
    assert rep["residual_rms_after"] < rep["residual_rms_before"]
    assert rep["residual_rms_after"] < 1e-9

    for got, want in zip(graph.poses[1:], truth[1:], strict=True):
        assert math.isclose(got.x, want.x, abs_tol=1e-6)
        assert math.isclose(got.y, want.y, abs_tol=1e-6)
        assert math.isclose(got.theta, want.theta, abs_tol=1e-6)
