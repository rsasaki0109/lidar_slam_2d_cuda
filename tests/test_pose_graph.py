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


def _square_loop() -> list[Pose2]:
    """Six poses around a loop that returns near the start (perfect-odometry chain)."""
    return [
        Pose2(0.0, 0.0, 0.0),
        Pose2(1.0, 0.0, 0.0),
        Pose2(2.0, 0.0, 0.5 * math.pi),
        Pose2(2.0, 1.0, math.pi),
        Pose2(1.0, 1.0, math.pi),
        Pose2(0.0, 1.0, -0.5 * math.pi),
    ]


def _chain_edges(truth: list[Pose2]) -> list[Edge]:
    """Consistent sequential odometry edges from the ground-truth trajectory."""
    return [Edge(i=i, j=i + 1, rel=_rel(truth[i], truth[i + 1])) for i in range(len(truth) - 1)]


def _max_dev(poses: list[Pose2], truth: list[Pose2]) -> float:
    return max(float(math.hypot(p.x - t.x, p.y - t.y)) for p, t in zip(poses, truth, strict=True))


def test_robust_loss_rejects_false_loop_edge() -> None:
    """A grossly wrong loop edge must not be allowed to wreck the trajectory.

    With perfect odometry + a true loop edge the solution is the ground truth. We then add
    one false loop edge claiming pose 3 coincides with pose 0 (a gross outlier). The OLD
    behaviour -- a plain L2 solve with the loop edge unweighted -- lets that single edge
    compromise the whole trajectory. The shipped behaviour -- the loop edge down-weighted
    to its gate-minimum confidence (inlier ratio 0.4) plus the engine's redescending cauchy
    kernel (f_scale 0.5) -- clips the outlier so the trajectory stays close to truth.
    """
    truth = _square_loop()
    good_loop = Edge(i=0, j=5, rel=_rel(truth[0], truth[5]))

    def _solve(weight: float, loss: str, f_scale: float) -> list[Pose2]:
        false_loop = Edge(i=0, j=3, rel=Pose2(0.0, 0.0, 0.0), weight=weight)
        g = PoseGraph(
            poses=list(truth),  # initialise at truth to isolate the false edge's effect
            edges=[*_chain_edges(truth), good_loop, false_loop],
            cfg=PoseGraphConfig(max_iterations=200, robust_loss=loss, robust_f_scale=f_scale),
        )
        g.optimize()
        return g.poses

    dev_naive = _max_dev(_solve(1.0, "linear", 1.0), truth)  # old: unweighted L2
    dev_robust = _max_dev(_solve(0.4, "cauchy", 0.5), truth)  # shipped: weighted + cauchy

    # the plain L2 solve is dragged a long way off the truth by the single outlier ...
    assert dev_naive > 1.0
    # ... while the weighted + robust solve keeps the trajectory bounded ...
    assert dev_robust < 0.35
    # ... an order-of-magnitude improvement over trusting the bad edge.
    assert dev_robust < 0.25 * dev_naive


def test_edge_weight_down_weights_conflicting_edge() -> None:
    """A low-weight conflicting edge perturbs the solution less than a full-weight one."""
    truth = _square_loop()
    conflicting = Pose2(0.6, 0.0, 0.0)  # claims pose 3 ~ pose 0 (wrong), under linear loss

    def _solve(weight: float) -> list[Pose2]:
        g = PoseGraph(
            poses=list(truth),
            edges=[*_chain_edges(truth), Edge(i=0, j=3, rel=conflicting, weight=weight)],
            cfg=PoseGraphConfig(max_iterations=100),  # linear loss: no robust clipping
        )
        g.optimize()
        return g.poses

    strong = _max_dev(_solve(1.0), truth)
    weak = _max_dev(_solve(0.1), truth)
    assert weak < strong
    # near-zero weight should leave the truth essentially untouched
    assert _max_dev(_solve(1e-4), truth) < 1e-2
