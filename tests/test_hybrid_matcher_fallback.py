from __future__ import annotations

import math

import numpy as np

from slamx.core.local_matching.correlative import CorrelativeGridConfig
from slamx.core.local_matching.hybrid import (
    HybridFallbackConfig,
    HybridRefinementConfig,
    HybridScanMatcher,
)
from slamx.core.local_matching.icp import IcpConfig
from slamx.core.types import LaserScan, MatchResult, Pose2, transform_points_xy


def _pose_error(a: Pose2, b: Pose2) -> float:
    return float(
        math.hypot(a.x - b.x, a.y - b.y)
        + abs(math.atan2(math.sin(a.theta - b.theta), math.cos(a.theta - b.theta)))
    )


def test_hybrid_fallback_recovers_when_primary_window_is_too_narrow() -> None:
    scan = LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=-1.0,
        angle_max=1.0,
        angle_increment=1.0 / 3.0,
        ranges=np.array([1.6, 1.1, 0.7, 1.3, 0.9, 1.4, 0.8], dtype=np.float64),
        range_min=0.05,
        range_max=10.0,
    )
    true_pose = Pose2(0.28, -0.15, math.radians(14.0))
    ref = transform_points_xy(true_pose.as_se2(), scan.points_xy())
    prediction = Pose2(0.0, 0.0, 0.0)

    primary_grid = CorrelativeGridConfig(
        linear_step_m=0.05,
        angular_step_deg=2.0,
        linear_window_m=0.05,
        angular_window_deg=4.0,
        sigma_hit_m=0.05,
    )
    icp = IcpConfig(
        max_iterations=20,
        max_correspondence_dist_m=0.10,
        min_correspondences=4,
        trim_fraction=0.0,
    )

    narrow = HybridScanMatcher(primary_grid, icp)
    fallback = HybridScanMatcher(
        primary_grid,
        icp,
        fallback=HybridFallbackConfig(
            enabled=True,
            trigger_score=0.0,
            correlative=CorrelativeGridConfig(
                linear_step_m=0.05,
                angular_step_deg=2.0,
                linear_window_m=0.30,
                angular_window_deg=20.0,
                sigma_hit_m=0.05,
            ),
        ),
    )

    narrow_mr = narrow.match(
        scan=scan,
        prediction_map=prediction,
        ref_points_xy_map=ref,
    )
    fallback_mr = fallback.match(
        scan=scan,
        prediction_map=prediction,
        ref_points_xy_map=ref,
    )

    assert _pose_error(narrow_mr.pose_map, true_pose) > 0.08
    assert _pose_error(fallback_mr.pose_map, true_pose) < 0.03
    assert fallback_mr.diagnostics["hybrid"]["used_fallback"] is True
    assert fallback_mr.score > narrow_mr.score


def test_hybrid_refines_multiple_candidates_and_keeps_best_icp_result() -> None:
    class FakeCoarse:
        def match(self, **_kwargs) -> MatchResult:
            return MatchResult(
                pose_map=Pose2(0.0, 0.0, 0.0),
                score=-1.0,
                candidates=[
                    (0.0, 0.0, 0.0, -1.0),
                    (1.0, 0.0, 0.0, -1.1),
                    (2.0, 0.0, 0.0, -1.2),
                ],
                diagnostics={"grid": {"lin": 3, "ang": 1}},
            )

    class FakeRefine:
        def match(self, *, prediction_map: Pose2, **_kwargs) -> MatchResult:
            score_by_x = {0.0: -0.5, 1.0: -0.1, 2.0: -0.3}
            score = score_by_x[prediction_map.x]
            return MatchResult(
                pose_map=prediction_map,
                score=score,
                candidates=[(prediction_map.x, prediction_map.y, prediction_map.theta, score)],
                diagnostics={"icp": {"final_rms": -score}},
            )

    scan = LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=-0.1,
        angle_max=0.1,
        angle_increment=0.1,
        ranges=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        range_min=0.05,
        range_max=10.0,
    )
    matcher = HybridScanMatcher(
        refinement=HybridRefinementConfig(top_k=3, min_linear_dist_m=0.1),
    )
    matcher._coarse = FakeCoarse()
    matcher._refine = FakeRefine()

    mr = matcher.match(
        scan=scan,
        prediction_map=Pose2(0.0, 0.0, 0.0),
        ref_points_xy_map=np.zeros((3, 2), dtype=np.float64),
    )

    assert mr.pose_map == Pose2(1.0, 0.0, 0.0)
    assert mr.score == -0.1
    assert mr.diagnostics["hybrid"]["n_refined_candidates"] == 3
    assert mr.diagnostics["hybrid"]["best_candidate_index"] == 1
