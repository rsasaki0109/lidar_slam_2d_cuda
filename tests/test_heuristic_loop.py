from __future__ import annotations

import numpy as np

from slamx.core.loop_detection.heuristic import HeuristicLoopConfig, HeuristicLoopDetector
from slamx.core.types import LaserScan, MatchResult, Pose2


class StaticMatcher:
    def __init__(self, pose: Pose2, score: float, diagnostics: dict | None = None) -> None:
        self.pose = pose
        self.score = score
        self.diagnostics = diagnostics or {}

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        return MatchResult(
            pose_map=self.pose,
            score=self.score,
            candidates=[],
            diagnostics=self.diagnostics,
        )


def _scan() -> LaserScan:
    return LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=0.0,
        angle_max=0.0,
        angle_increment=1.0,
        ranges=np.array([1.0], dtype=np.float64),
        range_min=0.0,
        range_max=10.0,
    )


def test_rejected_loop_candidate_keeps_relative_pose_and_gate_diagnostics() -> None:
    detector = HeuristicLoopDetector(
        HeuristicLoopConfig(
            enabled=True,
            search_radius_m=0.5,
            min_separation_nodes=1,
            max_candidates=1,
            accept_score=-1.0,
            icp_accept_rms=0.1,
        )
    )
    poses = [Pose2(0.0, 0.0, 0.0), Pose2(1.0, 0.0, 0.0), Pose2(0.05, 0.0, 0.0)]
    refs = [
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0]], dtype=np.float64),
    ]

    result = detector.detect_and_match(
        matcher=StaticMatcher(Pose2(0.1, 0.0, 0.0), 0.0),
        refiner=StaticMatcher(
            Pose2(0.2, 0.0, 0.0),
            -0.2,
            {"icp": {"final_rms": 0.2}},
        ),
        node_id=2,
        pose_map=poses[2],
        scan=_scan(),
        poses=poses,
        ref_points_by_node=refs,
    )

    assert len(result) == 1
    candidate = result[0]
    assert candidate.accepted is False
    assert candidate.rel_ij == Pose2(0.2, 0.0, 0.0)
    assert candidate.diagnostics["acceptance"]["score_ok"] is True
    assert candidate.diagnostics["acceptance"]["rms_ok"] is False
    assert candidate.diagnostics["acceptance"]["reject_reason"] == "icp_rms"
