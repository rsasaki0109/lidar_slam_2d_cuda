from __future__ import annotations

import numpy as np

from slamx.core.frontend.local_slam import LocalSlamConfig, LocalSlamEngine
from slamx.core.types import LaserScan, MatchResult, Pose2


class RecordingMatcher:
    def __init__(self, returned_poses: list[Pose2]) -> None:
        self._returned_poses = list(returned_poses)
        self.predictions: list[Pose2] = []

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        self.predictions.append(prediction_map)
        pose = self._returned_poses.pop(0)
        return MatchResult(pose_map=pose, score=0.0, candidates=[], diagnostics={})


def _scan(stamp_ns: int) -> LaserScan:
    return LaserScan(
        stamp_ns=stamp_ns,
        frame_id="laser",
        angle_min=0.0,
        angle_max=0.0,
        angle_increment=1.0,
        ranges=np.array([1.0], dtype=np.float64),
        range_min=0.0,
        range_max=10.0,
    )


def test_local_slam_hold_prediction_uses_last_pose() -> None:
    eng = LocalSlamEngine(cfg=LocalSlamConfig(prediction_mode="hold"))
    matcher = RecordingMatcher([Pose2(1.0, 0.0, 0.0), Pose2(1.5, 0.0, 0.0)])
    eng._matcher = matcher

    eng.handle_scan(_scan(0))
    eng.handle_scan(_scan(1))
    eng.handle_scan(_scan(2))

    assert matcher.predictions[0] == Pose2(0.0, 0.0, 0.0)
    assert matcher.predictions[1] == Pose2(1.0, 0.0, 0.0)


def test_local_slam_constant_velocity_prediction_uses_last_delta() -> None:
    eng = LocalSlamEngine(cfg=LocalSlamConfig(prediction_mode="constant_velocity"))
    matcher = RecordingMatcher([Pose2(1.0, 0.0, 0.0), Pose2(2.0, 0.0, 0.0)])
    eng._matcher = matcher

    eng.handle_scan(_scan(0))
    eng.handle_scan(_scan(1))
    eng.handle_scan(_scan(2))

    assert matcher.predictions[0] == Pose2(0.0, 0.0, 0.0)
    assert matcher.predictions[1] == Pose2(2.0, 0.0, 0.0)
