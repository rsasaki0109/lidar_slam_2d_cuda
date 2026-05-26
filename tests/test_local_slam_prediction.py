from __future__ import annotations

import numpy as np

from slamx.core.backend.pose_graph import PoseGraphConfig
from slamx.core.frontend.local_slam import LocalSlamConfig, LocalSlamEngine
from slamx.core.loop_detection.heuristic import LoopClosureResult
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


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event_type: str, payload: dict) -> None:
        self.events.append({"type": event_type, **payload})


class LoopOnce:
    def detect_and_match(self, **kwargs) -> list[LoopClosureResult]:
        node = int(kwargs["node_id"])
        if node != 2:
            return []
        return [
            LoopClosureResult(
                i=0,
                j=node,
                score=0.0,
                accepted=True,
                rel_ij=Pose2(2.0, 0.0, 0.0),
                diagnostics={},
            )
        ]


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


def test_loop_closure_can_trigger_low_budget_optimization() -> None:
    telemetry = RecordingTelemetry()
    eng = LocalSlamEngine(
        cfg=LocalSlamConfig(
            prediction_mode="constant_velocity",
            optimize_every_n_keyframes=0,
            optimize_on_loop_closure=True,
            loop_pose_graph=PoseGraphConfig(max_iterations=1, max_nfev_cap=32),
            loop_edge_weight=2.0,
        ),
        telemetry=telemetry,  # type: ignore[arg-type]
    )
    eng._matcher = RecordingMatcher([Pose2(1.0, 0.0, 0.0), Pose2(2.0, 0.0, 0.0)])
    eng._heuristic_loop = LoopOnce()  # type: ignore[assignment]

    eng.handle_scan(_scan(0))
    eng.handle_scan(_scan(1))
    eng.handle_scan(_scan(2))

    accepted = [e for e in telemetry.events if e["type"] == "loop_closure_accepted"]
    opts = [e for e in telemetry.events if e["type"] == "optimization"]

    assert accepted
    assert accepted[0]["weight"] == 2.0
    assert len(opts) == 1
    assert opts[0]["reason"] == "loop_closure"
    assert opts[0]["max_nfev"] == 32
