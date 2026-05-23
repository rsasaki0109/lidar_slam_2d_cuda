from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.io.bag import ImuSample
from slamx.core.observability import JsonlTelemetry
from slamx.core.preprocess.pipeline import PreprocessConfig, preprocess_scan
from slamx.core.scan_ba.align import align_scan_to_tsdf
from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.scan_ba.window import AnchorPrior, MotionPrior, WindowState, optimize_window
from slamx.core.types import LaserScan, Pose2


@dataclass
class ScanBaEngineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tsdf: Tsdf2DConfig = field(default_factory=Tsdf2DConfig)
    window_size: int = 10
    seed_scans: int = 3  # bootstrap with single-scan alignment before window opt
    motion_prior_info_xy: float = 5.0
    motion_prior_info_theta: float = 5.0
    anchor_info_xy: float = 1.0e5
    anchor_info_theta: float = 1.0e5
    huber_delta_m: float = 0.15
    optimize_max_iters: int = 20
    prediction_mode: str = "constant_velocity"  # "hold" | "constant_velocity"
    tsdf_weight_inc: float = 1.0
    tsdf_weight_max: float = 100.0


@dataclass
class _GraphShim:
    """Minimal stand-in for PoseGraph so the replay CLI can read poses uniformly."""

    poses: list[Pose2] = field(default_factory=list)

    def optimize(self) -> dict:
        return {"backend": "scan_ba", "skipped_global_optimize": True}


@dataclass
class ScanBaEngine:
    """Fixed-lag scan-level BA frontend (CPU reference path).

    Drop-in for the replay CLI: exposes handle_scan, graph.poses, stamps_ns,
    and a no-op set_imu_buffer.
    """

    cfg: ScanBaEngineConfig = field(default_factory=ScanBaEngineConfig)
    telemetry: JsonlTelemetry | None = None

    graph: _GraphShim = field(init=False)
    _tsdf: Tsdf2D = field(init=False, repr=False)
    _scans: list[np.ndarray] = field(default_factory=list, repr=False)
    _stamps: list[int | None] = field(default_factory=list)
    _last_rel: Pose2 | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.graph = _GraphShim(poses=[])
        self._tsdf = Tsdf2D.zeros(self.cfg.tsdf)

    def set_imu_buffer(self, samples: list[ImuSample]) -> None:  # noqa: ARG002 - parity stub
        return None

    @property
    def poses(self) -> list[Pose2]:
        return list(self.graph.poses)

    @property
    def stamps_ns(self) -> list[int | None]:
        return list(self._stamps)

    def _predict(self) -> Pose2:
        last = self.graph.poses[-1]
        mode = (self.cfg.prediction_mode or "hold").lower()
        if mode in {"hold", "zero_velocity"} or self._last_rel is None:
            return last
        return last.compose(self._last_rel)

    def _fold_tsdf(self, pose: Pose2, pts: np.ndarray) -> int:
        return update_tsdf_from_scan(
            self._tsdf,
            pose_map=pose,
            points_sensor=pts,
            weight_inc=self.cfg.tsdf_weight_inc,
            weight_max=self.cfg.tsdf_weight_max,
        )

    def handle_scan(self, scan: LaserScan) -> Pose2:
        scan_p = preprocess_scan(scan, self.cfg.preprocess)
        pts = scan_p.points_xy()

        if not self.graph.poses:
            init = Pose2(0.0, 0.0, 0.0)
            self.graph.poses.append(init)
            self._scans.append(pts)
            self._stamps.append(scan.stamp_ns)
            self._fold_tsdf(init, pts)
            self._emit(0, init, score=0.0)
            return init

        prediction = self._predict()

        if len(self.graph.poses) < self.cfg.seed_scans:
            res = align_scan_to_tsdf(
                tsdf=self._tsdf,
                scan_xy=pts,
                pose_init=prediction,
                max_iters=self.cfg.optimize_max_iters,
                huber_delta_m=self.cfg.huber_delta_m,
            )
            pose = res.pose
            score = res.final_cost
        else:
            k_hist = min(self.cfg.window_size - 1, len(self.graph.poses))
            hist_poses = self.graph.poses[-k_hist:]
            hist_scans = self._scans[-k_hist:]
            window_poses = list(hist_poses) + [prediction]
            window_scans = list(hist_scans) + [pts]
            motion_priors = [
                MotionPrior(
                    delta_x=window_poses[i + 1].x - window_poses[i].x,
                    delta_y=window_poses[i + 1].y - window_poses[i].y,
                    delta_theta=window_poses[i + 1].theta - window_poses[i].theta,
                    info_xy=self.cfg.motion_prior_info_xy,
                    info_theta=self.cfg.motion_prior_info_theta,
                )
                for i in range(len(window_poses) - 1)
            ]
            anchor = AnchorPrior(
                pose=window_poses[0],
                info_xy=self.cfg.anchor_info_xy,
                info_theta=self.cfg.anchor_info_theta,
            )
            state = WindowState(
                poses=window_poses,
                scans=window_scans,
                motion_priors=motion_priors,
                anchor=anchor,
            )
            res = optimize_window(
                tsdf=self._tsdf,
                state=state,
                max_iters=self.cfg.optimize_max_iters,
                huber_delta_m=self.cfg.huber_delta_m,
            )
            # write optimized history back (last k_hist poses), keep new pose separate
            self.graph.poses[-k_hist:] = res.state.poses[:-1]
            pose = res.state.poses[-1]
            score = res.final_cost

        prev = self.graph.poses[-1]
        self._last_rel = prev.inverse().compose(pose)
        self.graph.poses.append(pose)
        self._scans.append(pts)
        self._stamps.append(scan.stamp_ns)
        self._fold_tsdf(pose, pts)
        self._emit(len(self.graph.poses) - 1, pose, score=score)
        return pose

    def _emit(self, node: int, pose: Pose2, *, score: float) -> None:
        if not self.telemetry:
            return
        self.telemetry.emit(
            "keyframe",
            {
                "node": node,
                "stamp_ns": self._stamps[node] if node < len(self._stamps) else None,
                "pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
                "scan_match_score": float(score),
                "backend": "scan_ba",
            },
        )
