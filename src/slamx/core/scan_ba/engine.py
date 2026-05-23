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
    # Local map: rebuild a crisp TSDF from the most recent `map_window` scans each step
    # (scan-to-local-submap). A persistent global accumulation blurs surfaces as the
    # robot moves and causes the tracker to lag (stay-put becomes low cost), so we keep
    # only recent history in the active map.
    map_window: int = 20
    motion_prior_info_xy: float = 5.0
    motion_prior_info_theta: float = 5.0
    anchor_info_xy: float = 1.0e5
    anchor_info_theta: float = 1.0e5
    huber_delta_m: float = 0.15
    optimize_max_iters: int = 20
    prediction_mode: str = "constant_velocity"  # "hold" | "constant_velocity"
    tsdf_weight_inc: float = 1.0
    tsdf_weight_max: float = 100.0
    # robustness gating: reject alignments that are under-constrained or that jump
    # implausibly far, falling back to the motion prediction. Guards the sparse-map
    # bootstrap (spurious minima) and large mis-registrations.
    min_inlier_ratio: float = 0.25
    max_step_m: float = 1.0


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

    def _gate(self, pose: Pose2, prediction: Pose2, inliers: int, n_pts: int) -> Pose2:
        """Reject an alignment that is under-constrained or jumps too far.

        Returns `pose` (same identity) when accepted, else `prediction`.
        """
        if n_pts > 0 and (inliers / n_pts) < self.cfg.min_inlier_ratio:
            return prediction
        last = self.graph.poses[-1]
        step = float(np.hypot(pose.x - last.x, pose.y - last.y))
        if step > self.cfg.max_step_m:
            return prediction
        return pose

    def _rebuild_local_map(self, count: int) -> None:
        """Rebuild the active TSDF from the most recent `map_window` historical scans.

        Builds a crisp local submap at the current pose estimates rather than a
        persistent global accumulation (which blurs as the robot moves).
        `count` is the number of stored scans to treat as history (excludes the
        scan currently being aligned).
        """
        self._tsdf.phi[:] = 0.0
        self._tsdf.weight[:] = 0.0
        mw = max(1, int(self.cfg.map_window))
        start = max(0, count - mw)
        for i in range(start, count):
            update_tsdf_from_scan(
                self._tsdf,
                pose_map=self.graph.poses[i],
                points_sensor=self._scans[i],
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
            self._emit(0, init, score=0.0)
            return init

        prediction = self._predict()
        n_hist = len(self.graph.poses)
        # rebuild a crisp local map from recent history (excludes the new scan)
        self._rebuild_local_map(n_hist)

        n_pts = int(pts.shape[0])
        if n_hist < self.cfg.seed_scans:
            res = align_scan_to_tsdf(
                tsdf=self._tsdf,
                scan_xy=pts,
                pose_init=prediction,
                max_iters=self.cfg.optimize_max_iters,
                huber_delta_m=self.cfg.huber_delta_m,
            )
            pose = self._gate(res.pose, prediction, res.num_inliers, n_pts)
            score = res.final_cost
        else:
            k_hist = min(self.cfg.window_size - 1, n_hist)
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
            new_inliers = res.diagnostics.get("inliers_per_scan", [n_pts])[-1]
            gated = self._gate(res.state.poses[-1], prediction, new_inliers, n_pts)
            if gated is res.state.poses[-1]:
                # accept the joint solution (history refinement included)
                self.graph.poses[-k_hist:] = res.state.poses[:-1]
            pose = gated
            score = res.final_cost

        prev = self.graph.poses[-1]
        self._last_rel = prev.inverse().compose(pose)
        self.graph.poses.append(pose)
        self._scans.append(pts)
        self._stamps.append(scan.stamp_ns)
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
