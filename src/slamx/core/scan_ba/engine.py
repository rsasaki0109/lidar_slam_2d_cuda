from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.backend.pose_graph import Edge, PoseGraph, PoseGraphConfig
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
    # loop closure (CudaRobotics gpu_online_slam style): distance-based detection
    # against past nodes, TSDF-verified, then a global pose-graph solve. The fixed-lag
    # window is the front-end odometry; the pose graph corrects accumulated drift.
    loop_closure_enabled: bool = False
    loop_detect_every_n: int = 5
    loop_dist_m: float = 2.5
    loop_min_gap: int = 30
    loop_max_candidates: int = 2
    loop_submap_window: int = 10  # scans around a candidate used to build its verify TSDF
    loop_accept_inlier_ratio: float = 0.4
    # RMS residual of the verification alignment (scale-invariant: sqrt(2*cost/inliers)).
    # A total-cost threshold would scale with point count and never fire on real scans.
    loop_accept_rms_m: float = 0.3
    loop_max_correction_m: float = 1.5  # reject verify poses too far from odom prediction
    # Run the fixed-lag window LM solve on the GPU (fused CUDA kernel, P2.5/P2.9).
    # Numerically identical to the CPU path; falls back to CPU if cupy/CUDA is absent.
    use_cuda: bool = False
    # Joint pose+SDF bundle adjustment (P3): the window solve also refines the local
    # map's SDF voxels, not just the K poses. CPU-only for now (takes precedence over
    # use_cuda). The refined map is transient -- the local map is rebuilt from poses
    # each scan -- so this mainly sharpens the map the current pose is registered to.
    use_joint: bool = False
    joint_sdf_prior_info: float = 10.0
    joint_sdf_smooth_info: float = 0.0
    # Exact sliding-window marginalization (P4.1): when a pose leaves the fixed-lag
    # window, Schur-eliminate it into a MarginalizationPrior on the new oldest pose
    # instead of pinning that pose with a heuristic strong anchor. Preserves the
    # information the dropped pose carried (first-estimate Jacobians). Requires the
    # CPU or joint path -- the device-resident GPU map is not host-resident for the
    # linearization, so marginalization is disabled when use_cuda forces the GPU path.
    use_marginalization: bool = False
    # Persistent global TSDF map (P-map): a single large map every accepted scan is
    # folded into, kept separate from the throwaway local tracking submap. Rebuilt from
    # corrected poses after each loop closure so it stays consistent with the optimized
    # graph. Off by default (build_global_map). `global_tsdf` is its (larger) extent.
    build_global_map: bool = False
    global_tsdf: Tsdf2DConfig = field(
        default_factory=lambda: Tsdf2DConfig(
            resolution_m=0.05, origin_x_m=-30.0, origin_y_m=-30.0,
            size_x_m=60.0, size_y_m=60.0, truncation_m=0.4,
        )
    )


@dataclass
class ScanBaEngine:
    """Fixed-lag scan-level BA frontend (CPU reference path).

    Drop-in for the replay CLI: exposes handle_scan, graph.poses, stamps_ns,
    and a no-op set_imu_buffer.
    """

    cfg: ScanBaEngineConfig = field(default_factory=ScanBaEngineConfig)
    telemetry: JsonlTelemetry | None = None

    graph: PoseGraph = field(init=False)
    _tsdf: Tsdf2D = field(init=False, repr=False)
    _scans: list[np.ndarray] = field(default_factory=list, repr=False)
    _stamps: list[int | None] = field(default_factory=list)
    _last_rel: Pose2 | None = field(default=None, init=False)
    _loop_edges: set[tuple[int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.graph = PoseGraph(cfg=PoseGraphConfig(max_iterations=50))
        self._tsdf = Tsdf2D.zeros(self.cfg.tsdf)
        self._window_solver = self._resolve_window_solver()
        # Joint mode runs the CPU joint solver and refines the SDF in-place; it is
        # incompatible with the device-resident GPU map, so it forces the CPU path.
        self._joint_active = self.cfg.use_joint
        # Device-resident local map: when the GPU path is active, keep phi/weight on
        # the device across the per-scan rebuild + window solve so there is no host
        # upload each scan. _scans_dev mirrors _scans as (N,2) float64 device arrays.
        self._cuda_active = (not self._joint_active) and self._window_solver is not optimize_window
        # Exact marginalization (P4.1): a running MarginalizationPrior on the current
        # oldest in-window pose, with the global pose index it linearizes. Recursively
        # updated each time the window drops a pose. Needs a host-resident TSDF to
        # linearize the dropped pose's data term, so it is off on the GPU map path.
        self._marg_active = self.cfg.use_marginalization and not self._cuda_active
        self._marg_prior = None
        self._marg_idx = -1
        # Persistent global map (separate from the tracking submap); see GlobalTsdfMap.
        self._global_map = None
        if self.cfg.build_global_map:
            from slamx.core.scan_ba.global_map import GlobalTsdfMap

            self._global_map = GlobalTsdfMap(
                self.cfg.global_tsdf,
                weight_inc=self.cfg.tsdf_weight_inc,
                weight_max=self.cfg.tsdf_weight_max,
            )
        self._scans_dev: list = []
        if self._cuda_active:
            from slamx.core.scan_ba import cuda

            self._cuda = cuda
            self._cp = cuda._cupy()
            self._phi_d = self._cp.zeros((self._tsdf.height, self._tsdf.width), dtype=self._cp.float32)
            self._weight_d = self._cp.zeros_like(self._phi_d)

    def _resolve_window_solver(self):
        """Pick the window LM backend. Joint pose+SDF BA (CPU) takes precedence; then
        the GPU fused kernel when requested+available; else the CPU pose-only reference.
        The GPU and CPU pose-only paths are numerically identical."""
        if self.cfg.use_joint:
            from functools import partial

            from slamx.core.scan_ba.joint import optimize_window_joint

            return partial(
                optimize_window_joint,
                sdf_prior_info=self.cfg.joint_sdf_prior_info,
                sdf_smooth_info=self.cfg.joint_sdf_smooth_info,
            )
        if not self.cfg.use_cuda:
            return optimize_window
        try:
            from slamx.core.scan_ba import cuda

            if cuda.is_available():
                return cuda.optimize_window_cuda
        except Exception:
            pass
        return optimize_window

    def set_imu_buffer(self, samples: list[ImuSample]) -> None:  # noqa: ARG002 - parity stub
        return None

    @property
    def poses(self) -> list[Pose2]:
        return list(self.graph.poses)

    @property
    def stamps_ns(self) -> list[int | None]:
        return list(self._stamps)

    @property
    def global_map(self):
        """The persistent GlobalTsdfMap, or None when build_global_map is off."""
        return self._global_map

    def finalize_global_map(self):
        """Rebuild the global map from the current (final) poses and return it.

        Call after any final pose-graph optimization so the saved map reflects the
        last optimized trajectory; a no-op (returns None) when the map is disabled.
        """
        if self._global_map is None:
            return None
        self._global_map.rebuild(list(self.graph.poses), self._scans)
        return self._global_map

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

    def _store_scan(self, pts: np.ndarray) -> None:
        """Append a preprocessed scan, mirroring it to the device when GPU-active."""
        self._scans.append(pts)
        if self._cuda_active:
            self._scans_dev.append(self._cp.asarray(pts, dtype=self._cp.float64))

    def _rebuild_local_map(self, count: int) -> None:
        """Rebuild the active TSDF from the most recent `map_window` historical scans.

        Builds a crisp local submap at the current pose estimates rather than a
        persistent global accumulation (which blurs as the robot moves).
        `count` is the number of stored scans to treat as history (excludes the
        scan currently being aligned).
        """
        mw = max(1, int(self.cfg.map_window))
        start = max(0, count - mw)
        if self._cuda_active:
            self._phi_d[:] = 0.0
            self._weight_d[:] = 0.0
            for i in range(start, count):
                self._cuda.update_tsdf_from_scan_cuda(
                    self._phi_d,
                    self._weight_d,
                    cfg=self.cfg.tsdf,
                    pose=self.graph.poses[i],
                    pts_d=self._scans_dev[i],
                    weight_inc=self.cfg.tsdf_weight_inc,
                    weight_max=self.cfg.tsdf_weight_max,
                )
            return
        self._tsdf.phi[:] = 0.0
        self._tsdf.weight[:] = 0.0
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
            self._store_scan(pts)
            self._stamps.append(scan.stamp_ns)
            if self._global_map is not None:
                self._global_map.integrate(init, pts)
            self._emit(0, init, score=0.0)
            return init

        prediction = self._predict()
        n_hist = len(self.graph.poses)
        # rebuild a crisp local map from recent history (excludes the new scan)
        self._rebuild_local_map(n_hist)

        n_pts = int(pts.shape[0])
        if n_hist < self.cfg.seed_scans:
            if self._cuda_active:
                # bootstrap aligns a single scan on the CPU; mirror the device map down
                self._tsdf.phi[:] = self._cp.asnumpy(self._phi_d)
                self._tsdf.weight[:] = self._cp.asnumpy(self._weight_d)
            res = align_scan_to_tsdf(
                tsdf=self._tsdf,
                scan_xy=pts,
                pose_init=prediction,
                max_iters=self.cfg.optimize_max_iters,
                huber_delta_m=self.cfg.huber_delta_m,
            )
            pose = self._gate(res.pose, prediction, res.num_inliers, n_pts)
            score = res.final_cost
            self._pending_marg = None
        else:
            k_hist = min(self.cfg.window_size - 1, n_hist)
            oldest_idx = n_hist - k_hist
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
            # Pin pose 0 of the window. With marginalization on and a prior already
            # built for this oldest pose, use that exact prior (no anchor); otherwise
            # fall back to the heuristic strong anchor (also used during window growth
            # before the first pose is ever dropped).
            use_marg = (
                self._marg_active and self._marg_prior is not None and self._marg_idx == oldest_idx
            )
            anchor = None if use_marg else AnchorPrior(
                pose=window_poses[0],
                info_xy=self.cfg.anchor_info_xy,
                info_theta=self.cfg.anchor_info_theta,
            )
            state = WindowState(
                poses=window_poses,
                scans=window_scans,
                motion_priors=motion_priors,
                anchor=anchor,
                marg_prior=self._marg_prior if use_marg else None,
            )
            if self._cuda_active:
                res = self._window_solver(
                    tsdf=self._tsdf,
                    state=state,
                    max_iters=self.cfg.optimize_max_iters,
                    huber_delta_m=self.cfg.huber_delta_m,
                    phi_dev=self._phi_d,
                    weight_dev=self._weight_d,
                )
            else:
                res = self._window_solver(
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
            # remember what is needed to marginalize this window's oldest pose once the
            # new pose is committed (computed in the common tail below)
            self._pending_marg = (
                (oldest_idx, motion_priors[0], anchor, self._marg_prior if use_marg else None)
                if self._marg_active
                else None
            )

        prev = self.graph.poses[-1]
        rel = prev.inverse().compose(pose)
        self._last_rel = rel
        node = self.graph.add_pose(pose)
        self.graph.add_edge(Edge(i=node - 1, j=node, rel=rel))
        self._store_scan(pts)
        self._stamps.append(scan.stamp_ns)
        if self._marg_active:
            self._update_marginalization()
        if self._global_map is not None:
            self._global_map.integrate(pose, pts)
        self._emit(node, pose, score=score)

        if (
            self.cfg.loop_closure_enabled
            and node % max(1, self.cfg.loop_detect_every_n) == 0
        ):
            self._try_loop_closure(node, pts)

        return self.graph.poses[node]

    def _update_marginalization(self) -> None:
        """Once the new pose is committed, Schur-eliminate the pose that just left the
        window into a MarginalizationPrior on the new oldest pose (P4.1).

        Only fires when the window actually slid (a pose dropped); during window growth
        the oldest pose stays put and the prior is left untouched. The dropped pose's
        data term is linearized against the current local map (first-estimate Jacobians);
        the prior recurses on whatever pinned pose 0 this step (anchor or prior marginal).
        """
        pending = getattr(self, "_pending_marg", None)
        if pending is None:
            return
        oldest_idx, motion_prior0, anchor_used, prev_marg_used = pending
        n = len(self.graph.poses)
        next_k = min(self.cfg.window_size - 1, n)
        next_oldest = n - next_k
        if next_oldest != oldest_idx + 1:
            return  # window still growing -- nothing dropped yet
        from slamx.core.scan_ba.marginalize import marginalize_oldest_pose

        self._marg_prior = marginalize_oldest_pose(
            tsdf=self._tsdf,
            poses=[self.graph.poses[oldest_idx], self.graph.poses[oldest_idx + 1]],
            scans=[self._scans[oldest_idx], self._scans[oldest_idx + 1]],
            motion_prior=motion_prior0,
            huber_delta_m=self.cfg.huber_delta_m,
            anchor=anchor_used,
            prev_marg=prev_marg_used,
        )
        self._marg_idx = oldest_idx + 1

    def _try_loop_closure(self, node: int, pts: np.ndarray) -> None:
        """Detect distance-based loops against past nodes, verify by TSDF
        alignment, add accepted loop edges, then run a global pose-graph solve."""
        cur = self.graph.poses[node]
        # spatial candidates with a time gap
        cands: list[tuple[float, int]] = []
        for j in range(0, node - self.cfg.loop_min_gap):
            pj = self.graph.poses[j]
            d = float(np.hypot(cur.x - pj.x, cur.y - pj.y))
            if d < self.cfg.loop_dist_m:
                cands.append((d, j))
        if not cands:
            return
        cands.sort(key=lambda c: c[0])

        n_pts = int(pts.shape[0])
        added = False
        for _, j in cands[: self.cfg.loop_max_candidates]:
            if (j, node) in self._loop_edges:
                continue
            verify_tsdf = self._build_submap_tsdf(j)
            res = align_scan_to_tsdf(
                tsdf=verify_tsdf,
                scan_xy=pts,
                pose_init=cur,
                max_iters=self.cfg.optimize_max_iters,
                huber_delta_m=self.cfg.huber_delta_m,
            )
            inl_ratio = (res.num_inliers / n_pts) if n_pts else 0.0
            rms = float(np.sqrt(2.0 * res.final_cost / max(1, res.num_inliers)))
            corr = float(np.hypot(res.pose.x - cur.x, res.pose.y - cur.y))
            if (
                inl_ratio >= self.cfg.loop_accept_inlier_ratio
                and rms <= self.cfg.loop_accept_rms_m
                and corr <= self.cfg.loop_max_correction_m
            ):
                rel = self.graph.poses[j].inverse().compose(res.pose)
                self.graph.add_edge(Edge(i=j, j=node, rel=rel))
                self._loop_edges.add((j, node))
                added = True
                if self.telemetry:
                    self.telemetry.emit(
                        "loop_closure_accepted",
                        {"node": node, "i": j, "inlier_ratio": inl_ratio, "rms": rms},
                    )

        if added:
            opt = self.graph.optimize()
            if self.telemetry:
                self.telemetry.emit("optimization", {"node": node, **opt})
            self._last_rel = self.graph.poses[-2].inverse().compose(self.graph.poses[-1])
            # the trajectory just moved: rebuild the persistent global map from the
            # corrected poses so it does not bake in the pre-closure drift.
            if self._global_map is not None:
                self._global_map.rebuild(list(self.graph.poses), self._scans)

    def _build_submap_tsdf(self, center: int) -> Tsdf2D:
        """Build a verification TSDF from scans near node `center` at their poses."""
        tsdf = Tsdf2D.zeros(self.cfg.tsdf)
        w = max(1, self.cfg.loop_submap_window)
        lo = max(0, center - w)
        hi = min(len(self._scans), center + w + 1)
        for i in range(lo, hi):
            update_tsdf_from_scan(
                tsdf,
                pose_map=self.graph.poses[i],
                points_sensor=self._scans[i],
                weight_inc=self.cfg.tsdf_weight_inc,
                weight_max=self.cfg.tsdf_weight_max,
            )
        return tsdf

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
