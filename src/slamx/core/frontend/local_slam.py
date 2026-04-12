from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.backend.pose_graph import Edge, PoseGraph, PoseGraphConfig
from slamx.core.local_matching.correlative import CorrelativeGridConfig, CorrelativeScanMatcher
from slamx.core.local_matching.hybrid import (
    HybridFallbackConfig,
    HybridRefinementConfig,
    HybridScanMatcher,
)
from slamx.core.local_matching.icp import IcpConfig, IcpScanMatcher
from slamx.core.local_matching.protocol import ScanMatcher
from slamx.core.loop_detection.heuristic import HeuristicLoopConfig, HeuristicLoopDetector
from slamx.core.loop_detection.null_loop import LoopDetector, NullLoopDetector
from slamx.core.observability import JsonlTelemetry
from slamx.core.preprocess.pipeline import PreprocessConfig, preprocess_scan
from slamx.core.submap.builder import SubmapBuilder, SubmapBuilderConfig
from slamx.core.types import LaserScan, Pose2, Submap, transform_points_xy


@dataclass
class LocalSlamConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    matcher_type: str = "correlative"
    correlative: CorrelativeGridConfig = field(default_factory=CorrelativeGridConfig)
    icp: IcpConfig = field(default_factory=IcpConfig)
    hybrid_refinement: HybridRefinementConfig = field(default_factory=HybridRefinementConfig)
    hybrid_fallback: HybridFallbackConfig = field(default_factory=HybridFallbackConfig)
    prediction_mode: str = "hold"
    prediction_gain: float = 1.0
    submap: SubmapBuilderConfig = field(default_factory=SubmapBuilderConfig)
    optimize_every_n_keyframes: int = 10
    pose_graph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    # When node >= optimize_adaptive_from_node, spacing between graph solves is at least
    # optimize_min_interval_for_long_runs (keeps long offline replays tractable).
    optimize_adaptive_from_node: int | None = None
    optimize_min_interval_for_long_runs: int = 200
    pose_graph_skip_optimization_from_node: int | None = None
    loop: HeuristicLoopConfig = field(default_factory=HeuristicLoopConfig)
    loop_ref_submap_scans: int = 10  # accumulate this many scans for loop closure reference
    loop_icp: IcpConfig = field(default_factory=lambda: IcpConfig(
        max_iterations=30,
        max_correspondence_dist_m=2.0,
        min_correspondences=20,
        trim_fraction=0.3,
    ))
    loop_correlative: CorrelativeGridConfig = field(default_factory=lambda: CorrelativeGridConfig(
        linear_step_m=0.05,
        angular_step_deg=2.0,
        linear_window_m=0.5,
        angular_window_deg=30.0,
        sigma_hit_m=0.25,
    ))


@dataclass
class LocalSlamEngine:
    cfg: LocalSlamConfig = field(default_factory=LocalSlamConfig)
    telemetry: JsonlTelemetry | None = None
    loop: LoopDetector = field(default_factory=NullLoopDetector)

    _matcher: ScanMatcher | None = field(default=None, repr=False)
    _submaps: SubmapBuilder | None = field(default=None, repr=False)
    graph: PoseGraph = field(init=False)

    _stamps: list[int | None] = field(default_factory=list)
    _scan_window: list[tuple[Pose2, LaserScan]] = field(default_factory=list)
    _last_pose: Pose2 | None = field(default=None, init=False)
    _last_rel: Pose2 | None = field(default=None, init=False)
    _ref_points_by_node: list[np.ndarray] = field(default_factory=list, repr=False)
    _loop_matcher: ScanMatcher | None = field(default=None, repr=False)
    _loop_refiner: ScanMatcher | None = field(default=None, repr=False)
    _heuristic_loop: HeuristicLoopDetector | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.graph = PoseGraph(cfg=self.cfg.pose_graph)
        mt = (self.cfg.matcher_type or "correlative").lower()
        if mt in {"correlative", "grid", "csm"}:
            self._matcher = CorrelativeScanMatcher(self.cfg.correlative)
        elif mt in {"hybrid", "correlative_icp", "icp_refined"}:
            self._matcher = HybridScanMatcher(
                self.cfg.correlative,
                self.cfg.icp,
                refinement=self.cfg.hybrid_refinement,
                fallback=self.cfg.hybrid_fallback,
            )
        elif mt in {"icp"}:
            self._matcher = IcpScanMatcher(self.cfg.icp)
        else:
            raise ValueError(f"Unknown matcher_type: {self.cfg.matcher_type}")
        self._loop_matcher = CorrelativeScanMatcher(self.cfg.loop_correlative)
        self._loop_refiner = IcpScanMatcher(self.cfg.loop_icp)
        self._submaps = SubmapBuilder(self.cfg.submap)
        self._heuristic_loop = HeuristicLoopDetector(self.cfg.loop)

    @property
    def poses(self) -> list[Pose2]:
        return list(self.graph.poses)

    @property
    def stamps_ns(self) -> list[int | None]:
        return list(self._stamps)

    def handle_scan(self, scan: LaserScan) -> Pose2:
        assert self._matcher is not None and self._submaps is not None
        scan_p = preprocess_scan(scan, self.cfg.preprocess)

        if self._last_pose is None:
            init = Pose2(0.0, 0.0, 0.0)
            node = self.graph.add_pose(init)
            self._stamps.append(scan.stamp_ns)
            self._last_pose = init
            self._last_rel = None
            self._scan_window.append((init, scan_p))
            self._ref_points_by_node.append(np.zeros((0, 2), dtype=np.float64))
            self._emit_keyframe(node, init, scan_p, prediction=init, match_score=0.0, jump=0.0)
            return init

        prediction = self._predict_pose()
        ref = self._reference_points_map()

        mr = self._matcher.match(
            scan=scan_p,
            prediction_map=prediction,
            ref_points_xy_map=ref,
        )
        accepted = mr.pose_map

        jump = float(
            np.hypot(accepted.x - prediction.x, accepted.y - prediction.y)
            + abs(
                np.arctan2(
                    np.sin(accepted.theta - prediction.theta),
                    np.cos(accepted.theta - prediction.theta),
                )
            )
        )

        prev = self._last_pose
        rel = prev.inverse().compose(accepted)

        node = self.graph.add_pose(accepted)
        if node > 0:
            self.graph.add_edge(Edge(i=node - 1, j=node, rel=rel))

        self._stamps.append(scan.stamp_ns)
        self._last_pose = accepted
        self._last_rel = rel
        self._scan_window.append((accepted, scan_p))
        # store reference points for loop closure: local submap from recent scans
        stride = max(1, int(self.cfg.submap.downsample_stride))
        n_sub = max(1, self.cfg.loop_ref_submap_scans)
        acc_ref: list[np.ndarray] = []
        for pose, sc in self._scan_window[-n_sub:]:
            pts = sc.points_xy()
            if pts.size:
                acc_ref.append(transform_points_xy(pose.as_se2(), pts)[::stride])
        if acc_ref:
            self._ref_points_by_node.append(np.concatenate(acc_ref, axis=0))
        else:
            self._ref_points_by_node.append(np.zeros((0, 2), dtype=np.float64))

        if len(self._scan_window) > self.cfg.submap.max_submap_scans:
            self._scan_window.pop(0)

        self._emit_keyframe(
            node, accepted, scan_p, prediction=prediction, match_score=mr.score, jump=jump
        )
        self._emit_match_detail(node, mr, top_candidates=mr.candidates[:10])

        if self._heuristic_loop is not None and self._loop_matcher is not None:
            lrs = self._heuristic_loop.detect_and_match(
                matcher=self._loop_matcher,
                refiner=self._loop_refiner,
                node_id=node,
                pose_map=accepted,
                scan=scan_p,
                poses=self.graph.poses,
                ref_points_by_node=self._ref_points_by_node,
            )
            if lrs and self.telemetry:
                self.telemetry.emit(
                    "loop_closure_candidates",
                    {
                        "node": node,
                        "candidates": [
                            {
                                "i": r.i,
                                "j": r.j,
                                "score": r.score,
                                "accepted": r.accepted,
                                "rel_ij": (
                                    {"x": r.rel_ij.x, "y": r.rel_ij.y, "theta": r.rel_ij.theta}
                                    if r.rel_ij is not None
                                    else None
                                ),
                                "diagnostics": r.diagnostics,
                            }
                            for r in lrs
                        ],
                    },
                )
            for r in lrs:
                if r.accepted and r.rel_ij is not None:
                    # add loop closure edge
                    self.graph.add_edge(Edge(i=r.i, j=r.j, rel=r.rel_ij))
                    if self.telemetry:
                        self.telemetry.emit(
                            "loop_closure_accepted",
                            {
                                "node": node,
                                "i": r.i,
                                "j": r.j,
                                "score": r.score,
                                "rel_ij": {"x": r.rel_ij.x, "y": r.rel_ij.y, "theta": r.rel_ij.theta},
                            },
                        )
                elif self.telemetry:
                    self.telemetry.emit(
                        "loop_closure_rejected",
                        {"node": node, "i": r.i, "j": r.j, "score": r.score},
                    )

        opt_every = self.cfg.optimize_every_n_keyframes
        if (
            self.cfg.optimize_adaptive_from_node is not None
            and opt_every > 0
            and node >= int(self.cfg.optimize_adaptive_from_node)
        ):
            opt_every = max(opt_every, int(self.cfg.optimize_min_interval_for_long_runs))

        if opt_every > 0 and (node + 1) % opt_every == 0:
            skip_global = (
                self.cfg.pose_graph_skip_optimization_from_node is not None
                and node >= int(self.cfg.pose_graph_skip_optimization_from_node)
            )
            if not skip_global:
                opt = self.graph.optimize()
                if self.telemetry:
                    self.telemetry.emit("optimization", {"node": node, **opt})
                self._last_pose = self.graph.poses[-1]
                if len(self.graph.poses) >= 2:
                    self._last_rel = self.graph.poses[-2].inverse().compose(self.graph.poses[-1])

        return accepted

    def _predict_pose(self) -> Pose2:
        assert self._last_pose is not None
        mode = (self.cfg.prediction_mode or "hold").lower()
        if mode in {"hold", "last_pose", "zero_velocity"} or self._last_rel is None:
            return self._last_pose
        if mode not in {"constant_velocity", "const_vel", "cv"}:
            raise ValueError(f"Unknown prediction_mode: {self.cfg.prediction_mode}")
        gain = float(self.cfg.prediction_gain)
        if not np.isfinite(gain) or abs(gain) < 1e-12:
            return self._last_pose
        return self._last_pose.compose(
            Pose2(
                float(self._last_rel.x * gain),
                float(self._last_rel.y * gain),
                float(self._last_rel.theta * gain),
            )
        )

    def _reference_points_map(self) -> np.ndarray:
        if not self._scan_window:
            return np.zeros((0, 2), dtype=np.float64)
        acc: list[np.ndarray] = []
        for pose, scan in self._scan_window:
            pts = scan.points_xy()
            if pts.size == 0:
                continue
            acc.append(transform_points_xy(pose.as_se2(), pts))
        if not acc:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(acc, axis=0)

    def finalize_submap(self, node_id: int, origin: Pose2) -> Submap:
        assert self._submaps is not None
        return self._submaps.build(submap_node_id=node_id, origin=origin, scans_map=list(self._scan_window))

    def _emit_keyframe(
        self,
        node: int,
        pose: Pose2,
        scan: LaserScan,
        *,
        prediction: Pose2,
        match_score: float,
        jump: float,
    ) -> None:
        if not self.telemetry:
            return
        n_pts = int(scan.points_xy().shape[0])
        self.telemetry.emit(
            "keyframe",
            {
                "node": node,
                "stamp_ns": scan.stamp_ns,
                "pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
                "prediction": {"x": prediction.x, "y": prediction.y, "theta": prediction.theta},
                "scan_match_score": match_score,
                "pose_jump": jump,
                "submap_entropy_proxy": float(n_pts),
            },
        )

    def _emit_match_detail(self, node: int, mr, *, top_candidates: list) -> None:
        if not self.telemetry:
            return
        self.telemetry.emit(
            "scan_match_candidates",
            {"node": node, "best_score": mr.score, "top": top_candidates, "diagnostics": mr.diagnostics},
        )
