from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from slamx.core.local_matching.protocol import ScanMatcher
from slamx.core.types import LaserScan, Pose2


@dataclass(frozen=True)
class LoopClosureResult:
    i: int
    j: int
    score: float
    accepted: bool
    rel_ij: Pose2 | None  # pose of j in i frame (i.inverse().compose(j))
    diagnostics: dict


@dataclass
class HeuristicLoopConfig:
    enabled: bool = False
    search_radius_m: float = 1.5
    min_separation_nodes: int = 30
    max_candidates: int = 3
    accept_score: float = -0.25  # matcher score threshold (higher is better)
    icp_accept_rms: float = 0.15  # ICP RMS threshold for acceptance (when refiner present)


class HeuristicLoopDetector:
    """Nearest-neighbor in pose space + re-matching against stored reference points."""

    def __init__(self, cfg: HeuristicLoopConfig | None = None) -> None:
        self.cfg = cfg or HeuristicLoopConfig()

    def detect_and_match(
        self,
        *,
        matcher: ScanMatcher,
        refiner: ScanMatcher | None = None,
        node_id: int,
        pose_map: Pose2,
        scan: LaserScan,
        poses: list[Pose2],
        ref_points_by_node: list[np.ndarray],
    ) -> list[LoopClosureResult]:
        if not self.cfg.enabled:
            return []
        if node_id <= 0 or len(poses) < 2:
            return []
        if len(ref_points_by_node) != len(poses):
            return []

        # Build NN index excluding recent nodes
        min_j = max(0, node_id - int(self.cfg.min_separation_nodes))
        if min_j <= 0:
            return []
        P = np.array([[p.x, p.y] for p in poses[:min_j]], dtype=np.float64)
        if P.shape[0] == 0:
            return []
        tree = cKDTree(P)
        q = np.array([pose_map.x, pose_map.y], dtype=np.float64)
        idxs = tree.query_ball_point(q, r=float(self.cfg.search_radius_m))
        if not idxs:
            return []

        # deterministic: sort by distance then index
        dists = [(float(np.hypot(*(P[i] - q))), int(i)) for i in idxs]
        dists.sort(key=lambda t: (t[0], t[1]))
        cand_js = [j for _d, j in dists[: int(self.cfg.max_candidates)]]

        out: list[LoopClosureResult] = []
        for j in cand_js:
            ref = ref_points_by_node[j]
            if ref.size == 0:
                continue
            mr = matcher.match(scan=scan, prediction_map=pose_map, ref_points_xy_map=ref)

            final_pose = mr.pose_map
            final_score = float(mr.score)
            diag: dict = {"ref_points": int(ref.shape[0]), "matcher": mr.diagnostics}

            if refiner is not None and final_score >= float(self.cfg.accept_score):
                # ICP refinement using correlative result as initial guess
                icp_mr = refiner.match(scan=scan, prediction_map=mr.pose_map, ref_points_xy_map=ref)
                icp_rms = icp_mr.diagnostics.get("icp", {}).get("final_rms")
                diag["icp"] = icp_mr.diagnostics
                final_pose = icp_mr.pose_map
                final_score = float(icp_mr.score)
                accepted = icp_rms is not None and icp_rms <= float(self.cfg.icp_accept_rms)
            else:
                accepted = refiner is None and bool(final_score >= float(self.cfg.accept_score))

            rel = poses[j].inverse().compose(final_pose)
            out.append(
                LoopClosureResult(
                    i=j,
                    j=node_id,
                    score=final_score,
                    accepted=accepted,
                    rel_ij=rel if accepted else None,
                    diagnostics=diag,
                )
            )
        return out

