from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from slamx.core.types import Pose2


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class Edge:
    i: int
    j: int
    # relative pose rel such that T_j ≈ T_i.compose(rel)
    rel: Pose2


@dataclass
class PoseGraphConfig:
    # Budget multiplier: max_nfev ≈ max_iterations * n_edges * 3 (scipy least_squares, trf).
    # Without a cap this grows ~O(n^2) as odometry chains lengthen, stalling long replays.
    max_iterations: int = 50
    max_nfev_cap: int | None = None


@dataclass
class PoseGraph:
    poses: list[Pose2] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    cfg: PoseGraphConfig = field(default_factory=PoseGraphConfig)

    def add_pose(self, p: Pose2) -> int:
        self.poses.append(p)
        return len(self.poses) - 1

    def add_edge(self, e: Edge) -> None:
        self.edges.append(e)

    def optimize(self) -> dict:
        if len(self.poses) <= 1:
            return {"success": True, "cost": 0.0, "message": "single pose"}

        x0 = np.array(
            [[p.x, p.y, p.theta] for p in self.poses],
            dtype=np.float64,
        ).reshape(-1)

        # Pre-extract edge data for vectorised residuals
        ei = np.array([e.i for e in self.edges], dtype=np.intp)
        ej = np.array([e.j for e in self.edges], dtype=np.intp)
        rel = np.array([[e.rel.x, e.rel.y, e.rel.theta] for e in self.edges], dtype=np.float64)

        def residuals(uv: np.ndarray) -> np.ndarray:
            P = uv.reshape(-1, 3)
            pi = P[ei]  # (E, 3)
            pj = P[ej]  # (E, 3)
            ci = np.cos(pi[:, 2])
            si = np.sin(pi[:, 2])
            dx = pj[:, 0] - pi[:, 0]
            dy = pj[:, 1] - pi[:, 1]
            pred_x = ci * dx + si * dy
            pred_y = -si * dx + ci * dy
            pred_th = pj[:, 2] - pi[:, 2]
            res = np.empty((len(self.edges), 3), dtype=np.float64)
            res[:, 0] = pred_x - rel[:, 0]
            res[:, 1] = pred_y - rel[:, 1]
            res[:, 2] = np.arctan2(np.sin(pred_th - rel[:, 2]), np.cos(pred_th - rel[:, 2]))
            return res.reshape(-1)

        r0 = residuals(x0)
        rms0 = float(np.sqrt(np.mean(r0 * r0))) if r0.size else 0.0
        maxabs0 = float(np.max(np.abs(r0))) if r0.size else 0.0

        n_edges = max(1, len(self.edges))
        max_nfev = int(self.cfg.max_iterations) * n_edges * 3
        if self.cfg.max_nfev_cap is not None:
            max_nfev = min(max_nfev, int(self.cfg.max_nfev_cap))
        max_nfev = max(max_nfev, 32)

        r = least_squares(
            residuals,
            x0,
            method="trf",
            max_nfev=max_nfev,
        )
        xf = r.x.reshape(-1, 3)
        self.poses = [Pose2(float(row[0]), float(row[1]), float(row[2])) for row in xf]

        rf = residuals(r.x)
        rmsf = float(np.sqrt(np.mean(rf * rf))) if rf.size else 0.0
        maxabsf = float(np.max(np.abs(rf))) if rf.size else 0.0
        return {
            "success": bool(r.success),
            "cost": float(r.cost),
            "residual_rms_before": rms0,
            "residual_rms_after": rmsf,
            "residual_maxabs_before": maxabs0,
            "residual_maxabs_after": maxabsf,
            "message": str(r.message),
        }
