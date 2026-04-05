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
    max_iterations: int = 50


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

        def residuals(uv: np.ndarray) -> np.ndarray:
            P = uv.reshape(-1, 3)
            res: list[float] = []
            for e in self.edges:
                pi, pj = P[e.i], P[e.j]
                Ti = Pose2(float(pi[0]), float(pi[1]), float(pi[2]))
                Tj = Pose2(float(pj[0]), float(pj[1]), float(pj[2]))
                pred = Ti.inverse().compose(Tj)
                res.append(pred.x - e.rel.x)
                res.append(pred.y - e.rel.y)
                res.append(_wrap_pi(pred.theta - e.rel.theta))
            return np.asarray(res, dtype=np.float64)

        r0 = residuals(x0)
        rms0 = float(np.sqrt(np.mean(r0 * r0))) if r0.size else 0.0
        maxabs0 = float(np.max(np.abs(r0))) if r0.size else 0.0

        r = least_squares(
            residuals,
            x0,
            method="trf",
            max_nfev=self.cfg.max_iterations * max(1, len(self.edges)) * 3,
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
