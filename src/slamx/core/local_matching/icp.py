from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from slamx.core.types import LaserScan, MatchResult, Pose2, transform_points_xy


@dataclass
class IcpConfig:
    max_iterations: int = 20
    max_correspondence_dist_m: float = 0.5
    min_correspondences: int = 30
    trim_fraction: float = 0.2  # drop worst residuals


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _best_fit_se2(src: np.ndarray, dst: np.ndarray) -> Pose2:
    """Compute SE2 (x,y,theta) that maps src -> dst in least squares."""
    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = X.T @ Y
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    t = mu_d - (R @ mu_s)
    return Pose2(float(t[0]), float(t[1]), theta)


class IcpScanMatcher:
    """Point-to-point ICP against reference points in map frame.

    Deterministic: no randomness; KDTree queries are deterministic for fixed inputs.
    """

    def __init__(self, cfg: IcpConfig | None = None) -> None:
        self.cfg = cfg or IcpConfig()

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        pts_s = scan.points_xy()
        if pts_s.size == 0:
            return MatchResult(
                pose_map=prediction_map,
                score=float("-inf"),
                candidates=[],
                diagnostics={"reason": "empty scan"},
            )
        if ref_points_xy_map.size == 0:
            return MatchResult(
                pose_map=prediction_map,
                score=0.0,
                candidates=[(prediction_map.x, prediction_map.y, prediction_map.theta, 0.0)],
                diagnostics={"reason": "no reference"},
            )

        ref = np.asarray(ref_points_xy_map, dtype=np.float64).reshape(-1, 2)
        tree = cKDTree(ref)

        cur = prediction_map
        best_score = float("-inf")
        best_pose = cur
        last_rms = None

        for it in range(int(self.cfg.max_iterations)):
            pts_m = transform_points_xy(cur.as_se2(), pts_s)
            d, idx = tree.query(pts_m, k=1, workers=1)
            d = np.asarray(d, dtype=np.float64)
            idx = np.asarray(idx, dtype=np.int64)
            m = np.isfinite(d) & (d <= float(self.cfg.max_correspondence_dist_m))
            if int(np.sum(m)) < int(self.cfg.min_correspondences):
                break

            src = pts_m[m]
            dst = ref[idx[m]]

            # trimmed ICP
            if self.cfg.trim_fraction > 0.0:
                order = np.argsort(d[m])
                keep = int(max(1, np.floor((1.0 - self.cfg.trim_fraction) * order.size)))
                sel = order[:keep]
                src = src[sel]
                dst = dst[sel]

            delta = _best_fit_se2(src, dst)
            cur = cur.compose(delta)

            err = dst - transform_points_xy(delta.as_se2(), src)  # approx
            rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if err.size else 0.0

            score = float(-rms)
            if score > best_score:
                best_score = score
                best_pose = cur
            if last_rms is not None and abs(last_rms - rms) < 1e-4:
                last_rms = rms
                break
            last_rms = rms

        # Provide empty candidate list but keep schema happy in telemetry
        return MatchResult(
            pose_map=best_pose,
            score=float(best_score),
            candidates=[(best_pose.x, best_pose.y, best_pose.theta, float(best_score))],
            diagnostics={
                "icp": {
                    "max_iterations": int(self.cfg.max_iterations),
                    "max_correspondence_dist_m": float(self.cfg.max_correspondence_dist_m),
                    "min_correspondences": int(self.cfg.min_correspondences),
                    "final_rms": float(last_rms) if last_rms is not None else None,
                }
            },
        )

