from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from slamx.core.local_matching.range_weights import compute_range_weights
from slamx.core.types import LaserScan, MatchResult, Pose2, transform_points_xy


@dataclass
class IcpConfig:
    max_iterations: int = 20
    max_correspondence_dist_m: float = 0.5
    min_correspondences: int = 30
    trim_fraction: float = 0.2  # drop worst residuals
    range_weight_mode: str = "none"  # "none", "linear", "sigmoid"
    range_weight_min_m: float = 1.0  # below this range, weight is reduced
    icp_mode: str = "point"  # "point" (point-to-point) or "line" (point-to-line)
    normal_k: int = 5  # number of neighbors for normal estimation


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _best_fit_se2(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> Pose2:
    """Compute SE2 (x,y,theta) that maps src -> dst in (weighted) least squares."""
    if weights is not None:
        w = weights[:, None]  # (N, 1)
        w_sum = float(np.sum(weights))
        mu_s = np.sum(w * src, axis=0) / w_sum
        mu_d = np.sum(w * dst, axis=0) / w_sum
        X = src - mu_s
        Y = dst - mu_d
        H = (w * X).T @ Y
    else:
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


def _estimate_normals_2d(ref_points: np.ndarray, tree: cKDTree, k: int = 5) -> np.ndarray:
    """Estimate 2D normals at each reference point using local PCA.

    For each point, find k nearest neighbors, compute the covariance matrix,
    and take the eigenvector corresponding to the smallest eigenvalue as the normal.

    Returns: (N, 2) array of unit normals.
    """
    n = ref_points.shape[0]
    normals = np.zeros((n, 2), dtype=np.float64)
    # Use min(k, n) to handle small point sets
    actual_k = min(k, n)
    if actual_k < 2:
        # Cannot estimate normals with fewer than 2 points; return zeros
        return normals
    _, idxs = tree.query(ref_points, k=actual_k, workers=1)
    idxs = np.asarray(idxs, dtype=np.int64)
    if idxs.ndim == 1:
        idxs = idxs[:, None]
    for i in range(n):
        neighbors = ref_points[idxs[i]]
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Smallest eigenvalue eigenvector is the normal
        normals[i] = eigvecs[:, 0]
    # Normalize (should already be unit, but be safe)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms
    return normals


def _best_fit_se2_point_to_line(
    src: np.ndarray,
    dst: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray | None = None,
) -> Pose2:
    """Solve point-to-line ICP step via linearized least squares.

    Minimize sum of w_i * (n_i . (R*s_i + t - d_i))^2

    Linearize around current estimate (identity delta):
    For small delta = (dx, dy, dtheta):
      R*s_i + t ~ s_i + [dx, dy] + dtheta * [-s_iy, s_ix]

    So residual_i = n_ix*dx + n_iy*dy + (n_i . [-s_iy, s_ix])*dtheta - n_i . (d_i - s_i)

    This gives a 3x3 linear system (A^T W A) x = A^T W b.
    """
    n = src.shape[0]
    # Build the linear system
    nx = normals[:, 0]
    ny = normals[:, 1]
    sx = src[:, 0]
    sy = src[:, 1]
    # n_i . [-s_iy, s_ix] = nx*(-sy) + ny*sx
    cross = -nx * sy + ny * sx

    # A is (N, 3): columns are [nx, ny, cross]
    A = np.column_stack([nx, ny, cross])
    # b is (N,): n_i . (d_i - s_i)
    diff = dst - src
    b = nx * diff[:, 0] + ny * diff[:, 1]

    if weights is not None:
        w_sqrt = np.sqrt(weights)
        A = A * w_sqrt[:, None]
        b = b * w_sqrt

    # Solve via least squares
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    dx, dy, dtheta = float(result[0]), float(result[1]), float(result[2])
    return Pose2(dx, dy, _wrap_pi(dtheta))


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

        scan_ranges = np.linalg.norm(pts_s, axis=1)
        all_weights = compute_range_weights(scan_ranges, self.cfg.range_weight_mode, self.cfg.range_weight_min_m)

        use_p2l = self.cfg.icp_mode == "line"
        ref_normals: np.ndarray | None = None
        if use_p2l:
            ref_normals = _estimate_normals_2d(ref, tree, k=self.cfg.normal_k)
            # Check if normals are valid (all zeros means degenerate)
            if np.all(np.abs(ref_normals) < 1e-12):
                use_p2l = False  # fall back to point-to-point

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
            w = all_weights[m] if all_weights is not None else None
            matched_idx = idx[m]

            # trimmed ICP
            if self.cfg.trim_fraction > 0.0:
                order = np.argsort(d[m])
                keep = int(max(1, np.floor((1.0 - self.cfg.trim_fraction) * order.size)))
                sel = order[:keep]
                src = src[sel]
                dst = dst[sel]
                w = w[sel] if w is not None else None
                matched_idx = matched_idx[sel]

            if use_p2l:
                assert ref_normals is not None
                nrm = ref_normals[matched_idx]
                delta = _best_fit_se2_point_to_line(src, dst, nrm, weights=w)
            else:
                delta = _best_fit_se2(src, dst, weights=w)
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

