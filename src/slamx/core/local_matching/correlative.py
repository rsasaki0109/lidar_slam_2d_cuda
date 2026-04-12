from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from slamx.core.local_matching.range_weights import compute_range_weights
from slamx.core.types import LaserScan, MatchResult, Pose2, transform_points_xy


@dataclass
class CorrelativeGridConfig:
    linear_step_m: float = 0.05
    angular_step_deg: float = 2.0
    linear_window_m: float = 0.5
    angular_window_deg: float = 30.0
    sigma_hit_m: float = 0.15  # Gaussian on ref points for scoring
    range_weight_mode: str = "none"  # "none", "linear", "sigmoid"
    range_weight_min_m: float = 1.0  # below this range, weight is reduced


class CorrelativeScanMatcher:
    """Coarse grid search + Gaussian hit-cost against reference points (map frame).

    Not Cartographer CSM: intentionally small, deterministic, and easy to instrument.
    """

    def __init__(self, cfg: CorrelativeGridConfig | None = None) -> None:
        self.cfg = cfg or CorrelativeGridConfig()

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
                candidates=[
                    (prediction_map.x, prediction_map.y, prediction_map.theta, 0.0)
                ],
                diagnostics={"reason": "no reference"},
            )

        ref = np.asarray(ref_points_xy_map, dtype=np.float64).reshape(-1, 2)
        tree = cKDTree(ref)
        sig = max(self.cfg.sigma_hit_m, 1e-6)
        inv_2sig2 = 1.0 / (2.0 * sig * sig)

        scan_ranges = np.linalg.norm(pts_s, axis=1)
        rw = compute_range_weights(scan_ranges, self.cfg.range_weight_mode, self.cfg.range_weight_min_m)

        lin = np.arange(
            -self.cfg.linear_window_m,
            self.cfg.linear_window_m + 1e-9,
            self.cfg.linear_step_m,
        )
        ang = np.deg2rad(
            np.arange(
                -self.cfg.angular_window_deg,
                self.cfg.angular_window_deg + 1e-9,
                self.cfg.angular_step_deg,
            )
        )

        # Pre-compute translation offsets grid: (N_lin^2, 2)
        dxs, dys = np.meshgrid(lin, lin)
        offsets = np.column_stack([dxs.ravel(), dys.ravel()])  # (N_lin^2, 2)
        n_trans = offsets.shape[0]
        n_pts = pts_s.shape[0]

        px, py = prediction_map.x, prediction_map.y
        base_theta = prediction_map.theta

        all_scores: list[np.ndarray] = []
        all_thetas: list[np.ndarray] = []

        for dth in ang:
            theta = base_theta + float(dth)
            c, s = np.cos(theta), np.sin(theta)
            # Rotate scan points once per angle
            pts_rot = pts_s @ np.array([[c, -s], [s, c]], dtype=np.float64).T  # (N_pts, 2)
            # Translate by all offsets: (N_trans, N_pts, 2)
            pts_batch = pts_rot[None, :, :] + (offsets + np.array([px, py]))[: , None, :]
            pts_flat = pts_batch.reshape(-1, 2)  # (N_trans * N_pts, 2)
            nn_dist, _ = tree.query(pts_flat, k=1)
            nn_d2 = (nn_dist * nn_dist).reshape(n_trans, n_pts)
            if rw is not None:
                scores = np.average(-nn_d2 * inv_2sig2, weights=rw, axis=1)  # (N_trans,)
            else:
                scores = np.mean(-nn_d2 * inv_2sig2, axis=1)  # (N_trans,)
            all_scores.append(scores)
            all_thetas.append(np.full(n_trans, theta))

        all_scores_arr = np.concatenate(all_scores)   # (N_total,)
        all_thetas_arr = np.concatenate(all_thetas)    # (N_total,)
        all_xy = np.tile(offsets + np.array([px, py]), (len(ang), 1))  # (N_total, 2)

        best_idx = int(np.argmax(all_scores_arr))
        best_score = float(all_scores_arr[best_idx])
        best_pose = Pose2(float(all_xy[best_idx, 0]), float(all_xy[best_idx, 1]),
                          float(all_thetas_arr[best_idx]))

        # Top-k candidates for telemetry
        top_k_n = min(50, len(all_scores_arr))
        top_idxs = np.argpartition(all_scores_arr, -top_k_n)[-top_k_n:]
        top_idxs = top_idxs[np.argsort(all_scores_arr[top_idxs])[::-1]]
        top_k = [
            (float(all_xy[i, 0]), float(all_xy[i, 1]),
             float(all_thetas_arr[i]), float(all_scores_arr[i]))
            for i in top_idxs
        ]

        return MatchResult(
            pose_map=best_pose,
            score=best_score,
            candidates=top_k,
            diagnostics={"grid": {"lin": lin.size, "ang": ang.size}},
        )
