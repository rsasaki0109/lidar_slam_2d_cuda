from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.types import Pose2


@dataclass
class AlignmentResult:
    pose: Pose2
    iterations: int
    final_cost: float
    converged: bool
    num_inliers: int
    diagnostics: dict = field(default_factory=dict)


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(residuals)
    w = np.ones_like(residuals)
    big = abs_r > delta
    w[big] = delta / np.maximum(abs_r[big], 1e-12)
    return w


def align_scan_to_tsdf(
    *,
    tsdf: Tsdf2D,
    scan_xy: np.ndarray,
    pose_init: Pose2,
    max_iters: int = 30,
    huber_delta_m: float = 0.15,
    converge_dx_m: float = 1e-4,
    converge_dtheta_rad: float = 1e-5,
    lm_lambda_init: float = 1e-4,
    lm_lambda_min: float = 1e-8,
    lm_lambda_max: float = 1e6,
) -> AlignmentResult:
    """Gauss-Newton / LM alignment of a single scan against a fixed TSDF.

    scan_xy: (N, 2) points in the sensor frame.
    Cost: sum_i rho(phi(T(pose) * p_i)), Huber loss.
    """
    if scan_xy.ndim != 2 or scan_xy.shape[1] != 2:
        raise ValueError("scan_xy must be (N, 2)")

    pts = np.ascontiguousarray(scan_xy, dtype=np.float64)
    pose = pose_init
    lam = float(lm_lambda_init)
    prev_cost = float("inf")
    iterations = 0
    converged = False
    num_inliers = 0
    final_cost = float("inf")

    for it in range(max_iters):
        iterations = it + 1
        c, s = float(np.cos(pose.theta)), float(np.sin(pose.theta))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        t = np.array([pose.x, pose.y], dtype=np.float64)
        pts_w = pts @ R.T + t  # (N, 2)

        phi_val, grad, valid = tsdf.sample(pts_w)
        if not np.any(valid):
            break

        r = phi_val[valid]
        g = grad[valid]
        pw = pts_w[valid]

        # Jacobian rows: dphi/dpose = grad . d(pw)/d(pose).
        # d(pw_x)/d(theta) = -(pw_y - y), d(pw_y)/d(theta) = (pw_x - x)
        dtheta_x = -(pw[:, 1] - pose.y)
        dtheta_y = pw[:, 0] - pose.x
        J = np.column_stack((g[:, 0], g[:, 1], g[:, 0] * dtheta_x + g[:, 1] * dtheta_y))

        wts = _huber_weights(r, huber_delta_m)
        Jw = J * wts[:, None]
        rw = r * wts
        H = Jw.T @ J
        b = Jw.T @ r
        cost = 0.5 * float(np.sum(rw * r))

        # LM damping
        H_lm = H + lam * np.eye(3)
        try:
            dx = -np.linalg.solve(H_lm, b)
        except np.linalg.LinAlgError:
            lam = min(lam * 10.0, lm_lambda_max)
            continue

        new_pose = Pose2(pose.x + float(dx[0]), pose.y + float(dx[1]), pose.theta + float(dx[2]))

        # evaluate trial cost
        c2, s2 = float(np.cos(new_pose.theta)), float(np.sin(new_pose.theta))
        R2 = np.array([[c2, -s2], [s2, c2]], dtype=np.float64)
        pts_w2 = pts @ R2.T + np.array([new_pose.x, new_pose.y])
        phi2, _, valid2 = tsdf.sample(pts_w2)
        if not np.any(valid2):
            lam = min(lam * 10.0, lm_lambda_max)
            continue
        r2 = phi2[valid2]
        w2 = _huber_weights(r2, huber_delta_m)
        trial_cost = 0.5 * float(np.sum(w2 * r2 * r2))

        if trial_cost < cost:
            # accept
            step_xy = float(np.hypot(dx[0], dx[1]))
            step_theta = abs(float(dx[2]))
            pose = new_pose
            lam = max(lam * 0.5, lm_lambda_min)
            final_cost = trial_cost
            num_inliers = int(np.sum(valid2))
            if step_xy < converge_dx_m and step_theta < converge_dtheta_rad:
                converged = True
                break
            prev_cost = trial_cost
        else:
            lam = min(lam * 10.0, lm_lambda_max)
            if lam >= lm_lambda_max:
                final_cost = cost
                num_inliers = int(np.sum(valid))
                break

    diagnostics = {
        "lm_lambda_final": lam,
        "prev_cost": prev_cost,
    }
    return AlignmentResult(
        pose=pose,
        iterations=iterations,
        final_cost=final_cost,
        converged=converged,
        num_inliers=num_inliers,
        diagnostics=diagnostics,
    )
