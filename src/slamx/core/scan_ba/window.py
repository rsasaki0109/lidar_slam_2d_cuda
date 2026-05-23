from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.scan_ba.align import _huber_weights
from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.types import Pose2


@dataclass(frozen=True)
class MotionPrior:
    """Simplified global-frame relative motion prior between two consecutive poses.

    Residual r = (x_{i+1} - x_i, y_{i+1} - y_i, theta_{i+1} - theta_i) - (dx, dy, dtheta).
    This linearization is exact-ish for small angular changes and is sufficient for P1
    where prior strength is moderate and the data term dominates pose recovery.
    Use a stronger prior + tighter delta to express odometry / constant-velocity guesses.
    """

    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_theta: float = 0.0
    info_xy: float = 25.0
    info_theta: float = 25.0


@dataclass(frozen=True)
class AnchorPrior:
    """Strong prior used to fix the oldest pose in the window (marginalize-as-prior)."""

    pose: Pose2
    info_xy: float = 1.0e6
    info_theta: float = 1.0e6


@dataclass
class WindowState:
    poses: list[Pose2]
    scans: list[np.ndarray]  # each (N_i, 2) in sensor frame
    motion_priors: list[MotionPrior] = field(default_factory=list)
    anchor: AnchorPrior | None = None

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.scans):
            raise ValueError("poses and scans length mismatch")
        if self.motion_priors and len(self.motion_priors) != len(self.poses) - 1:
            raise ValueError("motion_priors length must equal len(poses) - 1")
        for s in self.scans:
            if s.ndim != 2 or s.shape[1] != 2:
                raise ValueError("each scan must be (N, 2)")

    @property
    def k(self) -> int:
        return len(self.poses)


@dataclass
class WindowResult:
    state: WindowState
    iterations: int
    final_cost: float
    converged: bool
    diagnostics: dict = field(default_factory=dict)


def _accumulate_data_block(
    *,
    tsdf: Tsdf2D,
    pose: Pose2,
    pts: np.ndarray,
    huber_delta_m: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Compute J^T W J (3x3), J^T W r (3,), cost contribution, inlier count for one scan."""
    c, s = float(np.cos(pose.theta)), float(np.sin(pose.theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = np.array([pose.x, pose.y], dtype=np.float64)
    pts_w = pts @ R.T + t
    phi_val, grad, valid = tsdf.sample(pts_w)
    if not np.any(valid):
        return np.zeros((3, 3)), np.zeros(3), 0.0, 0
    r = phi_val[valid]
    g = grad[valid]
    pw = pts_w[valid]
    dtheta_x = -(pw[:, 1] - pose.y)
    dtheta_y = pw[:, 0] - pose.x
    J = np.column_stack((g[:, 0], g[:, 1], g[:, 0] * dtheta_x + g[:, 1] * dtheta_y))
    w = _huber_weights(r, huber_delta_m)
    JtW = J.T * w
    H_block = JtW @ J
    b_block = JtW @ r
    cost = 0.5 * float(np.sum(w * r * r))
    return H_block, b_block, cost, int(valid.sum())


def _motion_prior_residual(
    pose_i: Pose2, pose_j: Pose2, mp: MotionPrior
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, W_diag) for the global-frame motion prior between pose_i and pose_j."""
    r = np.array(
        [
            (pose_j.x - pose_i.x) - mp.delta_x,
            (pose_j.y - pose_i.y) - mp.delta_y,
            (pose_j.theta - pose_i.theta) - mp.delta_theta,
        ],
        dtype=np.float64,
    )
    W_diag = np.array([mp.info_xy, mp.info_xy, mp.info_theta], dtype=np.float64)
    return r, W_diag


def _anchor_residual(pose: Pose2, anchor: AnchorPrior) -> tuple[np.ndarray, np.ndarray]:
    r = np.array(
        [pose.x - anchor.pose.x, pose.y - anchor.pose.y, pose.theta - anchor.pose.theta],
        dtype=np.float64,
    )
    W_diag = np.array([anchor.info_xy, anchor.info_xy, anchor.info_theta], dtype=np.float64)
    return r, W_diag


def _evaluate(state: WindowState, tsdf: Tsdf2D, huber_delta_m: float) -> tuple[np.ndarray, np.ndarray, float, list[int]]:
    """Assemble (H, b, cost, inlier_per_scan)."""
    k = state.k
    n = 3 * k
    H = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    cost = 0.0
    inliers: list[int] = []
    for i in range(k):
        H_block, b_block, c_i, ni = _accumulate_data_block(
            tsdf=tsdf, pose=state.poses[i], pts=state.scans[i], huber_delta_m=huber_delta_m
        )
        H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += H_block
        b[3 * i : 3 * i + 3] += b_block
        cost += c_i
        inliers.append(ni)

    # motion priors: r = x_{i+1} - x_i - delta; J_i = -I, J_{i+1} = +I
    for i, mp in enumerate(state.motion_priors):
        r, W_diag = _motion_prior_residual(state.poses[i], state.poses[i + 1], mp)
        W = np.diag(W_diag)
        H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += W
        H[3 * (i + 1) : 3 * (i + 1) + 3, 3 * (i + 1) : 3 * (i + 1) + 3] += W
        H[3 * i : 3 * i + 3, 3 * (i + 1) : 3 * (i + 1) + 3] -= W
        H[3 * (i + 1) : 3 * (i + 1) + 3, 3 * i : 3 * i + 3] -= W
        b[3 * i : 3 * i + 3] += -W_diag * r
        b[3 * (i + 1) : 3 * (i + 1) + 3] += W_diag * r
        cost += 0.5 * float(np.sum(W_diag * r * r))

    # anchor on pose 0
    if state.anchor is not None:
        r, W_diag = _anchor_residual(state.poses[0], state.anchor)
        H[0:3, 0:3] += np.diag(W_diag)
        b[0:3] += W_diag * r
        cost += 0.5 * float(np.sum(W_diag * r * r))

    return H, b, cost, inliers


def _apply_increment(poses: list[Pose2], dx: np.ndarray) -> list[Pose2]:
    out = []
    for i, p in enumerate(poses):
        out.append(Pose2(p.x + float(dx[3 * i]), p.y + float(dx[3 * i + 1]), p.theta + float(dx[3 * i + 2])))
    return out


def optimize_window(
    *,
    tsdf: Tsdf2D,
    state: WindowState,
    max_iters: int = 30,
    huber_delta_m: float = 0.15,
    converge_dx_m: float = 1e-4,
    converge_dtheta_rad: float = 1e-5,
    lm_lambda_init: float = 1e-4,
    lm_lambda_min: float = 1e-8,
    lm_lambda_max: float = 1e6,
) -> WindowResult:
    cur = WindowState(
        poses=list(state.poses),
        scans=state.scans,
        motion_priors=list(state.motion_priors),
        anchor=state.anchor,
    )
    lam = float(lm_lambda_init)
    iterations = 0
    converged = False
    inliers: list[int] = []
    cost = float("inf")

    for it in range(max_iters):
        iterations = it + 1
        H, b, cost, inliers = _evaluate(cur, tsdf, huber_delta_m)
        if not np.any(np.diag(H) > 0):
            break

        H_lm = H + lam * np.eye(H.shape[0])
        try:
            dx = -np.linalg.solve(H_lm, b)
        except np.linalg.LinAlgError:
            lam = min(lam * 10.0, lm_lambda_max)
            continue

        trial_poses = _apply_increment(cur.poses, dx)
        trial = WindowState(
            poses=trial_poses, scans=cur.scans, motion_priors=cur.motion_priors, anchor=cur.anchor
        )
        _, _, trial_cost, _ = _evaluate(trial, tsdf, huber_delta_m)

        if trial_cost < cost:
            step_xy = float(np.max(np.hypot(dx[0::3], dx[1::3])))
            step_theta = float(np.max(np.abs(dx[2::3])))
            cur = trial
            lam = max(lam * 0.5, lm_lambda_min)
            cost = trial_cost
            if step_xy < converge_dx_m and step_theta < converge_dtheta_rad:
                converged = True
                break
        else:
            lam = min(lam * 10.0, lm_lambda_max)
            if lam >= lm_lambda_max:
                break

    return WindowResult(
        state=cur,
        iterations=iterations,
        final_cost=cost,
        converged=converged,
        diagnostics={"lm_lambda_final": lam, "inliers_per_scan": inliers},
    )


def slide_window(
    state: WindowState,
    *,
    new_scan: np.ndarray,
    new_pose_init: Pose2,
    new_motion_prior: MotionPrior,
    bake_old_as_anchor: bool = True,
    anchor_info_xy: float = 1.0e6,
    anchor_info_theta: float = 1.0e6,
) -> WindowState:
    """Drop the oldest pose, append a new one. Optionally bake the displaced
    second-oldest pose as the new AnchorPrior (marginalize-as-prior approximation).
    """
    if state.k == 0:
        raise ValueError("cannot slide an empty window")
    new_poses = state.poses[1:] + [new_pose_init]
    new_scans = state.scans[1:] + [new_scan]
    # the old motion prior between pose_0 and pose_1 is dropped; remaining shift left.
    new_motion_priors = list(state.motion_priors[1:]) + [new_motion_prior]
    if bake_old_as_anchor:
        # the surviving oldest is what was state.poses[1]; pin it as the new anchor.
        anchor = AnchorPrior(
            pose=state.poses[1] if state.k >= 2 else state.poses[0],
            info_xy=anchor_info_xy,
            info_theta=anchor_info_theta,
        )
    else:
        anchor = state.anchor
    return WindowState(poses=new_poses, scans=new_scans, motion_priors=new_motion_priors, anchor=anchor)
