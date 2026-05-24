"""Exact sliding-window marginalization (P4).

`slide_window`'s default bakes the surviving oldest pose as a strong AnchorPrior --
a heuristic that throws away the information the dropped pose carried and replaces
it with an arbitrary `info_xy/theta`. The principled alternative is to *marginalize*
the dropped pose: Schur-complement it out of the factors that touch it, leaving a
dense Gaussian prior on the variables it was connected to.

In this fixed-lag window the oldest pose (pose 0) shares factors only with pose 1
(the motion prior between them); its data term and any anchor/previous-marginal
prior are unary on pose 0. So eliminating pose 0 yields a 3x3 linearized prior on
pose 1 -- the new oldest pose -- which exactly reproduces, for the retained
variables, the step the full window would have taken (first-estimate Jacobians).

Cost contributed by a `MarginalizationPrior` on a pose x (around x_lin):

    E(x) = 0.5 (x - x_lin)^T Lambda (x - x_lin) + g^T (x - x_lin)

so its gradient is `Lambda (x - x_lin) + g` and its Hessian is `Lambda`, matching
the (H += ..., b += ...) convention used in `window._evaluate` (H dx = -b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.scan_ba.align import _huber_weights
from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.types import Pose2


@dataclass(frozen=True)
class MarginalizationPrior:
    """Linearized Gaussian prior on a single pose produced by marginalization.

    x_lin: (3,) linearization point [x, y, theta] of the retained pose.
    Lambda: (3,3) information matrix (Hessian) of the prior.
    g:      (3,) gradient of the prior at x_lin.
    """

    x_lin: np.ndarray
    Lambda: np.ndarray
    g: np.ndarray

    def blocks(self, pose: Pose2) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (H_block 3x3, b_block 3, cost) for `pose` under this prior."""
        delta = np.array([pose.x - self.x_lin[0], pose.y - self.x_lin[1], pose.theta - self.x_lin[2]])
        b = self.Lambda @ delta + self.g
        cost = 0.5 * float(delta @ self.Lambda @ delta) + float(self.g @ delta)
        return self.Lambda.copy(), b, cost


def schur_complement(H: np.ndarray, b: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Marginalize the first `m` variables of (H, b); return (Lambda, g) over the rest.

    Lambda = H_rr - H_rm H_mm^-1 H_mr ;  g = b_r - H_rm H_mm^-1 b_m.
    Uses a solve against H_mm (a small symmetric PD block in practice).
    """
    Hmm, Hmr = H[:m, :m], H[:m, m:]
    Hrm, Hrr = H[m:, :m], H[m:, m:]
    bm, br = b[:m], b[m:]
    # H_mm^-1 [H_mr | b_m] in one solve
    rhs = np.column_stack([Hmr, bm])
    sol = np.linalg.solve(Hmm, rhs)
    Hmm_inv_Hmr, Hmm_inv_bm = sol[:, : H.shape[0] - m], sol[:, -1]
    Lambda = Hrr - Hrm @ Hmm_inv_Hmr
    g = br - Hrm @ Hmm_inv_bm
    Lambda = 0.5 * (Lambda + Lambda.T)  # symmetrize against round-off
    return Lambda, g


def _data_block(tsdf: Tsdf2D, pose: Pose2, pts: np.ndarray, huber_delta_m: float):
    """Pose data-term (H 3x3, b 3) at `pose`; mirrors window._accumulate_data_block."""
    if pts.shape[0] == 0:
        return np.zeros((3, 3)), np.zeros(3)
    c, s = float(np.cos(pose.theta)), float(np.sin(pose.theta))
    R = np.array([[c, -s], [s, c]])
    pts_w = pts @ R.T + np.array([pose.x, pose.y])
    phi_val, grad, valid = tsdf.sample(pts_w)
    if not np.any(valid):
        return np.zeros((3, 3)), np.zeros(3)
    r = phi_val[valid]
    g = grad[valid]
    pw = pts_w[valid]
    J = np.column_stack((g[:, 0], g[:, 1], g[:, 0] * (-(pw[:, 1] - pose.y)) + g[:, 1] * (pw[:, 0] - pose.x)))
    w = _huber_weights(r, huber_delta_m)
    JtW = J.T * w
    return JtW @ J, JtW @ r


def marginalize_oldest_pose(
    *,
    tsdf: Tsdf2D,
    poses: list[Pose2],
    scans: list[np.ndarray],
    motion_prior,
    huber_delta_m: float,
    anchor=None,
    prev_marg: MarginalizationPrior | None = None,
) -> MarginalizationPrior:
    """Eliminate pose 0 and return the resulting prior on pose 1 (the new oldest).

    Assembles the 6x6 information over [pose0, pose1] from exactly the factors that
    touch pose 0 -- its data term, its anchor / previous marginal prior, and the
    motion prior linking it to pose 1 -- then Schur-eliminates pose 0. Linearized at
    the current `poses` (first-estimate Jacobians for the nonlinear data term).
    """
    if len(poses) < 2:
        raise ValueError("need at least 2 poses to marginalize the oldest")
    p0, p1 = poses[0], poses[1]
    H = np.zeros((6, 6))
    b = np.zeros(6)

    # pose-0 data term (unary)
    Hd, bd = _data_block(tsdf, p0, scans[0], huber_delta_m)
    H[0:3, 0:3] += Hd
    b[0:3] += bd

    # anchor on pose 0 (unary), if present
    if anchor is not None:
        Wd = np.array([anchor.info_xy, anchor.info_xy, anchor.info_theta])
        r = np.array([p0.x - anchor.pose.x, p0.y - anchor.pose.y, p0.theta - anchor.pose.theta])
        H[0:3, 0:3] += np.diag(Wd)
        b[0:3] += Wd * r

    # previous marginal prior on pose 0 (unary), if recursing
    if prev_marg is not None:
        Hp, bp, _ = prev_marg.blocks(p0)
        H[0:3, 0:3] += Hp
        b[0:3] += bp

    # motion prior pose0 - pose1 (binary): r = (p1 - p0) - delta ; J0=-I, J1=+I
    Wd = np.array([motion_prior.info_xy, motion_prior.info_xy, motion_prior.info_theta])
    rm = np.array([
        (p1.x - p0.x) - motion_prior.delta_x,
        (p1.y - p0.y) - motion_prior.delta_y,
        (p1.theta - p0.theta) - motion_prior.delta_theta,
    ])
    W = np.diag(Wd)
    H[0:3, 0:3] += W
    H[3:6, 3:6] += W
    H[0:3, 3:6] -= W
    H[3:6, 0:3] -= W
    b[0:3] += -Wd * rm
    b[3:6] += Wd * rm

    Lambda, g = schur_complement(H, b, m=3)
    return MarginalizationPrior(x_lin=np.array([p1.x, p1.y, p1.theta]), Lambda=Lambda, g=g)
