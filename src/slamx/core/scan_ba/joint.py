"""Joint pose + SDF bundle adjustment over a fixed-lag window (P3, CPU reference).

The window solve in `window.py` treats the TSDF as fixed and optimises only the K
poses. Here the TSDF voxel values touched by the window's points become variables too,
so poses and the local map are refined together -- visual-SLAM style bundle adjustment
with the structure carried as a 2D signed-distance field.

For a sensor point p in scan t the point-to-SDF residual is the bilinear interpolation

    r = phi(T_t . p) = sum_c w_c * phi[v_c]

over its 4 surrounding voxels v_c with bilinear weights w_c. Its Jacobian therefore has
a pose part (grad(phi) . d(T p)/dxi, 3 entries, as in the pose-only solve) and an SDF
part (the 4 weights w_c, w.r.t. the 4 voxel values). A per-voxel prior pins phi to its
folded value phi0 so the map cannot collapse to the trivial phi == 0 everywhere.

This reference builds the full dense (3K + V) normal equations and solves them directly
(LM), so it is meant for modest windows. Schur elimination of the SDF block and a GPU
port are the next steps (P3.1+). Refined voxel values are written back into `tsdf`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.scan_ba.align import _huber_weights
from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.scan_ba.window import AnchorPrior, MotionPrior, WindowState
from slamx.core.types import Pose2


@dataclass
class JointWindowResult:
    state: WindowState
    iterations: int
    final_cost: float
    converged: bool
    num_active_voxels: int
    diagnostics: dict = field(default_factory=dict)


def _bilinear_terms(tsdf: Tsdf2D, pts_w: np.ndarray):
    """For map-frame points, return per-point (r, grad, valid, neigh_flat, weights).

    neigh_flat: (N, 4) flat voxel indices [00, 10, 01, 11]; weights: (N, 4) bilinear
    weights matching the order. Mirrors `Tsdf2D.sample` exactly (cell-centre offset).
    """
    res = float(tsdf.cfg.resolution_m)
    ox, oy = float(tsdf.cfg.origin_x_m), float(tsdf.cfg.origin_y_m)
    w, h = tsdf.width, tsdf.height
    phi = tsdf.phi
    wt = tsdf.weight

    gx = (pts_w[:, 0] - ox) / res - 0.5
    gy = (pts_w[:, 1] - oy) / res - 0.5
    ix0 = np.floor(gx).astype(np.int64)
    iy0 = np.floor(gy).astype(np.int64)
    ix1, iy1 = ix0 + 1, iy0 + 1
    in_bounds = (ix0 >= 0) & (ix1 < w) & (iy0 >= 0) & (iy1 < h)

    ix0c, ix1c = np.clip(ix0, 0, w - 1), np.clip(ix1, 0, w - 1)
    iy0c, iy1c = np.clip(iy0, 0, h - 1), np.clip(iy1, 0, h - 1)
    p00, p10, p01, p11 = phi[iy0c, ix0c], phi[iy0c, ix1c], phi[iy1c, ix0c], phi[iy1c, ix1c]
    w00, w10, w01, w11 = wt[iy0c, ix0c], wt[iy0c, ix1c], wt[iy1c, ix0c], wt[iy1c, ix1c]

    fx, fy = gx - ix0, gy - iy0
    ofx, ofy = 1.0 - fx, 1.0 - fy
    r = ofx * ofy * p00 + fx * ofy * p10 + ofx * fy * p01 + fx * fy * p11
    dphi_dx = ((-ofy) * p00 + ofy * p10 + (-fy) * p01 + fy * p11) / res
    dphi_dy = ((-ofx) * p00 + (-fx) * p10 + ofx * p01 + fx * p11) / res

    valid = in_bounds & (w00 > 0) & (w10 > 0) & (w01 > 0) & (w11 > 0)
    neigh = np.stack([iy0c * w + ix0c, iy0c * w + ix1c, iy1c * w + ix0c, iy1c * w + ix1c], axis=1)
    weights = np.stack([ofx * ofy, fx * ofy, ofx * fy, fx * fy], axis=1)
    grad = np.stack([dphi_dx, dphi_dy], axis=1)
    return r, grad, valid, neigh, weights


def optimize_window_joint(
    *,
    tsdf: Tsdf2D,
    state: WindowState,
    max_iters: int = 20,
    huber_delta_m: float = 0.15,
    sdf_prior_info: float = 10.0,
    converge_dx_m: float = 1e-4,
    converge_dtheta_rad: float = 1e-5,
    lm_lambda_init: float = 1e-3,
    lm_lambda_min: float = 1e-8,
    lm_lambda_max: float = 1e6,
) -> JointWindowResult:
    K = state.k
    phi0_full = tsdf.phi.copy()  # fold-time values; the SDF prior pins phi here

    def gather():
        """Active voxels + per-scan point terms at the current poses/phi."""
        per_scan = []
        active = set()
        for t in range(K):
            pts = state.scans[t]
            pose = cur_poses[t]
            if pts.shape[0] == 0:
                per_scan.append(None)
                continue
            c, s = np.cos(pose.theta), np.sin(pose.theta)
            R = np.array([[c, -s], [s, c]])
            pw = pts @ R.T + np.array([pose.x, pose.y])
            r, grad, valid, neigh, wts = _bilinear_terms(tsdf, pw)
            if not np.any(valid):
                per_scan.append(None)
                continue
            d = {
                "pw": pw[valid], "r": r[valid], "grad": grad[valid],
                "neigh": neigh[valid], "wts": wts[valid], "pose": pose,
            }
            per_scan.append(d)
            active.update(np.unique(d["neigh"]).tolist())
        active_list = sorted(active)
        idx_of = {v: i for i, v in enumerate(active_list)}
        return per_scan, active_list, idx_of

    def assemble(per_scan, active_list, idx_of):
        V = len(active_list)
        n = 3 * K + V
        H = np.zeros((n, n))
        b = np.zeros(n)
        cost = 0.0
        inliers = []
        flat_phi = tsdf.phi.ravel()
        for t in range(K):
            d = per_scan[t]
            if d is None:
                inliers.append(0)
                continue
            r, grad, pw, neigh, wts = d["r"], d["grad"], d["pw"], d["neigh"], d["wts"]
            pose = d["pose"]
            hw = _huber_weights(r, huber_delta_m)
            dtheta = grad[:, 0] * (-(pw[:, 1] - pose.y)) + grad[:, 1] * (pw[:, 0] - pose.x)
            Jp = np.column_stack([grad[:, 0], grad[:, 1], dtheta])  # (P,3)
            base = 3 * t
            # pose-pose block + pose b
            JtW = Jp.T * hw
            H[base:base + 3, base:base + 3] += JtW @ Jp
            b[base:base + 3] += JtW @ r
            vloc = np.vectorize(idx_of.get)(neigh).astype(np.int64)  # (P,4) local voxel indices
            # pose-sdf cross block + sdf-sdf block + sdf b
            for a in range(4):
                col = 3 * K + vloc[:, a]
                wa = wts[:, a]
                # H_xphi (3 x ...) : sum hw * Jp_d * w_a
                for dd in range(3):
                    np.add.at(H[base + dd], col, hw * Jp[:, dd] * wa)
                    np.add.at(H[:, base + dd], col, hw * Jp[:, dd] * wa)  # symmetric
                # sdf b
                np.add.at(b, col, hw * wa * r)
                for bb in range(4):
                    np.add.at(H, (col, 3 * K + vloc[:, bb]), hw * wa * wts[:, bb])
            cost += 0.5 * float(np.sum(hw * r * r))
            inliers.append(int(r.size))
        return H, b, cost, inliers, V, flat_phi

    def add_priors(H, b, cost, active_list):
        # pose motion priors (global-frame, as in window._evaluate)
        for i, mp in enumerate(state.motion_priors):
            Wd = np.array([mp.info_xy, mp.info_xy, mp.info_theta])
            r = np.array([
                cur_poses[i + 1].x - cur_poses[i].x - mp.delta_x,
                cur_poses[i + 1].y - cur_poses[i].y - mp.delta_y,
                cur_poses[i + 1].theta - cur_poses[i].theta - mp.delta_theta,
            ])
            W = np.diag(Wd)
            bi, bj = 3 * i, 3 * (i + 1)
            H[bi:bi + 3, bi:bi + 3] += W
            H[bj:bj + 3, bj:bj + 3] += W
            H[bi:bi + 3, bj:bj + 3] -= W
            H[bj:bj + 3, bi:bi + 3] -= W
            b[bi:bi + 3] += -Wd * r
            b[bj:bj + 3] += Wd * r
            cost += 0.5 * float(np.sum(Wd * r * r))
        if state.anchor is not None:
            Wd = np.array([state.anchor.info_xy, state.anchor.info_xy, state.anchor.info_theta])
            r = np.array([
                cur_poses[0].x - state.anchor.pose.x,
                cur_poses[0].y - state.anchor.pose.y,
                cur_poses[0].theta - state.anchor.pose.theta,
            ])
            H[0:3, 0:3] += np.diag(Wd)
            b[0:3] += Wd * r
            cost += 0.5 * float(np.sum(Wd * r * r))
        # SDF prior: pin each active voxel to its fold-time value phi0
        if active_list:
            av = np.array(active_list, dtype=np.int64)
            cur = tsdf.phi.ravel()[av]
            rp = cur - phi0_full.ravel()[av]
            diag = np.arange(3 * K, 3 * K + len(av))
            H[diag, diag] += sdf_prior_info
            b[diag] += sdf_prior_info * rp
            cost += 0.5 * float(sdf_prior_info * np.sum(rp * rp))
        return cost

    def total_cost():
        """Cost at the current poses/phi (data + priors), for accept/reject."""
        c = 0.0
        for t in range(K):
            pts = state.scans[t]
            if pts.shape[0] == 0:
                continue
            pose = cur_poses[t]
            cc, ss = np.cos(pose.theta), np.sin(pose.theta)
            pw = pts @ np.array([[cc, -ss], [ss, cc]]).T + np.array([pose.x, pose.y])
            r, _, valid, _, _ = _bilinear_terms(tsdf, pw)
            if np.any(valid):
                rv = r[valid]
                c += 0.5 * float(np.sum(_huber_weights(rv, huber_delta_m) * rv * rv))
        for i, mp in enumerate(state.motion_priors):
            Wd = np.array([mp.info_xy, mp.info_xy, mp.info_theta])
            r = np.array([
                cur_poses[i + 1].x - cur_poses[i].x - mp.delta_x,
                cur_poses[i + 1].y - cur_poses[i].y - mp.delta_y,
                cur_poses[i + 1].theta - cur_poses[i].theta - mp.delta_theta,
            ])
            c += 0.5 * float(np.sum(Wd * r * r))
        if state.anchor is not None:
            Wd = np.array([state.anchor.info_xy, state.anchor.info_xy, state.anchor.info_theta])
            r = np.array([
                cur_poses[0].x - state.anchor.pose.x,
                cur_poses[0].y - state.anchor.pose.y,
                cur_poses[0].theta - state.anchor.pose.theta,
            ])
            c += 0.5 * float(np.sum(Wd * r * r))
        av = np.where(phi0_full.ravel() != tsdf.phi.ravel())[0]
        if av.size:
            rp = tsdf.phi.ravel()[av] - phi0_full.ravel()[av]
            c += 0.5 * float(sdf_prior_info * np.sum(rp * rp))
        return c

    cur_poses = list(state.poses)
    lam = float(lm_lambda_init)
    iterations = 0
    converged = False
    cost = float("inf")
    n_active = 0

    for it in range(max_iters):
        iterations = it + 1
        per_scan, active_list, idx_of = gather()
        n_active = len(active_list)
        H, b, cost, inliers, V, _ = assemble(per_scan, active_list, idx_of)
        cost = add_priors(H, b, cost, active_list)
        if not np.any(np.diag(H) > 0):
            break

        H_lm = H + lam * np.eye(H.shape[0])
        try:
            dx = -np.linalg.solve(H_lm, b)
        except np.linalg.LinAlgError:
            lam = min(lam * 10.0, lm_lambda_max)
            continue

        # snapshot to allow rejection
        saved_poses = list(cur_poses)
        saved_phi = tsdf.phi.copy()
        cur_poses = [
            Pose2(p.x + dx[3 * i], p.y + dx[3 * i + 1], p.theta + dx[3 * i + 2])
            for i, p in enumerate(cur_poses)
        ]
        if active_list:
            av = np.array(active_list, dtype=np.int64)
            flat = tsdf.phi.ravel()
            flat[av] += dx[3 * K:3 * K + len(av)]
            tsdf.phi[:] = flat.reshape(tsdf.phi.shape)

        trial_cost = total_cost()
        if trial_cost < cost:
            step_xy = float(np.max(np.hypot(dx[0:3 * K:3], dx[1:3 * K:3]))) if K else 0.0
            step_th = float(np.max(np.abs(dx[2:3 * K:3]))) if K else 0.0
            lam = max(lam * 0.5, lm_lambda_min)
            cost = trial_cost
            if step_xy < converge_dx_m and step_th < converge_dtheta_rad:
                converged = True
                break
        else:
            cur_poses = saved_poses
            tsdf.phi[:] = saved_phi
            lam = min(lam * 10.0, lm_lambda_max)
            if lam >= lm_lambda_max:
                break

    out_state = WindowState(
        poses=cur_poses,
        scans=state.scans,
        motion_priors=list(state.motion_priors),
        anchor=state.anchor,
    )
    return JointWindowResult(
        state=out_state,
        iterations=iterations,
        final_cost=cost,
        converged=converged,
        num_active_voxels=n_active,
        diagnostics={"lm_lambda_final": lam},
    )
