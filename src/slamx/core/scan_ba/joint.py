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
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import splu

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


_GPU_SPARSE = None


def _gpu_sparse():
    """Lazy (cupy, cupyx.scipy.sparse, cupyx.scipy.sparse.linalg) handles, cached."""
    global _GPU_SPARSE
    if _GPU_SPARSE is None:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspl

        _GPU_SPARSE = (cp, csp, cspl)
    return _GPU_SPARSE


def _solve_step_gpu(Hxx_lm, Hxp, bx, rhs, pp_r, pp_c, pp_v, V, diagp, K):
    """GPU Schur step: factorize the sparse SDF block H_phiphi on the device
    (cuSOLVER via cupyx splu) and form the reduced pose system there. Numerically
    identical to backend='schur'. Requires CUDA headers (set CUDA_PATH for JIT)."""
    cp, csp, cspl = _gpu_sparse()
    Hpp = csp.coo_matrix(
        (cp.asarray(pp_v), (cp.asarray(pp_r), cp.asarray(pp_c))), shape=(V, V)
    ).tocsc()
    Hpp = Hpp + diagp * csp.identity(V, format="csc")
    Y = cspl.splu(Hpp).solve(cp.asarray(rhs))  # (V, 3K+1)
    Hxp_g = cp.asarray(Hxp)
    YH, Yb = Y[:, : 3 * K], Y[:, 3 * K]
    S = cp.asarray(Hxx_lm) - Hxp_g @ YH
    dxx = cp.linalg.solve(S, -(cp.asarray(bx) - Hxp_g @ Yb))
    dxp = -(Yb + YH @ dxx)
    return cp.asnumpy(cp.concatenate([dxx, dxp]))


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
    sdf_smooth_info: float = 0.0,
    converge_dx_m: float = 1e-4,
    converge_dtheta_rad: float = 1e-5,
    lm_lambda_init: float = 1e-3,
    lm_lambda_min: float = 1e-8,
    lm_lambda_max: float = 1e6,
    backend: str = "schur",
) -> JointWindowResult:
    """backend="schur" (default) eliminates the SDF block via a sparse Schur complement
    (scales to large active-voxel counts); "dense" builds the full (3K+V) system;
    "schur_gpu" runs the sparse Schur factorization on the GPU (cupyx splu / cuSOLVER).
    All three give the same Gauss-Newton/LM step."""
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

    def assemble_blocks(per_scan, active_list, idx_of):
        """Data-term blocks: Hxx (3K,3K), Hxp (3K,V), bx, bp, and the H_phiphi COO
        triplets (rows, cols, vals). The SDF block is kept sparse for the Schur solve."""
        V = len(active_list)
        Hxx = np.zeros((3 * K, 3 * K))
        Hxp = np.zeros((3 * K, V))
        bx = np.zeros(3 * K)
        bp = np.zeros(V)
        pp_r, pp_c, pp_v = [], [], []
        cost = 0.0
        inliers = []
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
            JtW = Jp.T * hw
            Hxx[base:base + 3, base:base + 3] += JtW @ Jp
            bx[base:base + 3] += JtW @ r
            vloc = np.vectorize(idx_of.get)(neigh).astype(np.int64)  # (P,4)
            for a in range(4):
                wa = wts[:, a]
                col = vloc[:, a]
                for dd in range(3):
                    np.add.at(Hxp[base + dd], col, hw * Jp[:, dd] * wa)
                np.add.at(bp, col, hw * wa * r)
                for bb in range(4):
                    pp_r.append(vloc[:, a])
                    pp_c.append(vloc[:, bb])
                    pp_v.append(hw * wa * wts[:, bb])
            cost += 0.5 * float(np.sum(hw * r * r))
            inliers.append(int(r.size))
        pp = (
            np.concatenate(pp_r) if pp_r else np.zeros(0, np.int64),
            np.concatenate(pp_c) if pp_c else np.zeros(0, np.int64),
            np.concatenate(pp_v) if pp_v else np.zeros(0),
        )
        return Hxx, Hxp, bx, bp, pp, cost, inliers

    def add_priors_blocks(Hxx, bx, bp, cost, active_list):
        """Add pose motion/anchor priors to (Hxx, bx) and the SDF fold-prior to bp.
        The SDF prior's H contribution is the uniform `sdf_prior_info` on the H_phiphi
        diagonal, added inside the solve."""
        for i, mp in enumerate(state.motion_priors):
            Wd = np.array([mp.info_xy, mp.info_xy, mp.info_theta])
            r = np.array([
                cur_poses[i + 1].x - cur_poses[i].x - mp.delta_x,
                cur_poses[i + 1].y - cur_poses[i].y - mp.delta_y,
                cur_poses[i + 1].theta - cur_poses[i].theta - mp.delta_theta,
            ])
            W = np.diag(Wd)
            bi, bj = 3 * i, 3 * (i + 1)
            Hxx[bi:bi + 3, bi:bi + 3] += W
            Hxx[bj:bj + 3, bj:bj + 3] += W
            Hxx[bi:bi + 3, bj:bj + 3] -= W
            Hxx[bj:bj + 3, bi:bi + 3] -= W
            bx[bi:bi + 3] += -Wd * r
            bx[bj:bj + 3] += Wd * r
            cost += 0.5 * float(np.sum(Wd * r * r))
        if state.anchor is not None:
            Wd = np.array([state.anchor.info_xy, state.anchor.info_xy, state.anchor.info_theta])
            r = np.array([
                cur_poses[0].x - state.anchor.pose.x,
                cur_poses[0].y - state.anchor.pose.y,
                cur_poses[0].theta - state.anchor.pose.theta,
            ])
            Hxx[0:3, 0:3] += np.diag(Wd)
            bx[0:3] += Wd * r
            cost += 0.5 * float(np.sum(Wd * r * r))
        if active_list:
            av = np.array(active_list, dtype=np.int64)
            rp = tsdf.phi.ravel()[av] - phi0_full.ravel()[av]
            bp += sdf_prior_info * rp
            cost += 0.5 * float(sdf_prior_info * np.sum(rp * rp))
        return cost

    def smoothness(active_list, idx_of):
        """SDF smoothness regulariser between adjacent active voxels: r = phi_u - phi_v.
        Returns (cost, COO rows/cols/vals for H_phiphi, b_phi delta). Off by default."""
        V = len(active_list)
        empty = (0.0, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(V))
        if sdf_smooth_info <= 0.0 or V < 2:
            return empty
        w = tsdf.width
        us, nbs = [], []
        for v in active_list:
            if (v % w) != w - 1 and (v + 1) in idx_of:
                us.append(idx_of[v]); nbs.append(idx_of[v + 1])
            if (v + w) in idx_of:
                us.append(idx_of[v]); nbs.append(idx_of[v + w])
        if not us:
            return empty
        us = np.array(us, np.int64)
        nbs = np.array(nbs, np.int64)
        av = np.array(active_list, np.int64)
        flat = tsdf.phi.ravel()
        res = flat[av[us]] - flat[av[nbs]]
        lam = float(sdf_smooth_info)
        bp_delta = np.zeros(V)
        np.add.at(bp_delta, us, lam * res)
        np.add.at(bp_delta, nbs, -lam * res)
        sr = np.concatenate([us, nbs, us, nbs])
        sc = np.concatenate([us, nbs, nbs, us])
        sv = np.concatenate([
            np.full(us.size, lam), np.full(nbs.size, lam),
            np.full(us.size, -lam), np.full(nbs.size, -lam),
        ])
        return 0.5 * lam * float(np.sum(res * res)), sr, sc, sv, bp_delta

    def solve_step(Hxx, Hxp, bx, bp, pp, V, lam):
        """One LM increment dx = [dx_pose (3K); dx_phi (V)] by Schur-eliminating the
        SDF block. H_phiphi = data + (sdf_prior_info + lam) I; H_xx damped by lam."""
        Hxx_lm = Hxx + lam * np.eye(3 * K)
        if V == 0:
            return -np.linalg.solve(Hxx_lm, bx)
        diagp = sdf_prior_info + lam
        pp_r, pp_c, pp_v = pp
        rhs = np.column_stack([Hxp.T, bp])  # (V, 3K+1) = [H_phix | b_phi]
        if backend == "schur_gpu":
            return _solve_step_gpu(Hxx_lm, Hxp, bx, rhs, pp_r, pp_c, pp_v, V, diagp, K)
        if backend == "dense":
            Hpp = np.zeros((V, V))
            if pp_v.size:
                np.add.at(Hpp, (pp_r, pp_c), pp_v)
            Hpp[np.diag_indices(V)] += diagp
            Y = np.linalg.solve(Hpp, rhs)
        else:  # sparse Schur
            Hpp = coo_matrix((pp_v, (pp_r, pp_c)), shape=(V, V)).tocsc() + diagp * identity(V, format="csc")
            Y = splu(Hpp).solve(rhs)
        YH, Yb = Y[:, :3 * K], Y[:, 3 * K]
        S = Hxx_lm - Hxp @ YH
        dxx = np.linalg.solve(S, -(bx - Hxp @ Yb))
        dxp = -(Yb + YH @ dxx)
        return np.concatenate([dxx, dxp])

    def total_cost(active_list=None, idx_of=None):
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
        if active_list is not None:
            c += smoothness(active_list, idx_of)[0]
        return c

    cur_poses = list(state.poses)
    lam = float(lm_lambda_init)
    iterations = 0
    converged = False
    cost = float("inf")
    n_active = 0
    inliers: list[int] = [0] * K

    for it in range(max_iters):
        iterations = it + 1
        per_scan, active_list, idx_of = gather()
        n_active = len(active_list)
        Hxx, Hxp, bx, bp, pp, cost, inliers = assemble_blocks(per_scan, active_list, idx_of)
        cost = add_priors_blocks(Hxx, bx, bp, cost, active_list)
        sm_cost, sr, sc, sv, bp_delta = smoothness(active_list, idx_of)
        if sr.size:
            bp += bp_delta
            cost += sm_cost
            pp = (np.concatenate([pp[0], sr]), np.concatenate([pp[1], sc]), np.concatenate([pp[2], sv]))
        if not np.any(np.diag(Hxx) > 0):
            break

        try:
            dx = solve_step(Hxx, Hxp, bx, bp, pp, n_active, lam)
        except Exception:
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

        trial_cost = total_cost(active_list, idx_of)
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
        diagnostics={"lm_lambda_final": lam, "inliers_per_scan": list(inliers)},
    )
