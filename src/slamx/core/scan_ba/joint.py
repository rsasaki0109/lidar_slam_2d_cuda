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


def _pcg_spd_multi(cp, A, B, tol=1e-12, max_iter=2000):
    """Jacobi-preconditioned CG solving A X = B for all columns of B at once (A SPD).

    The SDF block H_phiphi is symmetric positive-definite and, under a strong SDF
    prior (its diagonal is `sdf_prior_info + lambda`), strongly diagonally dominant --
    so a diagonally-preconditioned CG converges in a handful of iterations and replaces
    the cuSOLVER sparse-LU factorization (the profiled wall of the joint GPU solve)
    with cheap cuSPARSE spmm + reductions. Each RHS column runs an independent CG,
    vectorized over columns; we iterate until every column's relative residual < tol.
    A is a cupyx CSR matrix (V,V); B is (V, m). Returns X (V, m)."""
    diag = A.diagonal()
    diag = cp.where(diag == 0, 1.0, diag)
    Minv = (1.0 / diag)[:, None]
    X = cp.zeros_like(B)
    R = B.copy()  # B - A@0
    bnorm = cp.linalg.norm(B, axis=0)
    bnorm = cp.where(bnorm == 0, 1.0, bnorm)
    Z = Minv * R
    P = Z.copy()
    rz = cp.sum(R * Z, axis=0)
    for _ in range(max_iter):
        AP = A @ P
        pAp = cp.sum(P * AP, axis=0)
        alpha = (rz / cp.where(pAp == 0, 1.0, pAp))[None, :]
        X = X + alpha * P
        R = R - alpha * AP
        if bool(cp.all(cp.linalg.norm(R, axis=0) / bnorm < tol)):
            break
        Z = Minv * R
        rz_new = cp.sum(R * Z, axis=0)
        beta = (rz_new / cp.where(rz == 0, 1.0, rz))[None, :]
        P = Z + beta * P
        rz = rz_new
    return X


def _bilinear_terms_gpu(cp, phi_d, wt_d, cfg, pts_w):
    """Device port of `_bilinear_terms`. phi_d/wt_d are float32 (matching the host TSDF
    store so r is computed from the same float32 voxel values); returns device arrays
    (r, dphi_dx, dphi_dy, valid, neigh (N,4), wts (N,4))."""
    res = float(cfg.resolution_m)
    ox, oy = float(cfg.origin_x_m), float(cfg.origin_y_m)
    h, w = int(phi_d.shape[0]), int(phi_d.shape[1])
    gx = (pts_w[:, 0] - ox) / res - 0.5
    gy = (pts_w[:, 1] - oy) / res - 0.5
    ix0 = cp.floor(gx).astype(cp.int64)
    iy0 = cp.floor(gy).astype(cp.int64)
    ix1, iy1 = ix0 + 1, iy0 + 1
    in_bounds = (ix0 >= 0) & (ix1 < w) & (iy0 >= 0) & (iy1 < h)
    ix0c, ix1c = cp.clip(ix0, 0, w - 1), cp.clip(ix1, 0, w - 1)
    iy0c, iy1c = cp.clip(iy0, 0, h - 1), cp.clip(iy1, 0, h - 1)
    p00, p10, p01, p11 = phi_d[iy0c, ix0c], phi_d[iy0c, ix1c], phi_d[iy1c, ix0c], phi_d[iy1c, ix1c]
    w00, w10, w01, w11 = wt_d[iy0c, ix0c], wt_d[iy0c, ix1c], wt_d[iy1c, ix0c], wt_d[iy1c, ix1c]
    fx, fy = gx - ix0, gy - iy0
    ofx, ofy = 1.0 - fx, 1.0 - fy
    r = ofx * ofy * p00 + fx * ofy * p10 + ofx * fy * p01 + fx * fy * p11
    dphi_dx = ((-ofy) * p00 + ofy * p10 + (-fy) * p01 + fy * p11) / res
    dphi_dy = ((-ofx) * p00 + (-fx) * p10 + ofx * p01 + fx * p11) / res
    valid = in_bounds & (w00 > 0) & (w10 > 0) & (w01 > 0) & (w11 > 0)
    neigh = cp.stack([iy0c * w + ix0c, iy0c * w + ix1c, iy1c * w + ix0c, iy1c * w + ix1c], axis=1)
    wts = cp.stack([ofx * ofy, fx * ofy, ofx * fy, fx * fy], axis=1)
    return r, dphi_dx, dphi_dy, valid, neigh, wts


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


def _pose_priors_host(state: WindowState, cur_poses, K: int):
    """Pose motion/anchor/marginalization prior contributions (Hxx 3K x 3K, bx 3K,
    cost). Identical formulas to `add_priors_blocks`'s pose part and `total_cost`'s
    pose part, factored out so the GPU path reuses the exact CPU arithmetic."""
    H = np.zeros((3 * K, 3 * K))
    b = np.zeros(3 * K)
    cost = 0.0
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
    if state.marg_prior is not None:
        Hm, bm, cm = state.marg_prior.blocks(cur_poses[0])
        H[0:3, 0:3] += Hm
        b[0:3] += bm
        cost += cm
    return H, b, cost


def _optimize_window_joint_gpu(
    *, tsdf, state, max_iters, huber_delta_m, sdf_prior_info,
    converge_dx_m, converge_dtheta_rad, lm_lambda_init, lm_lambda_min, lm_lambda_max,
    gpu_solver="pcg",
) -> JointWindowResult:
    """Fully on-device joint pose+SDF window solve (P3.5).

    Moves the gather + assemble (the profiled bottleneck -- not the linear solve) onto
    the GPU: bilinear sampling, per-scan Jacobian reduction, the H_xphi / b_phi scatter
    and the sparse H_phiphi COO all run in cupy, and the Schur solve factorizes
    H_phiphi on device (cuSOLVER). Mirrors backend='schur' to float reduction order.
    Pose priors are evaluated on the host (tiny) for bit-identical arithmetic. SDF
    smoothness is not supported here (assert off)."""
    cp, csp, cspl = _gpu_sparse()
    K = state.k
    phi_d = cp.asarray(tsdf.phi, dtype=cp.float32)
    wt_d = cp.asarray(tsdf.weight, dtype=cp.float32)
    phi0_d = phi_d.copy()
    W = int(phi_d.shape[1])
    cfg = tsdf.cfg

    sizes = [int(s.shape[0]) for s in state.scans]
    ntot = int(sum(sizes))
    if ntot > 0:
        pts_d = cp.asarray(np.concatenate(state.scans, axis=0), dtype=cp.float64)
        seg_all = cp.repeat(cp.arange(K, dtype=cp.int64), cp.asarray(sizes, dtype=cp.int64))
    else:
        pts_d = cp.zeros((0, 2), dtype=cp.float64)
        seg_all = cp.zeros((0,), dtype=cp.int64)

    # block-diagonal scatter indices for the per-scan 3x3 data Hessian blocks
    base = cp.arange(K, dtype=cp.int64) * 3
    lr = cp.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=cp.int64)
    lc = cp.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=cp.int64)
    rows_bd = (base[:, None] + lr[None, :]).reshape(-1)
    cols_bd = (base[:, None] + lc[None, :]).reshape(-1)

    def _world(P):
        th = P[seg_all, 2]
        c, s = cp.cos(th), cp.sin(th)
        tx, ty = P[seg_all, 0], P[seg_all, 1]
        pwx = c * pts_d[:, 0] - s * pts_d[:, 1] + tx
        pwy = s * pts_d[:, 0] + c * pts_d[:, 1] + ty
        return cp.stack([pwx, pwy], axis=1), pwx, pwy, tx, ty

    def _huber(r):
        ar = cp.abs(r)
        return cp.where(ar > huber_delta_m, huber_delta_m / cp.maximum(ar, 1e-12), 1.0)

    def data_cost(P):
        if ntot == 0:
            return 0.0
        pw, _, _, _, _ = _world(P)
        r, _, _, valid, _, _ = _bilinear_terms_gpu(cp, phi_d, wt_d, cfg, pw)
        if not bool(valid.any()):
            return 0.0
        rv = r[valid]
        return 0.5 * float(cp.sum(_huber(rv) * rv * rv))

    def sdf_cost_all_changed():
        diff = phi0_d.ravel() != phi_d.ravel()
        if not bool(diff.any()):
            return 0.0
        rp = phi_d.ravel()[diff] - phi0_d.ravel()[diff]
        return 0.5 * float(sdf_prior_info * cp.sum(rp * rp))

    cur_poses = list(state.poses)
    lam = float(lm_lambda_init)
    iterations = 0
    converged = False
    cost = float("inf")
    n_active = 0
    inliers = [0] * K

    for it in range(max_iters):
        iterations = it + 1
        P = cp.asarray([[p.x, p.y, p.theta] for p in cur_poses], dtype=cp.float64)

        Hxx = cp.zeros((3 * K, 3 * K), dtype=cp.float64)
        bx = cp.zeros(3 * K, dtype=cp.float64)
        data_c = 0.0
        active = cp.zeros(0, dtype=cp.int64)
        vloc = None
        pp_r = pp_c = pp_v = None
        bp = cp.zeros(0, dtype=cp.float64)
        Hxp = cp.zeros((3 * K, 0), dtype=cp.float64)

        if ntot > 0:
            pw, pwx, pwy, tx, ty = _world(P)
            r, gx, gy, valid, neigh, wts = _bilinear_terms_gpu(cp, phi_d, wt_d, cfg, pw)
            if bool(valid.any()):
                segv = seg_all[valid]
                rv, gxv, gyv = r[valid], gx[valid], gy[valid]
                pwyv, pwxv = pwy[valid], pwx[valid]
                txv, tyv = tx[valid], ty[valid]
                neighv, wtsv = neigh[valid], wts[valid]
                hw = _huber(rv)
                j2 = gxv * (-(pwyv - tyv)) + gyv * (pwxv - txv)
                Jcols = (gxv, gyv, j2)

                active = cp.unique(neighv.ravel())
                n_active = int(active.size)
                vloc = cp.searchsorted(active, neighv)  # (M,4) local voxel indices
                inliers = [int(v) for v in cp.asnumpy(cp.bincount(segv, minlength=K))]

                def _seg(vals):
                    return cp.bincount(segv, weights=vals, minlength=K)

                h = [_seg(hw * Jcols[i] * Jcols[j]) for i, j in
                     ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
                Hblk = cp.empty((K, 3, 3), dtype=cp.float64)
                Hblk[:, 0, 0] = h[0]
                Hblk[:, 0, 1] = Hblk[:, 1, 0] = h[1]
                Hblk[:, 0, 2] = Hblk[:, 2, 0] = h[2]
                Hblk[:, 1, 1] = h[3]
                Hblk[:, 1, 2] = Hblk[:, 2, 1] = h[4]
                Hblk[:, 2, 2] = h[5]
                Hxx[rows_bd, cols_bd] = Hblk.reshape(-1)
                bx = cp.stack([_seg(hw * Jcols[d] * rv) for d in range(3)], axis=1).reshape(-1)

                V = n_active
                bp = cp.zeros(V, dtype=cp.float64)
                Hxp_flat = cp.zeros(3 * K * V, dtype=cp.float64)
                pr, pcl, pv = [], [], []
                for a in range(4):
                    col = vloc[:, a]
                    wa = wtsv[:, a]
                    for d in range(3):
                        cp.add.at(Hxp_flat, (3 * segv + d) * V + col, hw * Jcols[d] * wa)
                    cp.add.at(bp, col, hw * wa * rv)
                    for b2 in range(4):
                        pr.append(vloc[:, a])
                        pcl.append(vloc[:, b2])
                        pv.append(hw * wa * wtsv[:, b2])
                Hxp = Hxp_flat.reshape(3 * K, V)
                pp_r, pp_c, pp_v = cp.concatenate(pr), cp.concatenate(pcl), cp.concatenate(pv)
                data_c = 0.5 * float(cp.sum(hw * rv * rv))

        # pose priors (host, exact float64) + SDF fold prior (device)
        pH, pb, pcost = _pose_priors_host(state, cur_poses, K)
        Hxx = Hxx + cp.asarray(pH)
        bx = bx + cp.asarray(pb)
        sdf_c = 0.0
        if n_active > 0:
            rp = phi_d.ravel()[active] - phi0_d.ravel()[active]
            bp = bp + sdf_prior_info * rp
            sdf_c = 0.5 * float(sdf_prior_info * cp.sum(rp * rp))
        cost = data_c + pcost + sdf_c

        if not bool((cp.diag(Hxx) > 0).any()):
            break

        # Schur solve on device
        Hxx_lm = Hxx + lam * cp.eye(3 * K, dtype=cp.float64)
        try:
            if n_active == 0:
                dx = cp.linalg.solve(Hxx_lm, -bx)
            else:
                diagp = sdf_prior_info + lam
                rhs = cp.concatenate([Hxp.T, bp[:, None]], axis=1)
                if gpu_solver == "pcg":
                    # matrix-free-ish: cuSPARSE spmm + Jacobi PCG, no LU factorization
                    Hpp = csp.coo_matrix((pp_v, (pp_r, pp_c)), shape=(n_active, n_active)).tocsr()
                    Hpp = Hpp + diagp * csp.identity(n_active, format="csr")
                    Y = _pcg_spd_multi(cp, Hpp, rhs)
                else:
                    Hpp = csp.coo_matrix((pp_v, (pp_r, pp_c)), shape=(n_active, n_active)).tocsc()
                    Hpp = Hpp + diagp * csp.identity(n_active, format="csc")
                    Y = cspl.splu(Hpp).solve(rhs)
                YH, Yb = Y[:, : 3 * K], Y[:, 3 * K]
                S = Hxx_lm - Hxp @ YH
                dxx = cp.linalg.solve(S, -(bx - Hxp @ Yb))
                dxp = -(Yb + YH @ dxx)
                dx = cp.concatenate([dxx, dxp])
        except Exception:
            lam = min(lam * 10.0, lm_lambda_max)
            continue

        dx_h = cp.asnumpy(dx[: 3 * K])
        saved_poses = list(cur_poses)
        saved_phi = phi_d.copy()
        cur_poses = [
            Pose2(p.x + float(dx_h[3 * i]), p.y + float(dx_h[3 * i + 1]), p.theta + float(dx_h[3 * i + 2]))
            for i, p in enumerate(cur_poses)
        ]
        if n_active > 0:
            pf = phi_d.reshape(-1)
            pf[active] = (pf[active] + dx[3 * K: 3 * K + n_active]).astype(cp.float32)

        trial_cost = data_cost(cp.asarray([[p.x, p.y, p.theta] for p in cur_poses], dtype=cp.float64))
        _, _, tpcost = _pose_priors_host(state, cur_poses, K)
        trial_cost += tpcost + sdf_cost_all_changed()

        if trial_cost < cost:
            step_xy = float(cp.max(cp.hypot(dx[0: 3 * K: 3], dx[1: 3 * K: 3]))) if K else 0.0
            step_th = float(cp.max(cp.abs(dx[2: 3 * K: 3]))) if K else 0.0
            lam = max(lam * 0.5, lm_lambda_min)
            cost = trial_cost
            if step_xy < converge_dx_m and step_th < converge_dtheta_rad:
                converged = True
                break
        else:
            cur_poses = saved_poses
            phi_d = saved_phi
            lam = min(lam * 10.0, lm_lambda_max)
            if lam >= lm_lambda_max:
                break

    tsdf.phi[:] = cp.asnumpy(phi_d)
    out_state = WindowState(
        poses=cur_poses, scans=state.scans, motion_priors=list(state.motion_priors),
        anchor=state.anchor, marg_prior=state.marg_prior,
    )
    return JointWindowResult(
        state=out_state, iterations=iterations, final_cost=cost, converged=converged,
        num_active_voxels=n_active,
        diagnostics={"lm_lambda_final": lam, "inliers_per_scan": list(inliers), "backend": "gpu"},
    )


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
    gpu_solver: str = "pcg",
) -> JointWindowResult:
    """backend="schur" (default) eliminates the SDF block via a sparse Schur complement
    (scales to large active-voxel counts); "dense" builds the full (3K+V) system;
    "schur_gpu" runs the sparse Schur factorization on the GPU (cupyx splu / cuSOLVER).
    All three give the same Gauss-Newton/LM step. backend="gpu" additionally moves the
    gather+assemble onto the device (the actual bottleneck); it does not support the
    SDF smoothness term. For backend="gpu", gpu_solver selects the SDF-block solve:
    "pcg" (default) uses Jacobi-preconditioned CG (cuSPARSE spmm, no factorization --
    H_phiphi is diagonally dominant under the SDF prior); "splu" keeps the cuSOLVER
    sparse-LU factorization for comparison."""
    if backend == "gpu":
        if sdf_smooth_info > 0.0:
            raise NotImplementedError("backend='gpu' does not support sdf_smooth_info > 0")
        return _optimize_window_joint_gpu(
            tsdf=tsdf, state=state, max_iters=max_iters, huber_delta_m=huber_delta_m,
            sdf_prior_info=sdf_prior_info, converge_dx_m=converge_dx_m,
            converge_dtheta_rad=converge_dtheta_rad, lm_lambda_init=lm_lambda_init,
            lm_lambda_min=lm_lambda_min, lm_lambda_max=lm_lambda_max,
            gpu_solver=gpu_solver,
        )
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
        if state.marg_prior is not None:
            Hm, bm, cm = state.marg_prior.blocks(cur_poses[0])
            Hxx[0:3, 0:3] += Hm
            bx[0:3] += bm
            cost += cm
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
        if state.marg_prior is not None:
            c += state.marg_prior.blocks(cur_poses[0])[2]
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
        marg_prior=state.marg_prior,
    )
    return JointWindowResult(
        state=out_state,
        iterations=iterations,
        final_cost=cost,
        converged=converged,
        num_active_voxels=n_active,
        diagnostics={"lm_lambda_final": lam, "inliers_per_scan": list(inliers)},
    )
