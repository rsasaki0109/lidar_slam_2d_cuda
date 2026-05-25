"""CUDA (cupy) backend for the scan-BA inner loop.

Mirrors the CPU TSDF sampling and per-scan data-block accumulation
(`Tsdf2D.sample` + `align._accumulate_data_block`) on the GPU so the
fixed-lag window solve can keep the TSDF resident on the device and run
the hot residual/Jacobian/reduction step in parallel.

cupy needs CUDA toolkit headers at JIT time; on this machine they live at
/usr/include, so we set CUDA_PATH if it is not already set.
"""
from __future__ import annotations

import os

import numpy as np

from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.scan_ba.window import WindowResult, WindowState
from slamx.core.types import Pose2


def _detect_cuda_path() -> str | None:
    for p in ("/usr/local/cuda", "/usr"):
        if os.path.exists(os.path.join(p, "include", "cuda_runtime.h")):
            return p
    return None


def _cupy():
    path = _detect_cuda_path()
    if path:
        os.environ.setdefault("CUDA_PATH", path)
    import cupy as cp

    return cp


def is_available() -> bool:
    try:
        cp = _cupy()
        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def upload_tsdf(tsdf: Tsdf2D):
    """Move phi/weight to the device as float64 (returns (phi_d, weight_d))."""
    cp = _cupy()
    return cp.asarray(tsdf.phi, dtype=cp.float64), cp.asarray(tsdf.weight, dtype=cp.float64)


def _sample_cuda(cp, phi_d, weight_d, cfg, xy_d):
    """Bilinear sample phi + gradient at device points xy_d (N,2). Mirrors Tsdf2D.sample."""
    res = float(cfg.resolution_m)
    ox, oy = float(cfg.origin_x_m), float(cfg.origin_y_m)
    h, w = int(phi_d.shape[0]), int(phi_d.shape[1])

    gx = (xy_d[:, 0] - ox) / res - 0.5
    gy = (xy_d[:, 1] - oy) / res - 0.5
    ix0 = cp.floor(gx).astype(cp.int64)
    iy0 = cp.floor(gy).astype(cp.int64)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    in_bounds = (ix0 >= 0) & (ix1 < w) & (iy0 >= 0) & (iy1 < h)

    ix0c = cp.clip(ix0, 0, w - 1)
    ix1c = cp.clip(ix1, 0, w - 1)
    iy0c = cp.clip(iy0, 0, h - 1)
    iy1c = cp.clip(iy1, 0, h - 1)

    p00 = phi_d[iy0c, ix0c]
    p10 = phi_d[iy0c, ix1c]
    p01 = phi_d[iy1c, ix0c]
    p11 = phi_d[iy1c, ix1c]
    w00 = weight_d[iy0c, ix0c]
    w10 = weight_d[iy0c, ix1c]
    w01 = weight_d[iy1c, ix0c]
    w11 = weight_d[iy1c, ix1c]

    fx = gx - ix0
    fy = gy - iy0
    one_fx = 1.0 - fx
    one_fy = 1.0 - fy

    phi_val = one_fx * one_fy * p00 + fx * one_fy * p10 + one_fx * fy * p01 + fx * fy * p11
    dphi_dx = ((-one_fy) * p00 + one_fy * p10 + (-fy) * p01 + fy * p11) / res
    dphi_dy = ((-one_fx) * p00 + (-fx) * p10 + one_fx * p01 + fx * p11) / res

    valid = in_bounds & (w00 > 0) & (w10 > 0) & (w01 > 0) & (w11 > 0)
    return phi_val, dphi_dx, dphi_dy, valid


def accumulate_data_block_cuda(*, phi_d, weight_d, cfg, pose: Pose2, pts_d, huber_delta_m: float):
    """GPU per-scan data block: returns (H_block 3x3, b_block 3, cost, inliers) on host.

    `pts_d` is an (N,2) device array of sensor-frame points.
    """
    cp = _cupy()
    c, s = float(np.cos(pose.theta)), float(np.sin(pose.theta))
    R = cp.asarray([[c, -s], [s, c]], dtype=cp.float64)
    t = cp.asarray([pose.x, pose.y], dtype=cp.float64)
    pts_w = pts_d @ R.T + t

    phi_val, gx, gy, valid = _sample_cuda(cp, phi_d, weight_d, cfg, pts_w)
    n_valid = int(valid.sum().get())
    if n_valid == 0:
        return np.zeros((3, 3)), np.zeros(3), 0.0, 0

    r = phi_val[valid]
    gxv = gx[valid]
    gyv = gy[valid]
    pwx = pts_w[valid, 0]
    pwy = pts_w[valid, 1]
    dtheta = gxv * (-(pwy - pose.y)) + gyv * (pwx - pose.x)
    J = cp.stack([gxv, gyv, dtheta], axis=1)  # (M, 3)

    abs_r = cp.abs(r)
    wts = cp.ones_like(r)
    big = abs_r > huber_delta_m
    wts = cp.where(big, huber_delta_m / cp.maximum(abs_r, 1e-12), wts)

    JtW = J.T * wts
    H = JtW @ J
    b = JtW @ r
    cost = 0.5 * float((wts * r * r).sum().get())
    return cp.asnumpy(H), cp.asnumpy(b), cost, n_valid


# Fused data-term kernel (P2.5c): one thread per window point does warp + bilinear
# sample + Jacobian + Huber, then atomicAdds its 10 contributions (6 upper-tri H + 3 b
# + cost) and inlier flag into its scan's slot of `acc` (K x 11). Collapses the ~11
# bincounts + elementwise ops of the cupy path into a single kernel launch per evaluate.
_DATA_KERNEL_SRC = r"""
extern "C" __global__
void eval_window_data(
    const double* __restrict__ poses,   // K*3
    const double* __restrict__ pts,      // N*2 (sensor frame)
    const long long* __restrict__ seg,   // N   (scan index per point)
    const double* __restrict__ phi,      // H*W
    const double* __restrict__ wt,       // H*W
    double* __restrict__ acc,            // K*11, pre-zeroed
    const int N, const int W, const int H,
    const double res, const double ox, const double oy, const double huber)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int k = (int)seg[idx];
    double px = pts[2*idx], py = pts[2*idx + 1];
    double x = poses[3*k], y = poses[3*k + 1], th = poses[3*k + 2];
    double cs = cos(th), sn = sin(th);
    double pwx = cs*px - sn*py + x;
    double pwy = sn*px + cs*py + y;
    double gx = (pwx - ox)/res - 0.5;
    double gy = (pwy - oy)/res - 0.5;
    int ix0 = (int)floor(gx), iy0 = (int)floor(gy);
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    if (ix0 < 0 || ix1 >= W || iy0 < 0 || iy1 >= H) return;
    double p00 = phi[iy0*W + ix0], p10 = phi[iy0*W + ix1];
    double p01 = phi[iy1*W + ix0], p11 = phi[iy1*W + ix1];
    double w00 = wt[iy0*W + ix0], w10 = wt[iy0*W + ix1];
    double w01 = wt[iy1*W + ix0], w11 = wt[iy1*W + ix1];
    if (!(w00 > 0.0 && w10 > 0.0 && w01 > 0.0 && w11 > 0.0)) return;
    double fx = gx - (double)ix0, fy = gy - (double)iy0;
    double ofx = 1.0 - fx, ofy = 1.0 - fy;
    double r  = ofx*ofy*p00 + fx*ofy*p10 + ofx*fy*p01 + fx*fy*p11;
    double j0 = ((-ofy)*p00 + ofy*p10 + (-fy)*p01 + fy*p11) / res;
    double j1 = ((-ofx)*p00 + (-fx)*p10 + ofx*p01 + fx*p11) / res;
    double j2 = j0*(-(pwy - y)) + j1*(pwx - x);
    double ar = fabs(r);
    double wgt = ar > huber ? huber / fmax(ar, 1e-12) : 1.0;
    double* a = acc + (long long)k * 11;
    atomicAdd(&a[0], wgt*j0*j0);
    atomicAdd(&a[1], wgt*j0*j1);
    atomicAdd(&a[2], wgt*j0*j2);
    atomicAdd(&a[3], wgt*j1*j1);
    atomicAdd(&a[4], wgt*j1*j2);
    atomicAdd(&a[5], wgt*j2*j2);
    atomicAdd(&a[6], wgt*j0*r);
    atomicAdd(&a[7], wgt*j1*r);
    atomicAdd(&a[8], wgt*j2*r);
    atomicAdd(&a[9], 0.5*wgt*r*r);
    atomicAdd(&a[10], 1.0);
}
"""

_kernel_cache: dict = {}


def _data_kernel(cp):
    if "eval" not in _kernel_cache:
        _kernel_cache["eval"] = cp.RawKernel(_DATA_KERNEL_SRC, "eval_window_data")
    return _kernel_cache["eval"]


def optimize_window_cuda(
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
    backend: str = "fused",
) -> WindowResult:
    """Fully on-device fixed-lag window LM solve (P2.5).

    Mirrors `window.optimize_window` but keeps the TSDF, all window scan points,
    poses, and the 3K x 3K normal equations resident on the GPU. Every LM iteration
    evaluates all window points in a single batched pass and assembles/solves the
    block system on device; only the scalar cost (accept/reject) and the final poses
    cross back to the host, instead of the per-scan host syncs of the naive port.

    backend="fused" (default) runs the data term in one custom kernel (P2.5c);
    "bincount" uses the pure-cupy reduction path (kept for comparison/fallback).
    """
    cp = _cupy()
    K = state.k
    phi_d, weight_d = upload_tsdf(tsdf)
    cfg = tsdf.cfg
    n3 = 3 * K

    # window points concatenated with a per-point scan index (segment id)
    sizes = [int(s.shape[0]) for s in state.scans]
    ntot = int(sum(sizes))
    if ntot > 0:
        pts = cp.asarray(np.concatenate(state.scans, axis=0), dtype=cp.float64)
        seg = cp.repeat(cp.arange(K, dtype=cp.int64), cp.asarray(sizes, dtype=cp.int64))
    else:
        pts = cp.zeros((0, 2), dtype=cp.float64)
        seg = cp.zeros((0,), dtype=cp.int64)

    poses = cp.asarray([[p.x, p.y, p.theta] for p in state.poses], dtype=cp.float64)

    has_mp = len(state.motion_priors) > 0
    if has_mp:
        mp_delta = cp.asarray(
            [[m.delta_x, m.delta_y, m.delta_theta] for m in state.motion_priors], dtype=cp.float64
        )
        mp_info = cp.asarray(
            [[m.info_xy, m.info_xy, m.info_theta] for m in state.motion_priors], dtype=cp.float64
        )
    anchor = state.anchor
    if anchor is not None:
        anc_pose = cp.asarray([anchor.pose.x, anchor.pose.y, anchor.pose.theta], dtype=cp.float64)
        anc_info = cp.asarray([anchor.info_xy, anchor.info_xy, anchor.info_theta], dtype=cp.float64)

    # block-diagonal scatter indices for the data-term H blocks (built once)
    base = cp.arange(K, dtype=cp.int64) * 3
    lr = cp.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=cp.int64)
    lc = cp.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=cp.int64)
    rows_bd = (base[:, None] + lr[None, :]).reshape(-1)
    cols_bd = (base[:, None] + lc[None, :]).reshape(-1)

    def _seg_sum(vals):
        return cp.bincount(seg, weights=vals, minlength=K)

    # fused-kernel inputs (data term)
    phi_flat = phi_d.ravel()
    wt_flat = weight_d.ravel()
    grid_h, grid_w = int(phi_d.shape[0]), int(phi_d.shape[1])
    res_f, ox_f, oy_f = float(cfg.resolution_m), float(cfg.origin_x_m), float(cfg.origin_y_m)
    use_fused = backend == "fused"
    kern = _data_kernel(cp) if use_fused else None
    threads = 256
    blocks = (ntot + threads - 1) // threads if ntot > 0 else 1

    def evaluate(P):
        """Assemble (H 3K x 3K, b 3K, cost scalar, inliers K) for poses P (K,3), on device."""
        H = cp.zeros((n3, n3), dtype=cp.float64)
        b = cp.zeros(n3, dtype=cp.float64)
        cost = cp.asarray(0.0, dtype=cp.float64)
        inliers = cp.zeros(K, dtype=cp.float64)

        if ntot > 0:
            if use_fused:
                acc = cp.zeros((K, 11), dtype=cp.float64)
                Pc = cp.ascontiguousarray(P)
                kern(
                    (blocks,),
                    (threads,),
                    (Pc, pts, seg, phi_flat, wt_flat, acc,
                     np.int32(ntot), np.int32(grid_w), np.int32(grid_h),
                     np.float64(res_f), np.float64(ox_f), np.float64(oy_f), np.float64(huber_delta_m)),
                )
                h00, h01, h02, h11, h12, h22 = (acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3], acc[:, 4], acc[:, 5])
                bv = acc[:, 6:9]
                cost = cost + cp.sum(acc[:, 9])
                inliers = acc[:, 10]
            else:
                th = P[seg, 2]
                c, s = cp.cos(th), cp.sin(th)
                tx, ty = P[seg, 0], P[seg, 1]
                px, py = pts[:, 0], pts[:, 1]
                pwx = c * px - s * py + tx
                pwy = s * px + c * py + ty
                xy = cp.stack([pwx, pwy], axis=1)
                r, gx, gy, valid = _sample_cuda(cp, phi_d, weight_d, cfg, xy)
                j2 = gx * (-(pwy - ty)) + gy * (pwx - tx)
                abs_r = cp.abs(r)
                wts = cp.where(abs_r > huber_delta_m, huber_delta_m / cp.maximum(abs_r, 1e-12), 1.0)
                wts = cp.where(valid, wts, 0.0)
                h00 = _seg_sum(wts * gx * gx)
                h01 = _seg_sum(wts * gx * gy)
                h02 = _seg_sum(wts * gx * j2)
                h11 = _seg_sum(wts * gy * gy)
                h12 = _seg_sum(wts * gy * j2)
                h22 = _seg_sum(wts * j2 * j2)
                bv = cp.stack([_seg_sum(wts * gx * r), _seg_sum(wts * gy * r), _seg_sum(wts * j2 * r)], axis=1)
                cost = cost + 0.5 * cp.sum(wts * r * r)
                inliers = _seg_sum(valid.astype(cp.float64))

            Hblk = cp.empty((K, 3, 3), dtype=cp.float64)
            Hblk[:, 0, 0] = h00
            Hblk[:, 0, 1] = Hblk[:, 1, 0] = h01
            Hblk[:, 0, 2] = Hblk[:, 2, 0] = h02
            Hblk[:, 1, 1] = h11
            Hblk[:, 1, 2] = Hblk[:, 2, 1] = h12
            Hblk[:, 2, 2] = h22
            H[rows_bd, cols_bd] = Hblk.reshape(-1)
            b[:] = bv.reshape(-1)

        if has_mp:
            for i in range(K - 1):
                Wd = mp_info[i]
                ri = (P[i + 1] - P[i]) - mp_delta[i]
                Wm = cp.diag(Wd)
                H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += Wm
                H[3 * (i + 1) : 3 * (i + 1) + 3, 3 * (i + 1) : 3 * (i + 1) + 3] += Wm
                H[3 * i : 3 * i + 3, 3 * (i + 1) : 3 * (i + 1) + 3] -= Wm
                H[3 * (i + 1) : 3 * (i + 1) + 3, 3 * i : 3 * i + 3] -= Wm
                b[3 * i : 3 * i + 3] += -Wd * ri
                b[3 * (i + 1) : 3 * (i + 1) + 3] += Wd * ri
                cost = cost + 0.5 * cp.sum(Wd * ri * ri)

        if anchor is not None:
            ri = P[0] - anc_pose
            H[0:3, 0:3] += cp.diag(anc_info)
            b[0:3] += anc_info * ri
            cost = cost + 0.5 * cp.sum(anc_info * ri * ri)

        return H, b, cost, inliers

    eye = cp.eye(n3, dtype=cp.float64)
    cur = poses.copy()
    lam = float(lm_lambda_init)
    iterations = 0
    converged = False
    inliers_dev = cp.zeros(K, dtype=cp.float64)
    cost = float("inf")

    for it in range(max_iters):
        iterations = it + 1
        H, b, cost_d, inliers_dev = evaluate(cur)
        cost = float(cost_d)
        if not bool((cp.diag(H) > 0).any()):
            break

        try:
            dx = -cp.linalg.solve(H + lam * eye, b)
        except Exception:
            lam = min(lam * 10.0, lm_lambda_max)
            continue

        trial = cur + dx.reshape(K, 3)
        _, _, trial_cost_d, _ = evaluate(trial)
        trial_cost = float(trial_cost_d)

        if trial_cost < cost:
            step_xy = float(cp.max(cp.hypot(dx[0::3], dx[1::3])))
            step_theta = float(cp.max(cp.abs(dx[2::3])))
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

    poses_host = cp.asnumpy(cur)
    out_poses = [Pose2(float(row[0]), float(row[1]), float(row[2])) for row in poses_host]
    inliers = [int(v) for v in cp.asnumpy(inliers_dev)]
    result_state = WindowState(
        poses=out_poses,
        scans=state.scans,
        motion_priors=list(state.motion_priors),
        anchor=state.anchor,
    )
    return WindowResult(
        state=result_state,
        iterations=iterations,
        final_cost=cost,
        converged=converged,
        diagnostics={"lm_lambda_final": lam, "inliers_per_scan": inliers, "backend": f"cuda-{backend}"},
    )
