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
