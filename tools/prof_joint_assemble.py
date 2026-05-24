"""Profile the sub-steps of the full-GPU joint assemble (P3.7 target finding).

P3.6 removed the linear-solve wall (splu -> Jacobi-PCG); the running lesson is that
the bottleneck has moved to gather+assemble. Before fusing a kernel we must know which
sub-step actually dominates. This replicates the inner assemble block of
`_optimize_window_joint_gpu` and times each piece (bilinear gather, per-scan reductions,
the cp.add.at scatters for Hxp/bp, the H_phiphi triplet build + tocsr, and the PCG solve)
with stream synchronization, best-of-reps.

Usage:
  CUDA_PATH=/usr env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/prof_joint_assemble.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp  # noqa: E402

import cupyx.scipy.sparse as csp  # noqa: E402

from slamx.core.scan_ba.joint import _bilinear_terms_gpu, _pcg_spd_multi  # noqa: E402
from tools.bench_joint_gpu_solver import _build, _cfg  # noqa: E402


def _sync_time(fn, reps=5):
    best = float("inf")
    out = None
    for r in range(reps + 1):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        out = fn()
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        if r > 0:
            best = min(best, dt)
    return best * 1e3, out


def profile(res, trunc, K=5, n_beams=1080, sdf_prior_info=10.0, lam=1e-3):
    cfg = _cfg(res, trunc)
    base, state, m, noise = _build(cfg, K, n_beams)
    phi_d = cp.asarray(base.phi, dtype=cp.float32)
    phi_d.reshape(-1)[cp.asarray(np.flatnonzero(m))] += cp.asarray(noise)
    wt_d = cp.asarray(base.weight, dtype=cp.float32)

    sizes = [s.shape[0] for s in state.scans]
    pts_d = cp.asarray(np.concatenate(state.scans, axis=0), dtype=cp.float64)
    seg_all = cp.repeat(cp.arange(K, dtype=cp.int64), cp.asarray(sizes, dtype=cp.int64))
    P = cp.asarray([[p.x, p.y, p.theta] for p in state.poses], dtype=cp.float64)

    th = P[:, 2][seg_all]
    c, s = cp.cos(th), cp.sin(th)
    tx, ty = P[:, 0][seg_all], P[:, 1][seg_all]
    pwx = c * pts_d[:, 0] - s * pts_d[:, 1] + tx
    pwy = s * pts_d[:, 0] + c * pts_d[:, 1] + ty
    pw = cp.stack([pwx, pwy], axis=1)

    # --- bilinear gather ---
    def _gather():
        return _bilinear_terms_gpu(cp, phi_d, wt_d, cfg, pw)

    t_gather, (r, gx, gy, valid, neigh, wts) = _sync_time(_gather)

    segv = seg_all[valid]
    rv, gxv, gyv = r[valid], gx[valid], gy[valid]
    pwyv, pwxv, txv, tyv = pwy[valid], pwx[valid], tx[valid], ty[valid]
    neighv, wtsv = neigh[valid], wts[valid]
    hw = cp.where(cp.abs(rv) > 0.2, 0.2 / cp.maximum(cp.abs(rv), 1e-12), 1.0)
    j2 = gxv * (-(pwyv - tyv)) + gyv * (pwxv - txv)
    Jcols = (gxv, gyv, j2)
    active = cp.unique(neighv.ravel())
    V = int(active.size)
    vloc = cp.searchsorted(active, neighv)
    M = int(rv.size)

    # --- pose-block reductions (Hxx, bx) ---
    def _poseblk():
        h = [cp.bincount(segv, weights=hw * Jcols[i] * Jcols[j], minlength=K)
             for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
        bx = cp.stack([cp.bincount(segv, weights=hw * Jcols[d] * rv, minlength=K) for d in range(3)], axis=1)
        return h, bx

    t_poseblk, _ = _sync_time(_poseblk)

    # --- Hxp + bp scatter via cp.add.at (vectorized path) ---
    def _scatter_hxpbp():
        bp = cp.zeros(V, dtype=cp.float64)
        Hxp_flat = cp.zeros(3 * K * V, dtype=cp.float64)
        for a in range(4):
            col = vloc[:, a]
            wa = wtsv[:, a]
            for d in range(3):
                cp.add.at(Hxp_flat, (3 * segv + d) * V + col, hw * Jcols[d] * wa)
            cp.add.at(bp, col, hw * wa * rv)
        return Hxp_flat, bp

    t_scatter, (Hxp_flat, bp) = _sync_time(_scatter_hxpbp)
    Hxp = Hxp_flat.reshape(3 * K, V)

    # --- fused RawKernel: pose-block reduce + Hxp + bp in one launch (P3.7) ---
    from slamx.core.scan_ba.joint import _fused_assemble_gpu

    def _fused():
        return _fused_assemble_gpu(cp, segv, hw, rv, Jcols, vloc, wtsv, K, V)

    t_fused, _ = _sync_time(_fused)

    # --- H_phiphi triplet build ---
    def _triplets():
        pr, pcl, pv = [], [], []
        for a in range(4):
            wa = wtsv[:, a]
            for b2 in range(4):
                pr.append(vloc[:, a])
                pcl.append(vloc[:, b2])
                pv.append(hw * wa * wtsv[:, b2])
        return cp.concatenate(pr), cp.concatenate(pcl), cp.concatenate(pv)

    t_trip, (pp_r, pp_c, pp_v) = _sync_time(_triplets)

    # --- COO -> CSR coalesce (+ prior diagonal) ---
    diagp = sdf_prior_info + lam

    def _tocsr():
        Hpp = csp.coo_matrix((pp_v, (pp_r, pp_c)), shape=(V, V)).tocsr()
        return Hpp + diagp * csp.identity(V, format="csr")

    t_csr, Hpp = _sync_time(_tocsr)

    # --- PCG solve (multi-RHS) ---
    rhs = cp.concatenate([Hxp.T, bp[:, None]], axis=1)

    def _solve():
        return _pcg_spd_multi(cp, Hpp, rhs)

    t_solve, _ = _sync_time(_solve)

    # vectorized assemble = poseblk + scatter; fused = the single RawKernel
    total = t_gather + t_poseblk + t_scatter + t_trip + t_csr + t_solve
    total_fused = t_gather + t_fused + t_trip + t_csr + t_solve
    print(f"\n=== res={res} trunc={trunc}  M(valid pts)={M}  V_active={V}  3K+1 RHS={3*K+1} ===")
    rows = [
        ("bilinear gather", t_gather),
        ("pose-block reduce (Hxx,bx)", t_poseblk),
        ("Hxp+bp scatter (16x add.at)", t_scatter),
        ("  -> fused assemble RawKernel", t_fused),
        ("H_phiphi triplet build", t_trip),
        ("COO->CSR coalesce", t_csr),
        ("PCG solve", t_solve),
    ]
    for name, ms in rows:
        print(f"  {name:<32} {ms:8.2f} ms")
    print(f"  {'TOTAL vectorized (assemble+solve)':<32} {total:8.2f} ms")
    print(f"  {'TOTAL fused     (assemble+solve)':<32} {total_fused:8.2f} ms  "
          f"({total / total_fused:.2f}x)")
    print(f"  assemble only: poseblk+scatter {t_poseblk + t_scatter:.2f} ms "
          f"-> fused {t_fused:.2f} ms  ({(t_poseblk + t_scatter) / t_fused:.1f}x)")


def main():
    for res, trunc in ((0.03, 0.8), (0.02, 1.0), (0.015, 1.2)):
        profile(res, trunc)


if __name__ == "__main__":
    main()
