"""Benchmark the joint pose+SDF GPU solve: cuSOLVER sparse-LU vs Jacobi-PCG (P3.6).

The full-GPU joint window solve (`backend="gpu"`) was profiled (P3.5) to be bound by
the SDF-block linear solve: cuSOLVER sparse-LU factorization (`cupyx.scipy.sparse.linalg.splu`)
dominated each LM iteration. H_phiphi is SPD and, under the SDF prior, strongly diagonally
dominant -- so a Jacobi-preconditioned CG (cuSPARSE spmm + reductions, no factorization)
should converge in a handful of iterations and remove that wall.

This times a full `optimize_window_joint(backend="gpu", ...)` (all LM iterations) for both
gpu_solver="splu" and gpu_solver="pcg" across a range of active-voxel counts (driven by
the beam count), reporting wall ms/solve and the speedup. Both produce the same GN/LM step
(pinned by test_joint_gpu_pcg_matches_splu).

Usage:
  CUDA_PATH=/usr env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/bench_joint_gpu_solver.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for tests.* helpers

from slamx.core.scan_ba.joint import optimize_window_joint
from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.scan_ba.window import AnchorPrior, MotionPrior, WindowState
from slamx.core.types import Pose2
from tests.test_scan_ba_align import _raycast_scan


def _cfg(res: float, trunc: float) -> Tsdf2DConfig:
    return Tsdf2DConfig(
        resolution_m=res, origin_x_m=-2.0, origin_y_m=-2.0, size_x_m=12.0, size_y_m=12.0, truncation_m=trunc
    )


def _gt(k: int):
    return [Pose2(2.0 + 0.25 * i, 1.5 + 0.15 * i, 0.05 * i) for i in range(k)]


def _build(cfg, k: int, n_beams: int):
    gt = _gt(k)
    scans = [_raycast_scan(p, n_beams=n_beams) for p in gt]
    base = Tsdf2D.zeros(cfg)
    for p, sc in zip(gt, scans):
        update_tsdf_from_scan(base, pose_map=p, points_sensor=sc, weight_inc=1.0, weight_max=100.0)
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x, delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta, info_xy=3.0, info_theta=3.0,
        )
        for i in range(k - 1)
    ]
    state = WindowState(poses=list(gt), scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=gt[0]))
    m = base.weight > 0
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.05, size=int(m.sum())).astype(np.float32)
    return base, state, m, noise


def _time(cfg, base, state, m, noise, solver: str, reps: int):
    import cupy as cp

    n_active = 0
    iters = 0
    best = float("inf")
    for r in range(reps + 1):  # one warmup
        t = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
        t.phi[m] += noise
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        res = optimize_window_joint(tsdf=t, state=state, max_iters=12, huber_delta_m=0.2,
                                    backend="gpu", gpu_solver=solver)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        if r > 0:
            best = min(best, dt)
        n_active, iters = res.num_active_voxels, res.iterations
    return best * 1e3, n_active, iters


def main() -> None:
    K = 5
    reps = 3
    n_beams = 1080
    print(f"joint GPU solve: splu (cuSOLVER LU) vs pcg (Jacobi-CG)  K={K}, max_iters=12, beams={n_beams}, best of {reps}")
    print(f"{'res_m':>6} {'trunc':>6} {'V_active':>9} {'iters':>6} {'splu ms':>9} {'pcg ms':>9} {'speedup':>8}")
    # finer resolution + wider truncation drive the active-voxel count (the size of the
    # SDF block being factorized) up into the regime where LU factorization hurts.
    for res, trunc in ((0.04, 0.6), (0.03, 0.8), (0.02, 1.0), (0.015, 1.2)):
        cfg = _cfg(res, trunc)
        base, state, m, noise = _build(cfg, K, n_beams)
        ms_lu, V, it = _time(cfg, base, state, m, noise, "splu", reps)
        ms_cg, V2, it2 = _time(cfg, base, state, m, noise, "pcg", reps)
        assert V == V2, (V, V2)
        print(f"{res:>6.3f} {trunc:>6.2f} {V:>9} {it:>6} {ms_lu:>9.1f} {ms_cg:>9.1f} {ms_lu / ms_cg:>7.2f}x")


if __name__ == "__main__":
    main()
