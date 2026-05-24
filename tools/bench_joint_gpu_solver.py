"""Benchmark the joint pose+SDF GPU assemble: cupy vectorized vs fused RawKernel (P3.7).

After P3.6 removed the linear-solve wall (splu -> Jacobi-PCG), profiling the full-GPU
joint window solve (`backend="gpu"`) showed the bottleneck had moved to the gather +
assemble -- specifically the 6 per-block `bincount` reductions and the 16 `cp.add.at`
scatters that build the data block of the normal equations (Hxx / b_x / H_xphi / b_phi),
each a launch-bound sorted/atomic pass. P3.7 fuses all of them into a single atomicAdd
RawKernel (`gpu_assemble="fused"`, default).

This times a full `optimize_window_joint(backend="gpu", gpu_solver="pcg")` (all LM
iterations) for gpu_assemble="vectorized" vs "fused" across a range of active-voxel
counts, reporting wall ms/solve and the speedup. Both produce the same GN/LM step to
round-off (pinned by test_joint_gpu_fused_matches_vectorized).

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


def _time(cfg, base, state, m, noise, solver: str, reps: int, assemble: str = "fused"):
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
                                    backend="gpu", gpu_solver=solver, gpu_assemble=assemble)
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
    print(f"joint GPU window solve (backend='gpu', pcg), K={K}, max_iters=12, beams={n_beams}, best of {reps}")
    print("end-to-end ms/solve (all LM iters): vectorized assemble (bincount+add.at) vs fused RawKernel (P3.7)")
    print(f"{'res_m':>6} {'trunc':>6} {'V_active':>9} {'iters':>6} {'vec ms':>9} {'fused ms':>9} {'speedup':>8}")
    # finer resolution + wider truncation drive the active-voxel count up.
    for res, trunc in ((0.04, 0.6), (0.03, 0.8), (0.02, 1.0), (0.015, 1.2)):
        cfg = _cfg(res, trunc)
        base, state, m, noise = _build(cfg, K, n_beams)
        ms_vec, V, it = _time(cfg, base, state, m, noise, "pcg", reps, assemble="vectorized")
        ms_fus, V2, it2 = _time(cfg, base, state, m, noise, "pcg", reps, assemble="fused")
        assert V == V2, (V, V2)
        print(f"{res:>6.3f} {trunc:>6.2f} {V:>9} {it:>6} {ms_vec:>9.1f} {ms_fus:>9.1f} {ms_vec / ms_fus:>7.2f}x")


if __name__ == "__main__":
    main()
