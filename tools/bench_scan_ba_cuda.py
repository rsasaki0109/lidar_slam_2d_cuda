"""Benchmark the fixed-lag window LM solve: CPU (numpy) vs on-device (cupy, P2.5).

Times a full `optimize_window` / `optimize_window_cuda` call (all LM iterations) over
a K-scan window at several per-scan point counts. The CUDA path keeps the TSDF, points,
poses and normal equations resident on the GPU and syncs only the scalar cost per
iteration, so it should win once the per-window point total is large enough to amortise
launch + sync overhead.

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/bench_scan_ba_cuda.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for tests.* helpers

from slamx.core.scan_ba import cuda
from slamx.core.scan_ba.tsdf import Tsdf2DConfig, build_tsdf_from_signed_distance
from slamx.core.scan_ba.window import AnchorPrior, MotionPrior, WindowState, optimize_window
from slamx.core.types import Pose2
from tests.test_scan_ba_align import _l_room_sdf, _raycast_scan


def _tile(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] >= n:
        return pts[:n].copy()
    reps = int(np.ceil(n / pts.shape[0]))
    return np.tile(pts, (reps, 1))[:n].copy()


def _make_window(tsdf, *, k: int, n_per_scan: int) -> WindowState:
    gt = [Pose2(2.0 + 0.3 * i, 1.5 + 0.2 * i, 0.1 + 0.05 * i) for i in range(k)]
    scans = [_tile(_raycast_scan(p), n_per_scan) for p in gt]
    init = [Pose2(p.x + 0.08, p.y - 0.06, p.theta + 0.03) for p in gt]
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x,
            delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta,
            info_xy=3.0,
            info_theta=3.0,
        )
        for i in range(k - 1)
    ]
    return WindowState(poses=init, scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=init[0]))


def _time(fn, *, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    cfg = Tsdf2DConfig(
        resolution_m=0.05, origin_x_m=-2.0, origin_y_m=-2.0, size_x_m=10.0, size_y_m=10.0, truncation_m=0.6
    )
    tsdf = build_tsdf_from_signed_distance(cfg, _l_room_sdf)
    k = 10
    has_gpu = cuda.is_available()
    print(f"window K={k}, CUDA={'yes' if has_gpu else 'no'}")
    print(f"{'pts/scan':>9} {'window pts':>11} {'CPU ms':>9} {'GPU ms':>9} {'speedup':>8}")
    for n_per in (200, 1000, 5000, 20000):
        st = _make_window(tsdf, k=k, n_per_scan=n_per)
        cpu_ms = _time(lambda: optimize_window(tsdf=tsdf, state=st, max_iters=25), repeats=5) * 1e3
        gpu_ms = float("nan")
        if has_gpu:
            cuda.optimize_window_cuda(tsdf=tsdf, state=st, max_iters=25)  # warm up JIT
            gpu_ms = _time(lambda: cuda.optimize_window_cuda(tsdf=tsdf, state=st, max_iters=25), repeats=5) * 1e3
        sp = cpu_ms / gpu_ms if gpu_ms == gpu_ms else float("nan")
        print(f"{n_per:>9} {k * n_per:>11} {cpu_ms:>9.2f} {gpu_ms:>9.2f} {sp:>7.2f}x")


if __name__ == "__main__":
    main()
