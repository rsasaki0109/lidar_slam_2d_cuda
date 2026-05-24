from __future__ import annotations

import numpy as np
import pytest

from slamx.core.scan_ba import Tsdf2DConfig
from slamx.core.scan_ba.window import (
    AnchorPrior,
    MotionPrior,
    WindowState,
    _accumulate_data_block,
    optimize_window,
)
from slamx.core.scan_ba.tsdf import build_tsdf_from_signed_distance
from slamx.core.types import Pose2

from tests.test_scan_ba_align import _l_room_sdf, _raycast_scan

cuda = pytest.importorskip("slamx.core.scan_ba.cuda", reason="cuda module import")


pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device / cupy")


def _tsdf():
    cfg = Tsdf2DConfig(
        resolution_m=0.05,
        origin_x_m=-2.0,
        origin_y_m=-2.0,
        size_x_m=10.0,
        size_y_m=10.0,
        truncation_m=0.6,
    )
    return build_tsdf_from_signed_distance(cfg, _l_room_sdf)


def test_gpu_data_block_matches_cpu():
    tsdf = _tsdf()
    pose = Pose2(2.0, 1.5, 0.3)
    pts = _raycast_scan(pose)

    H_cpu, b_cpu, cost_cpu, n_cpu = _accumulate_data_block(
        tsdf=tsdf, pose=pose, pts=pts, huber_delta_m=0.15
    )

    cp = cuda._cupy()
    phi_d, weight_d = cuda.upload_tsdf(tsdf)
    pts_d = cp.asarray(pts, dtype=cp.float64)
    H_gpu, b_gpu, cost_gpu, n_gpu = cuda.accumulate_data_block_cuda(
        phi_d=phi_d, weight_d=weight_d, cfg=tsdf.cfg, pose=pose, pts_d=pts_d, huber_delta_m=0.15
    )

    assert n_gpu == n_cpu
    np.testing.assert_allclose(H_gpu, H_cpu, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(b_gpu, b_cpu, rtol=1e-9, atol=1e-9)
    assert abs(cost_gpu - cost_cpu) < 1e-9


def test_gpu_data_block_empty_when_off_map():
    tsdf = _tsdf()
    pose = Pose2(0.0, 0.0, 0.0)
    pts = np.array([[100.0, 100.0], [101.0, 99.0]], dtype=np.float64)
    cp = cuda._cupy()
    phi_d, weight_d = cuda.upload_tsdf(tsdf)
    pts_d = cp.asarray(pts, dtype=cp.float64)
    H, b, cost, n = cuda.accumulate_data_block_cuda(
        phi_d=phi_d, weight_d=weight_d, cfg=tsdf.cfg, pose=pose, pts_d=pts_d, huber_delta_m=0.15
    )
    assert n == 0
    assert cost == 0.0
    assert np.allclose(H, 0.0)


def _window_state():
    gt = [Pose2(2.0 + 0.3 * i, 1.5 + 0.2 * i, 0.1 + 0.05 * i) for i in range(4)]
    scans = [_raycast_scan(p) for p in gt]
    init = [Pose2(p.x + 0.08, p.y - 0.06, p.theta + 0.03) for p in gt]
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x,
            delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta,
            info_xy=3.0,
            info_theta=3.0,
        )
        for i in range(len(gt) - 1)
    ]
    return WindowState(poses=init, scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=init[0]))


def test_gpu_window_solve_matches_cpu():
    tsdf = _tsdf()
    cpu = optimize_window(tsdf=tsdf, state=_window_state(), max_iters=25, huber_delta_m=0.15)
    gpu = cuda.optimize_window_cuda(tsdf=tsdf, state=_window_state(), max_iters=25, huber_delta_m=0.15)

    assert gpu.iterations == cpu.iterations
    assert gpu.diagnostics["inliers_per_scan"] == cpu.diagnostics["inliers_per_scan"]
    assert abs(gpu.final_cost - cpu.final_cost) < 1e-9
    for a, b in zip(cpu.state.poses, gpu.state.poses):
        np.testing.assert_allclose([b.x, b.y, b.theta], [a.x, a.y, a.theta], rtol=0, atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
