from __future__ import annotations

import numpy as np

from slamx.core.scan_ba.joint import optimize_window_joint
from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.scan_ba.window import AnchorPrior, MotionPrior, WindowState, optimize_window
from slamx.core.types import Pose2

from tests.test_scan_ba_align import _raycast_scan


def _cfg() -> Tsdf2DConfig:
    return Tsdf2DConfig(
        resolution_m=0.05, origin_x_m=-2.0, origin_y_m=-2.0, size_x_m=10.0, size_y_m=10.0, truncation_m=0.6
    )


def _gt():
    return [Pose2(2.0 + 0.25 * i, 1.5 + 0.15 * i, 0.05 * i) for i in range(4)]


def _state(poses, scans):
    gt = _gt()
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x,
            delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta,
            info_xy=3.0,
            info_theta=3.0,
        )
        for i in range(3)
    ]
    return WindowState(
        poses=list(poses), scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=poses[0], info_xy=1e5, info_theta=1e5)
    )


def _clean_map(cfg, gt, scans) -> Tsdf2D:
    base = Tsdf2D.zeros(cfg)
    for p, sc in zip(gt, scans):
        update_tsdf_from_scan(base, pose_map=p, points_sensor=sc, weight_inc=1.0, weight_max=100.0)
    return base


def test_joint_refines_map_below_pose_only():
    """With a corrupted SDF, joint pose+SDF BA fits better than pose-only by also
    refining the map, while keeping poses near ground truth."""
    cfg = _cfg()
    gt = _gt()
    scans = [_raycast_scan(p) for p in gt]
    base = _clean_map(cfg, gt, scans)

    rng = np.random.default_rng(0)
    m = base.weight > 0
    noise = rng.normal(0.0, 0.05, size=int(m.sum())).astype(np.float32)

    corrupt = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
    corrupt.phi[m] += noise
    pose_only = optimize_window(tsdf=corrupt, state=_state(gt, scans), max_iters=20, huber_delta_m=0.2)

    corrupt2 = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
    corrupt2.phi[m] += noise
    joint = optimize_window_joint(
        tsdf=corrupt2, state=_state(gt, scans), max_iters=20, huber_delta_m=0.2, sdf_prior_info=5.0
    )

    assert joint.num_active_voxels > 0
    # joint has the extra SDF degrees of freedom -> strictly lower objective
    assert joint.final_cost < pose_only.final_cost
    # the map was actually refined at the active voxels
    assert np.any(np.abs(corrupt2.phi - corrupt.phi) > 1e-6)
    # poses stay anchored near ground truth (no blow-up)
    dev = max(
        max(abs(a.x - b.x), abs(a.y - b.y), abs(a.theta - b.theta)) for a, b in zip(gt, joint.state.poses)
    )
    assert dev < 0.1, f"pose deviation {dev:.4f} too large"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
