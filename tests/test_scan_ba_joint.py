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


def test_joint_schur_matches_dense():
    """Sparse Schur elimination of the SDF block gives the same step as the dense
    full (3K+V) solve. Few beams keep the dense path fast."""
    cfg = _cfg()
    gt = _gt()[:2]
    scans = [_raycast_scan(p, n_beams=60) for p in gt]
    base = _clean_map(cfg, gt, scans)
    rng = np.random.default_rng(1)
    m = base.weight > 0
    noise = rng.normal(0.0, 0.04, size=int(m.sum())).astype(np.float32)

    def state():
        mps = [
            MotionPrior(
                delta_x=gt[1].x - gt[0].x, delta_y=gt[1].y - gt[0].y, delta_theta=gt[1].theta - gt[0].theta,
                info_xy=3.0, info_theta=3.0,
            )
        ]
        return WindowState(poses=list(gt), scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=gt[0]))

    td = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
    td.phi[m] += noise
    rd = optimize_window_joint(tsdf=td, state=state(), max_iters=10, huber_delta_m=0.2, backend="dense")

    ts = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
    ts.phi[m] += noise
    rs = optimize_window_joint(tsdf=ts, state=state(), max_iters=10, huber_delta_m=0.2, backend="schur")

    assert abs(rd.final_cost - rs.final_cost) < 1e-9
    for a, b in zip(rd.state.poses, rs.state.poses):
        np.testing.assert_allclose([b.x, b.y, b.theta], [a.x, a.y, a.theta], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ts.phi, td.phi, rtol=0, atol=1e-6)


def _tv(t: Tsdf2D) -> float:
    p = t.phi
    mm = t.weight > 0
    dx = np.abs(np.diff(p, axis=1))
    dy = np.abs(np.diff(p, axis=0))
    return float(dx[mm[:, 1:] & mm[:, :-1]].sum() + dy[mm[1:, :] & mm[:-1, :]].sum())


def test_joint_smoothness_term():
    """The SDF smoothness regulariser (a) keeps schur == dense and (b) reduces the
    refined map's roughness (total variation) vs no smoothing."""
    cfg = _cfg()
    gt = _gt()[:2]
    scans = [_raycast_scan(p, n_beams=80) for p in gt]
    base = _clean_map(cfg, gt, scans)
    rng = np.random.default_rng(2)
    m = base.weight > 0
    noise = rng.normal(0.0, 0.06, size=int(m.sum())).astype(np.float32)

    def state():
        mps = [
            MotionPrior(
                delta_x=gt[1].x - gt[0].x, delta_y=gt[1].y - gt[0].y, delta_theta=gt[1].theta - gt[0].theta,
                info_xy=3.0, info_theta=3.0,
            )
        ]
        return WindowState(poses=list(gt), scans=scans, motion_priors=mps, anchor=AnchorPrior(pose=gt[0]))

    def fresh():
        t = Tsdf2D(cfg=cfg, phi=base.phi.copy(), weight=base.weight.copy())
        t.phi[m] += noise
        return t

    ts = fresh()
    rs = optimize_window_joint(tsdf=ts, state=state(), max_iters=12, huber_delta_m=0.2, sdf_smooth_info=2.0, backend="schur")
    td = fresh()
    rd = optimize_window_joint(tsdf=td, state=state(), max_iters=12, huber_delta_m=0.2, sdf_smooth_info=2.0, backend="dense")
    assert abs(rs.final_cost - rd.final_cost) < 1e-9
    np.testing.assert_allclose(ts.phi, td.phi, rtol=0, atol=1e-6)

    tn = fresh()
    optimize_window_joint(tsdf=tn, state=state(), max_iters=12, huber_delta_m=0.2, sdf_smooth_info=0.0)
    assert _tv(ts) < _tv(tn), "smoothing should reduce map total variation"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
