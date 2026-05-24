from __future__ import annotations

import numpy as np

from slamx.core.scan_ba.marginalize import (
    MarginalizationPrior,
    marginalize_oldest_pose,
    schur_complement,
)
from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.scan_ba.window import (
    AnchorPrior,
    MotionPrior,
    WindowState,
    _evaluate,
    optimize_window,
    slide_window,
)
from slamx.core.types import Pose2

from tests.test_scan_ba_align import _raycast_scan


def test_schur_complement_matches_full_linear_solve():
    """Marginalizing variables from a dense SPD system and solving the reduced system
    gives the exact same step for the retained variables as the full solve."""
    rng = np.random.default_rng(0)
    n = 9  # 3 pose blocks
    A = rng.normal(size=(n, n))
    H = A @ A.T + n * np.eye(n)  # SPD
    b = rng.normal(size=n)

    dx_full = -np.linalg.solve(H, b)
    Lambda, g = schur_complement(H, b, m=3)
    dx_red = -np.linalg.solve(Lambda, g)
    np.testing.assert_allclose(dx_red, dx_full[3:], rtol=0, atol=1e-10)


def _cfg() -> Tsdf2DConfig:
    return Tsdf2DConfig(
        resolution_m=0.05, origin_x_m=-2.0, origin_y_m=-2.0, size_x_m=10.0, size_y_m=10.0, truncation_m=0.6
    )


def _map(cfg, gt, scans) -> Tsdf2D:
    t = Tsdf2D.zeros(cfg)
    for p, sc in zip(gt, scans):
        update_tsdf_from_scan(t, pose_map=p, points_sensor=sc, weight_inc=1.0, weight_max=100.0)
    return t


def test_marginalize_oldest_matches_full_window_step():
    """One Gauss-Newton step on the retained poses is identical whether we (a) solve
    the full K-pose window and keep the retained part, or (b) eliminate pose 0 into a
    marginal prior and solve the reduced window. This is the exact-marginalization
    guarantee against the real data + motion + anchor machinery."""
    cfg = _cfg()
    gt = [Pose2(2.0 + 0.25 * i, 1.5 + 0.12 * i, 0.04 * i) for i in range(3)]
    scans = [_raycast_scan(p, n_beams=120) for p in gt]
    tsdf = _map(cfg, gt, scans)

    # perturb away from the optimum so the GN step is non-trivial
    poses = [Pose2(gt[0].x + 0.05, gt[0].y - 0.03, gt[0].theta + 0.02),
             Pose2(gt[1].x - 0.04, gt[1].y + 0.05, gt[1].theta - 0.01),
             Pose2(gt[2].x + 0.03, gt[2].y + 0.02, gt[2].theta + 0.015)]
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x, delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta, info_xy=4.0, info_theta=4.0,
        )
        for i in range(2)
    ]
    anchor = AnchorPrior(pose=gt[0], info_xy=50.0, info_theta=50.0)
    hub = 0.3

    full = WindowState(poses=poses, scans=scans, motion_priors=mps, anchor=anchor)
    H, b, _, _ = _evaluate(full, tsdf, hub)
    dx_full = -np.linalg.solve(H, b)

    marg = marginalize_oldest_pose(
        tsdf=tsdf, poses=poses, scans=scans, motion_prior=mps[0], huber_delta_m=hub, anchor=anchor,
    )
    reduced = WindowState(poses=poses[1:], scans=scans[1:], motion_priors=mps[1:], anchor=None, marg_prior=marg)
    Hr, br, _, _ = _evaluate(reduced, tsdf, hub)
    dx_red = -np.linalg.solve(Hr, br)

    np.testing.assert_allclose(dx_red, dx_full[3:], rtol=0, atol=1e-9)


def test_recursive_marginalization_matches_full_window():
    """Marginalizing twice (eliminate pose 0, then pose 1 carrying the first marginal
    prior) reproduces the full 4-pose window's step for the two surviving poses."""
    cfg = _cfg()
    gt = [Pose2(2.0 + 0.22 * i, 1.5 + 0.1 * i, 0.03 * i) for i in range(4)]
    scans = [_raycast_scan(p, n_beams=120) for p in gt]
    tsdf = _map(cfg, gt, scans)

    poses = [
        Pose2(gt[0].x + 0.04, gt[0].y - 0.02, gt[0].theta + 0.015),
        Pose2(gt[1].x - 0.03, gt[1].y + 0.04, gt[1].theta - 0.01),
        Pose2(gt[2].x + 0.02, gt[2].y + 0.03, gt[2].theta + 0.012),
        Pose2(gt[3].x - 0.025, gt[3].y + 0.02, gt[3].theta - 0.008),
    ]
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x, delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta, info_xy=4.0, info_theta=4.0,
        )
        for i in range(3)
    ]
    anchor = AnchorPrior(pose=gt[0], info_xy=50.0, info_theta=50.0)
    hub = 0.3

    full = WindowState(poses=poses, scans=scans, motion_priors=mps, anchor=anchor)
    H, b, _, _ = _evaluate(full, tsdf, hub)
    dx_full = -np.linalg.solve(H, b)

    m1 = marginalize_oldest_pose(
        tsdf=tsdf, poses=poses, scans=scans, motion_prior=mps[0], huber_delta_m=hub, anchor=anchor,
    )
    m2 = marginalize_oldest_pose(
        tsdf=tsdf, poses=poses[1:], scans=scans[1:], motion_prior=mps[1], huber_delta_m=hub,
        anchor=None, prev_marg=m1,
    )
    reduced = WindowState(poses=poses[2:], scans=scans[2:], motion_priors=mps[2:], anchor=None, marg_prior=m2)
    Hr, br, _, _ = _evaluate(reduced, tsdf, hub)
    dx_red = -np.linalg.solve(Hr, br)

    np.testing.assert_allclose(dx_red, dx_full[6:], rtol=0, atol=1e-9)


def test_window_marg_prior_threads_through_optimize():
    """optimize_window must carry marg_prior through its accept/reject and converge to
    the prior's linearization point when no data/motion factors compete."""
    cfg = _cfg()
    x_lin = np.array([1.0, 2.0, 0.1])
    Lambda = np.diag([100.0, 100.0, 100.0])
    g = np.zeros(3)
    marg = MarginalizationPrior(x_lin=x_lin, Lambda=Lambda, g=g)

    empty = np.zeros((0, 2))
    state = WindowState(poses=[Pose2(0.0, 0.0, 0.0)], scans=[empty], motion_priors=[], anchor=None, marg_prior=marg)
    res = optimize_window(tsdf=Tsdf2D.zeros(cfg), state=state, max_iters=20)
    p = res.state.poses[0]
    np.testing.assert_allclose([p.x, p.y, p.theta], x_lin, rtol=0, atol=1e-6)


def test_slide_window_marginalize_produces_prior():
    """slide_window(marginalize=True) drops the anchor and attaches a MarginalizationPrior
    on the new oldest pose, matching a direct marginalize_oldest_pose call."""
    cfg = _cfg()
    gt = [Pose2(2.0 + 0.25 * i, 1.5 + 0.12 * i, 0.04 * i) for i in range(3)]
    scans = [_raycast_scan(p, n_beams=120) for p in gt]
    tsdf = _map(cfg, gt, scans)
    mps = [
        MotionPrior(
            delta_x=gt[i + 1].x - gt[i].x, delta_y=gt[i + 1].y - gt[i].y,
            delta_theta=gt[i + 1].theta - gt[i].theta, info_xy=4.0, info_theta=4.0,
        )
        for i in range(2)
    ]
    anchor = AnchorPrior(pose=gt[0], info_xy=50.0, info_theta=50.0)
    state = WindowState(poses=list(gt), scans=scans, motion_priors=mps, anchor=anchor)

    new_pose = Pose2(gt[2].x + 0.25, gt[2].y + 0.12, gt[2].theta + 0.04)
    new_scan = _raycast_scan(new_pose, n_beams=120)
    new_mp = MotionPrior(delta_x=0.25, delta_y=0.12, delta_theta=0.04, info_xy=4.0, info_theta=4.0)

    out = slide_window(
        state, new_scan=new_scan, new_pose_init=new_pose, new_motion_prior=new_mp,
        marginalize=True, tsdf=tsdf, huber_delta_m=0.3,
    )
    assert out.anchor is None
    assert out.marg_prior is not None
    assert out.k == 3

    direct = marginalize_oldest_pose(
        tsdf=tsdf, poses=list(gt), scans=scans, motion_prior=mps[0], huber_delta_m=0.3, anchor=anchor,
    )
    np.testing.assert_allclose(out.marg_prior.Lambda, direct.Lambda, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.marg_prior.g, direct.g, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.marg_prior.x_lin, direct.x_lin, rtol=0, atol=1e-12)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
