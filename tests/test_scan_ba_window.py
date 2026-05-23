from __future__ import annotations

import math

import pytest

from slamx.core.scan_ba import (
    AnchorPrior,
    MotionPrior,
    Tsdf2DConfig,
    WindowState,
    optimize_window,
    slide_window,
)
from slamx.core.scan_ba.tsdf import build_tsdf_from_signed_distance
from slamx.core.types import Pose2

from tests.test_scan_ba_align import _l_room_sdf, _raycast_scan


def _make_l_room_tsdf():
    cfg = Tsdf2DConfig(
        resolution_m=0.05,
        origin_x_m=-2.0,
        origin_y_m=-2.0,
        size_x_m=10.0,
        size_y_m=10.0,
        truncation_m=0.6,
    )
    return build_tsdf_from_signed_distance(cfg, _l_room_sdf)


def _max_pose_err(actual: list[Pose2], gt: list[Pose2]) -> tuple[float, float]:
    xy = max(math.hypot(a.x - g.x, a.y - g.y) for a, g in zip(actual, gt))
    th = max(abs(a.theta - g.theta) for a, g in zip(actual, gt))
    return xy, th


def test_window_recovers_three_poses():
    tsdf = _make_l_room_tsdf()
    gt = [
        Pose2(1.0, 0.5, 0.05),
        Pose2(1.4, 0.7, 0.15),
        Pose2(1.8, 0.9, 0.25),
    ]
    scans = [_raycast_scan(p) for p in gt]
    init = [
        Pose2(gt[0].x + 0.15, gt[0].y - 0.10, gt[0].theta - 0.05),
        Pose2(gt[1].x - 0.12, gt[1].y + 0.13, gt[1].theta + 0.06),
        Pose2(gt[2].x + 0.10, gt[2].y - 0.08, gt[2].theta - 0.04),
    ]
    # weak motion priors using the *initial-guess* deltas (no privileged info)
    mps = [
        MotionPrior(
            delta_x=init[i + 1].x - init[i].x,
            delta_y=init[i + 1].y - init[i].y,
            delta_theta=init[i + 1].theta - init[i].theta,
            info_xy=1.0,
            info_theta=1.0,
        )
        for i in range(2)
    ]
    state = WindowState(poses=init, scans=scans, motion_priors=mps, anchor=None)
    result = optimize_window(tsdf=tsdf, state=state, max_iters=60)

    assert result.converged, f"failed: {result.diagnostics}"
    xy_err, th_err = _max_pose_err(result.state.poses, gt)
    assert xy_err < 0.025, f"xy max err {xy_err:.4f}"
    assert th_err < 0.01, f"theta max err {th_err:.4f}"


def test_anchor_pins_first_pose():
    tsdf = _make_l_room_tsdf()
    gt = [Pose2(1.0, 0.5, 0.0), Pose2(1.4, 0.7, 0.1)]
    scans = [_raycast_scan(p) for p in gt]
    # initial guess far off for pose 1
    init = [
        Pose2(gt[0].x, gt[0].y, gt[0].theta),  # close to GT
        Pose2(gt[1].x + 0.2, gt[1].y - 0.15, gt[1].theta - 0.08),
    ]
    # strong anchor pinning pose 0 at an *offset* location: optimizer must keep pose 0 near anchor
    offset_anchor_pose = Pose2(gt[0].x + 0.3, gt[0].y - 0.2, gt[0].theta + 0.05)
    anchor = AnchorPrior(pose=offset_anchor_pose, info_xy=1.0e8, info_theta=1.0e8)
    state = WindowState(
        poses=init,
        scans=scans,
        motion_priors=[
            MotionPrior(
                delta_x=gt[1].x - gt[0].x,
                delta_y=gt[1].y - gt[0].y,
                delta_theta=gt[1].theta - gt[0].theta,
                info_xy=0.1,
                info_theta=0.1,
            )
        ],
        anchor=anchor,
    )
    result = optimize_window(tsdf=tsdf, state=state, max_iters=60)

    # pose 0 should be pulled to the anchor (not GT)
    p0 = result.state.poses[0]
    assert math.hypot(p0.x - offset_anchor_pose.x, p0.y - offset_anchor_pose.y) < 0.01
    assert abs(p0.theta - offset_anchor_pose.theta) < 0.005

    # pose 1 has free data; should still converge near its GT
    p1 = result.state.poses[1]
    assert math.hypot(p1.x - gt[1].x, p1.y - gt[1].y) < 0.05


def test_slide_window_drops_oldest_and_bakes_anchor():
    poses = [Pose2(1.0, 0.5, 0.0), Pose2(1.2, 0.6, 0.05), Pose2(1.4, 0.7, 0.1)]
    scans = [_raycast_scan(p) for p in poses]
    mps = [MotionPrior() for _ in range(2)]
    state = WindowState(poses=poses, scans=scans, motion_priors=mps, anchor=None)

    new_pose = Pose2(1.6, 0.8, 0.15)
    new_scan = _raycast_scan(new_pose)
    new_mp = MotionPrior(delta_x=0.2, delta_y=0.1, delta_theta=0.05)

    slid = slide_window(state, new_scan=new_scan, new_pose_init=new_pose, new_motion_prior=new_mp)

    assert slid.k == 3
    # surviving oldest should be old poses[1]
    assert slid.poses[0].x == pytest.approx(poses[1].x)
    assert slid.poses[2].x == pytest.approx(new_pose.x)
    assert slid.anchor is not None
    assert slid.anchor.pose.x == pytest.approx(poses[1].x)
    # motion prior list shifts left by 1, with new prior appended
    assert len(slid.motion_priors) == 2
    assert slid.motion_priors[-1] is new_mp


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
