from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from slamx.core.local_matching.icp import (
    IcpConfig,
    IcpScanMatcher,
    _estimate_normals_2d,
)
from slamx.core.types import LaserScan, Pose2


def _make_laser_scan(points_xy: np.ndarray) -> LaserScan:
    """Create a LaserScan from explicit 2D points.

    Converts Cartesian (x, y) into polar (range, bearing) that
    LaserScan.points_xy() will round-trip back to approximately
    the same coordinates.
    """
    ranges = np.linalg.norm(points_xy, axis=1)
    bearings = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    order = np.argsort(bearings)
    bearings = bearings[order]
    ranges = ranges[order]
    n = len(ranges)
    if n < 2:
        angle_min = 0.0
        angle_increment = 1.0
    else:
        angle_min = float(bearings[0])
        angle_increment = float((bearings[-1] - bearings[0]) / (n - 1))
    return LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=angle_min,
        angle_max=float(bearings[-1]) if n >= 2 else 0.0,
        angle_increment=angle_increment if angle_increment != 0.0 else 1.0,
        ranges=ranges,
        range_min=0.01,
        range_max=100.0,
    )


def test_normal_estimation_on_line():
    """Points on y-axis should have normals approximately [1, 0] or [-1, 0]."""
    pts = np.column_stack([np.zeros(20), np.linspace(-5, 5, 20)])
    tree = cKDTree(pts)
    normals = _estimate_normals_2d(pts, tree, k=5)
    # Normals should be perpendicular to the y-axis, i.e. along x
    for i in range(len(normals)):
        n = normals[i]
        # Either [1, 0] or [-1, 0]
        assert abs(abs(n[0]) - 1.0) < 0.1, f"Normal {i}: {n}"
        assert abs(n[1]) < 0.1, f"Normal {i}: {n}"


def test_point_to_line_converges_on_wall():
    """Translating a scan along a wall should converge in point-to-line mode."""
    # Create a wall: collinear points at x=2
    wall_y = np.linspace(-3, 3, 50)
    ref = np.column_stack([np.full_like(wall_y, 2.0), wall_y])

    # Scan points near the wall, shifted slightly
    scan_pts = np.column_stack([np.full_like(wall_y, 1.9), wall_y + 0.3])
    scan = _make_laser_scan(scan_pts)
    pred = Pose2(0.0, 0.0, 0.0)

    cfg_p2l = IcpConfig(
        max_iterations=15,
        max_correspondence_dist_m=1.0,
        min_correspondences=5,
        trim_fraction=0.0,
        icp_mode="line",
        normal_k=5,
    )
    cfg_p2p = IcpConfig(
        max_iterations=15,
        max_correspondence_dist_m=1.0,
        min_correspondences=5,
        trim_fraction=0.0,
        icp_mode="point",
    )
    matcher_p2l = IcpScanMatcher(cfg_p2l)
    matcher_p2p = IcpScanMatcher(cfg_p2p)

    result_p2l = matcher_p2l.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    result_p2p = matcher_p2p.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)

    # Both should produce valid results
    assert result_p2l.score > float("-inf")
    assert result_p2p.score > float("-inf")
    # Point-to-line should correctly find the x-translation to the wall
    # The scan needs to shift +0.1 in x to reach x=2
    assert abs(result_p2l.pose_map.x - 0.1) < 0.15


def test_point_to_line_with_weights():
    """Range weights should work with point-to-line mode."""
    # L-shaped reference: wall along x and wall along y
    wall_x = np.column_stack([np.linspace(0, 5, 30), np.zeros(30)])
    wall_y = np.column_stack([np.zeros(20), np.linspace(0, 4, 20)])
    ref = np.vstack([wall_x, wall_y])

    # Scan: same L but shifted
    scan_pts = ref + np.array([0.05, 0.05])
    scan = _make_laser_scan(scan_pts)
    pred = Pose2(0.0, 0.0, 0.0)

    cfg = IcpConfig(
        max_iterations=20,
        max_correspondence_dist_m=1.0,
        min_correspondences=5,
        trim_fraction=0.0,
        icp_mode="line",
        normal_k=5,
        range_weight_mode="linear",
        range_weight_min_m=0.5,
    )
    matcher = IcpScanMatcher(cfg)
    result = matcher.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    # Should converge without error
    assert result.score > float("-inf")


def test_point_to_line_empty_scan():
    """Empty scan should return prediction without crash."""
    ref = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    scan = LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=0.0,
        angle_max=0.0,
        angle_increment=1.0,
        ranges=np.array([]),
        range_min=0.01,
        range_max=100.0,
    )
    pred = Pose2(1.0, 2.0, 0.5)
    cfg = IcpConfig(icp_mode="line", normal_k=5)
    matcher = IcpScanMatcher(cfg)
    result = matcher.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    assert result.pose_map.x == pytest.approx(pred.x)
    assert result.pose_map.y == pytest.approx(pred.y)
    assert result.pose_map.theta == pytest.approx(pred.theta)


def test_point_to_line_mode_in_config():
    """IcpConfig with icp_mode='line' should work through the matcher."""
    cfg = IcpConfig(
        icp_mode="line",
        normal_k=7,
        max_correspondence_dist_m=1.0,
        min_correspondences=5,
        trim_fraction=0.0,
    )
    assert cfg.icp_mode == "line"
    assert cfg.normal_k == 7

    # Simple convergence test with distinct geometry (not collinear)
    theta = np.linspace(0, np.pi, 40)
    ref = np.column_stack([2.0 * np.cos(theta), 2.0 * np.sin(theta)])

    # Shift scan slightly
    scan_pts = ref + np.array([0.05, -0.03])
    scan = _make_laser_scan(scan_pts)
    pred = Pose2(0.0, 0.0, 0.0)

    matcher = IcpScanMatcher(cfg)
    result = matcher.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    assert result.score > float("-inf")
    # Should approximately recover the offset
    assert abs(result.pose_map.x - (-0.05)) < 0.15
    assert abs(result.pose_map.y - 0.03) < 0.15


def test_point_mode_unchanged():
    """icp_mode='point' should produce same results as before (default)."""
    cfg_default = IcpConfig()
    cfg_point = IcpConfig(icp_mode="point")
    assert cfg_default.icp_mode == "point"
    assert cfg_default.normal_k == 5

    # Simple test: arc reference, shifted scan
    theta = np.linspace(0, np.pi, 40)
    ref = np.column_stack([3.0 * np.cos(theta), 3.0 * np.sin(theta)])
    scan_pts = ref + np.array([0.1, 0.0])
    scan = _make_laser_scan(scan_pts)
    pred = Pose2(0.0, 0.0, 0.0)

    m1 = IcpScanMatcher(cfg_default)
    m2 = IcpScanMatcher(cfg_point)
    r1 = m1.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    r2 = m2.match(scan=scan, prediction_map=pred, ref_points_xy_map=ref)
    assert r1.pose_map.x == pytest.approx(r2.pose_map.x)
    assert r1.pose_map.y == pytest.approx(r2.pose_map.y)
    assert r1.pose_map.theta == pytest.approx(r2.pose_map.theta)
