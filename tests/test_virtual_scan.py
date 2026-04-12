"""Tests for virtual 2D scan generation from 3D point clouds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slamx.core.io.bag import PointCloud3D
from slamx.core.preprocess.virtual_scan import VirtualScanConfig, pointcloud_to_virtual_scan


def _make_cloud(points: list[list[float]], stamp_ns: int = 0) -> PointCloud3D:
    """Helper to create a PointCloud3D from a list of [x, y, z] points."""
    return PointCloud3D(
        stamp_ns=stamp_ns,
        frame_id="lidar",
        points_xyz=np.array(points, dtype=np.float64),
    )


def test_pointcloud_to_virtual_scan_basic():
    """A wall at x=2, y=-1..1, z=-0.5..0.5 -- only z-band points contribute.

    With z_min=-0.1, z_max=0.3 and the wall at x=2, the scan should show
    hits at range ~2 m near bearing 0.
    """
    pts = []
    for y in np.linspace(-1.0, 1.0, 21):
        for z in np.linspace(-0.5, 0.5, 11):
            pts.append([2.0, y, z])
    cloud = _make_cloud(pts)

    cfg = VirtualScanConfig(z_min=-0.1, z_max=0.3, angular_resolution_deg=1.0, max_range=30.0)
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    # The scan should be a valid LaserScan
    assert scan.ranges.size > 0
    assert scan.stamp_ns == 0
    assert scan.frame_id == "lidar"

    # Check that there are valid (non-NaN) bins near bearing 0
    bearings = scan.bearings()
    zero_idx = int(np.argmin(np.abs(bearings)))
    # The wall spans roughly atan2(+-1, 2) ~ +-26.6 deg, so several bins around 0 should be valid
    window = 5
    nearby = scan.ranges[max(0, zero_idx - window) : zero_idx + window + 1]
    valid_nearby = nearby[np.isfinite(nearby)]
    assert valid_nearby.size > 0, "Expected valid hits near bearing 0"

    # Range should be close to 2.0 (exact for points at y=0)
    assert np.min(valid_nearby) == pytest.approx(2.0, abs=0.2)


def test_virtual_scan_empty_after_z_filter():
    """All points outside the z band should yield an all-NaN scan."""
    pts = [[3.0, 0.0, 1.0], [3.0, 1.0, -1.0], [3.0, -1.0, 2.0]]
    cloud = _make_cloud(pts)

    cfg = VirtualScanConfig(z_min=-0.1, z_max=0.3, max_range=30.0)
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    assert scan.ranges.size > 0
    assert np.all(np.isnan(scan.ranges)), "All bins should be NaN when no points pass z-filter"


def test_virtual_scan_takes_minimum_range_per_cell():
    """Two points at the same bearing but different ranges -> takes minimum."""
    # Both at bearing = 0 (along +x axis), z within band
    pts = [[5.0, 0.0, 0.1], [2.0, 0.0, 0.1]]
    cloud = _make_cloud(pts)

    cfg = VirtualScanConfig(z_min=-0.1, z_max=0.3, angular_resolution_deg=1.0, max_range=30.0)
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    bearings = scan.bearings()
    zero_idx = int(np.argmin(np.abs(bearings)))
    assert np.isfinite(scan.ranges[zero_idx])
    assert scan.ranges[zero_idx] == pytest.approx(2.0, abs=0.01)


def test_virtual_scan_max_range_filter():
    """Points beyond max_range should not appear in the scan."""
    pts = [[50.0, 0.0, 0.1]]
    cloud = _make_cloud(pts)

    cfg = VirtualScanConfig(z_min=-0.1, z_max=0.3, max_range=10.0)
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    assert np.all(np.isnan(scan.ranges))


def test_virtual_scan_empty_cloud():
    """An empty point cloud yields an all-NaN scan."""
    cloud = PointCloud3D(stamp_ns=42, frame_id="sensor", points_xyz=np.empty((0, 3)))
    cfg = VirtualScanConfig()
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    assert scan.stamp_ns == 42
    assert scan.ranges.size > 0
    assert np.all(np.isnan(scan.ranges))


def test_virtual_scan_angular_coverage():
    """A point at bearing ~90 deg (along +y) should land in the correct bin."""
    pts = [[0.0, 3.0, 0.0]]  # bearing = pi/2 ~ 90 deg
    cloud = _make_cloud(pts)

    cfg = VirtualScanConfig(
        z_min=-0.5, z_max=0.5, angular_resolution_deg=1.0, max_range=10.0
    )
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    bearings_deg = np.rad2deg(scan.bearings())
    # Find the bin closest to 90 degrees
    idx_90 = int(np.argmin(np.abs(bearings_deg - 90.0)))
    assert np.isfinite(scan.ranges[idx_90])
    assert scan.ranges[idx_90] == pytest.approx(3.0, abs=0.05)


def test_virtual_scan_preserves_stamp_and_frame():
    """Metadata from the cloud should carry through to the scan."""
    cloud = PointCloud3D(
        stamp_ns=123456789,
        frame_id="velodyne",
        points_xyz=np.array([[1.0, 0.0, 0.0]]),
    )
    cfg = VirtualScanConfig(z_min=-1.0, z_max=1.0)
    scan = pointcloud_to_virtual_scan(cloud, cfg)

    assert scan.stamp_ns == 123456789
    assert scan.frame_id == "velodyne"
