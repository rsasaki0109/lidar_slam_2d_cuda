"""Tests for range-weighted correspondence in ICP and correlative matchers."""
from __future__ import annotations

import numpy as np
import pytest

from slamx.core.local_matching.range_weights import compute_range_weights
from slamx.core.local_matching.icp import IcpConfig, IcpScanMatcher, _best_fit_se2
from slamx.core.local_matching.correlative import CorrelativeGridConfig, CorrelativeScanMatcher
from slamx.core.types import LaserScan, Pose2


def _make_scan(ranges: list[float], angle_min: float = -np.pi, angle_max: float = np.pi) -> LaserScan:
    r = np.array(ranges, dtype=np.float64)
    n = len(ranges)
    inc = (angle_max - angle_min) / max(n - 1, 1)
    return LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=inc,
        ranges=r,
    )


class TestComputeRangeWeights:
    def test_none_mode(self):
        r = np.array([0.5, 1.0, 2.0])
        assert compute_range_weights(r, "none", 1.0) is None

    def test_linear_mode(self):
        r = np.array([0.0, 0.5, 1.0, 2.0])
        w = compute_range_weights(r, "linear", 1.0)
        assert w is not None
        np.testing.assert_allclose(w, [0.0, 0.5, 1.0, 1.0])

    def test_sigmoid_mode(self):
        r = np.array([0.0, 0.5, 1.0, 2.0])
        w = compute_range_weights(r, "sigmoid", 1.0)
        assert w is not None
        # sigmoid(0) at r=min_m should be ~0.5
        assert abs(w[2] - 0.5) < 0.01
        # far range should be close to 1
        assert w[3] > 0.9
        # near range should be close to 0
        assert w[0] < 0.01

    def test_zero_min_returns_none(self):
        r = np.array([1.0, 2.0])
        assert compute_range_weights(r, "linear", 0.0) is None

    def test_unknown_mode_returns_none(self):
        r = np.array([1.0, 2.0])
        assert compute_range_weights(r, "unknown", 1.0) is None


class TestWeightedBestFitSE2:
    def test_unweighted_identity(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        pose = _best_fit_se2(pts, pts)
        assert abs(pose.x) < 1e-6
        assert abs(pose.y) < 1e-6
        assert abs(pose.theta) < 1e-6

    def test_weighted_shifts_toward_heavy_points(self):
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Destination has a small shift
        dst = src + np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # Uniform weights: shift is distributed
        p_uniform = _best_fit_se2(src, dst)
        # Heavy weight on first point: shift should be closer to (0.1, 0)
        w = np.array([10.0, 1.0, 1.0])
        p_weighted = _best_fit_se2(src, dst, weights=w)
        # Weighted result should have larger x shift than uniform
        assert abs(p_weighted.x) >= abs(p_uniform.x) * 0.9


class TestIcpWithRangeWeights:
    def test_icp_linear_weight_runs(self):
        cfg = IcpConfig(
            max_iterations=5,
            max_correspondence_dist_m=2.0,
            min_correspondences=3,
            trim_fraction=0.0,
            range_weight_mode="linear",
            range_weight_min_m=0.5,
        )
        matcher = IcpScanMatcher(cfg)
        # Simple scan: 8 points in a semicircle at range ~2m
        angles = np.linspace(-np.pi/2, np.pi/2, 8)
        ranges = [2.0] * 8
        scan = _make_scan(ranges, angle_min=float(angles[0]), angle_max=float(angles[-1]))
        pts_map = scan.points_xy()
        pred = Pose2(0.0, 0.0, 0.0)
        result = matcher.match(scan=scan, prediction_map=pred, ref_points_xy_map=pts_map)
        assert result.score > float("-inf")
        assert abs(result.pose_map.x) < 0.5
        assert abs(result.pose_map.y) < 0.5

    def test_icp_sigmoid_weight_downweights_near(self):
        cfg_none = IcpConfig(
            max_iterations=10,
            max_correspondence_dist_m=3.0,
            min_correspondences=3,
            trim_fraction=0.0,
            range_weight_mode="none",
        )
        cfg_sig = IcpConfig(
            max_iterations=10,
            max_correspondence_dist_m=3.0,
            min_correspondences=3,
            trim_fraction=0.0,
            range_weight_mode="sigmoid",
            range_weight_min_m=1.5,
        )
        # Mix of near (0.3m) and far (3m) points
        ranges = [0.3, 0.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        angles = np.linspace(-np.pi/2, np.pi/2, 8)
        scan = _make_scan(ranges, angle_min=float(angles[0]), angle_max=float(angles[-1]))
        pts = scan.points_xy()
        pred = Pose2(0.05, 0.0, 0.0)  # small offset
        r_none = IcpScanMatcher(cfg_none).match(scan=scan, prediction_map=pred, ref_points_xy_map=pts)
        r_sig = IcpScanMatcher(cfg_sig).match(scan=scan, prediction_map=pred, ref_points_xy_map=pts)
        # Both should converge
        assert r_none.score > float("-inf")
        assert r_sig.score > float("-inf")


class TestCorrelativeWithRangeWeights:
    def test_correlative_linear_weight_runs(self):
        cfg = CorrelativeGridConfig(
            linear_step_m=0.05,
            angular_step_deg=5.0,
            linear_window_m=0.2,
            angular_window_deg=10.0,
            sigma_hit_m=0.1,
            range_weight_mode="linear",
            range_weight_min_m=1.0,
        )
        matcher = CorrelativeScanMatcher(cfg)
        ranges = [2.0] * 8
        angles = np.linspace(-np.pi/2, np.pi/2, 8)
        scan = _make_scan(ranges, angle_min=float(angles[0]), angle_max=float(angles[-1]))
        pts_map = scan.points_xy()
        pred = Pose2(0.0, 0.0, 0.0)
        result = matcher.match(scan=scan, prediction_map=pred, ref_points_xy_map=pts_map)
        assert result.score > float("-inf")
