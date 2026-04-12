"""Tests for the branch-and-bound correlative scan matcher."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slamx.core.local_matching.branch_bound import (
    BranchBoundConfig,
    BranchBoundScanMatcher,
    HybridBBScanMatcher,
    ProbabilityGrid,
)
from slamx.core.local_matching.icp import IcpConfig
from slamx.core.types import LaserScan, Pose2


def _make_scan(points_xy: np.ndarray) -> LaserScan:
    """Create a LaserScan from Cartesian points (for testing).

    Places each point into a fine angular grid so that LaserScan.points_xy()
    faithfully reconstructs the Cartesian coordinates.
    """
    r = np.linalg.norm(points_xy, axis=1)
    theta = np.arctan2(points_xy[:, 1], points_xy[:, 0])

    # Use a fine uniform grid covering the angular span
    angle_min = float(np.min(theta))
    angle_max = float(np.max(theta))
    n = len(r)
    if n < 2:
        return LaserScan(
            stamp_ns=0, frame_id="test",
            angle_min=angle_min, angle_max=angle_max,
            angle_increment=0.01, ranges=r,
            range_min=0.0, range_max=100.0,
        )

    # Create a grid with exactly n bins spanning [angle_min, angle_max]
    # so that index i -> angle_min + i * inc reconstructs the right angle.
    # We sort by angle and set inc = (max-min)/(n-1).
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]

    # Build uniform-angle grid with one bin per point
    n_bins = max(n, 2)
    angle_inc = (angle_max - angle_min) / (n_bins - 1) if n_bins > 1 else 0.01
    ranges = np.full(n_bins, np.nan)
    for i in range(n):
        idx = int(round((theta_sorted[i] - angle_min) / angle_inc))
        idx = max(0, min(n_bins - 1, idx))
        ranges[idx] = r_sorted[i]

    return LaserScan(
        stamp_ns=0,
        frame_id="test",
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=float(angle_inc),
        ranges=ranges,
        range_min=0.0,
        range_max=100.0,
    )


def _wall_points(x_range: tuple[float, float], y: float, n: int = 50) -> np.ndarray:
    """Generate a horizontal wall at y with n points."""
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.full(n, y)
    return np.column_stack([xs, ys])


def _l_shape_points() -> np.ndarray:
    """Generate an L-shaped reference (two walls)."""
    wall1 = _wall_points((-2.0, 2.0), 3.0, 40)
    wall2 = np.column_stack([np.full(30, 2.0), np.linspace(0.0, 3.0, 30)])
    return np.vstack([wall1, wall2])


class TestProbabilityGrid:
    def test_scores_near_reference(self) -> None:
        """Points near reference should score high."""
        ref = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        grid = ProbabilityGrid(
            ref, resolution=0.05, sigma=0.10,
            origin=(0.0, 0.0), size=(100, 100), n_levels=1,
        )
        # Score at exact reference location should be high
        near_pts = np.array([[1.0, 1.0], [2.0, 2.0]])
        score_near = grid.score_at(near_pts, level=0)
        # Score far away should be low
        far_pts = np.array([[4.5, 4.5], [4.6, 4.6]])
        score_far = grid.score_at(far_pts, level=0)
        assert score_near > score_far, f"near={score_near}, far={score_far}"
        assert score_near > 0.0

    def test_multi_resolution_upper_bound(self) -> None:
        """Coarser levels should have >= scores of finer levels (max-pooling)."""
        ref = np.array([[1.5, 1.5], [2.5, 2.5]])
        grid = ProbabilityGrid(
            ref, resolution=0.05, sigma=0.10,
            origin=(0.0, 0.0), size=(80, 80), n_levels=3,
        )
        test_pts = np.array([[1.5, 1.5], [2.0, 2.0]])
        score_fine = grid.score_at(test_pts, level=0)
        score_mid = grid.score_at(test_pts, level=1)
        score_coarse = grid.score_at(test_pts, level=2)
        # Max-pooling guarantee: coarser >= finer
        assert score_mid >= score_fine - 1e-6, f"mid={score_mid} < fine={score_fine}"
        assert score_coarse >= score_mid - 1e-6, f"coarse={score_coarse} < mid={score_mid}"

    def test_empty_reference(self) -> None:
        """Empty reference produces all-zero grid."""
        ref = np.zeros((0, 2))
        grid = ProbabilityGrid(
            ref, resolution=0.05, sigma=0.10,
            origin=(0.0, 0.0), size=(20, 20), n_levels=2,
        )
        pts = np.array([[0.5, 0.5]])
        assert grid.score_at(pts, level=0) == 0.0
        assert grid.score_at(pts, level=1) == 0.0


class TestBranchBoundMatcher:
    def test_finds_known_translation(self) -> None:
        """Given a known shift, B&B should find it."""
        ref = _wall_points((-2.0, 2.0), 3.0, 60)
        # Scan points are wall shifted by (0.1, 0.05) in scan frame
        shift_x, shift_y = 0.1, 0.05
        scan_pts = ref - np.array([shift_x, shift_y])
        scan = _make_scan(scan_pts)

        cfg = BranchBoundConfig(
            resolution_m=0.02,
            n_levels=3,
            linear_window_m=0.3,
            angular_window_deg=5.0,
            angular_step_deg=1.0,
            sigma_hit_m=0.08,
        )
        matcher = BranchBoundScanMatcher(cfg)
        result = matcher.match(
            scan=scan,
            prediction_map=Pose2(0.0, 0.0, 0.0),
            ref_points_xy_map=ref,
        )

        assert result.score > 0.0
        assert abs(result.pose_map.x - shift_x) < 0.1, f"x={result.pose_map.x}"
        assert abs(result.pose_map.y - shift_y) < 0.1, f"y={result.pose_map.y}"

    def test_finds_known_rotation(self) -> None:
        """Given a known rotation, B&B should find it."""
        ref = _l_shape_points()
        true_theta = np.deg2rad(5.0)
        c, s = np.cos(-true_theta), np.sin(-true_theta)
        rot_inv = np.array([[c, -s], [s, c]])
        scan_pts = ref @ rot_inv.T  # scan in sensor frame
        scan = _make_scan(scan_pts)

        cfg = BranchBoundConfig(
            resolution_m=0.03,
            n_levels=3,
            linear_window_m=0.2,
            angular_window_deg=15.0,
            angular_step_deg=1.0,
            sigma_hit_m=0.08,
        )
        matcher = BranchBoundScanMatcher(cfg)
        result = matcher.match(
            scan=scan,
            prediction_map=Pose2(0.0, 0.0, 0.0),
            ref_points_xy_map=ref,
        )

        assert result.score > 0.0
        theta_err = abs(math.atan2(
            math.sin(result.pose_map.theta - true_theta),
            math.cos(result.pose_map.theta - true_theta),
        ))
        assert theta_err < np.deg2rad(5.0), f"theta_err={np.rad2deg(theta_err)} deg"

    def test_beats_exhaustive_on_l_shape(self) -> None:
        """B&B should find similar or better result as exhaustive grid search on L-shape."""
        ref = _l_shape_points()
        shift = np.array([0.08, 0.04])
        scan_pts = ref - shift
        scan = _make_scan(scan_pts)

        cfg = BranchBoundConfig(
            resolution_m=0.03,
            n_levels=3,
            linear_window_m=0.2,
            angular_window_deg=5.0,
            angular_step_deg=1.0,
            sigma_hit_m=0.08,
        )
        matcher = BranchBoundScanMatcher(cfg)
        result = matcher.match(
            scan=scan,
            prediction_map=Pose2(0.0, 0.0, 0.0),
            ref_points_xy_map=ref,
        )

        # L-shape is not degenerate, so we should recover the shift well
        dist = math.hypot(result.pose_map.x - shift[0], result.pose_map.y - shift[1])
        assert dist < 0.15, f"dist={dist}"

    def test_empty_scan(self) -> None:
        """Empty scan returns prediction."""
        ref = np.array([[1.0, 1.0]])
        scan = LaserScan(
            stamp_ns=0,
            frame_id="test",
            angle_min=0.0,
            angle_max=0.0,
            angle_increment=0.01,
            ranges=np.array([]),
            range_min=0.0,
            range_max=100.0,
        )
        matcher = BranchBoundScanMatcher()
        pred = Pose2(1.0, 2.0, 0.5)
        result = matcher.match(
            scan=scan,
            prediction_map=pred,
            ref_points_xy_map=ref,
        )
        assert result.pose_map.x == pred.x
        assert result.pose_map.y == pred.y
        assert result.pose_map.theta == pred.theta
        assert result.score == float("-inf")

    def test_empty_reference(self) -> None:
        """Empty reference returns prediction."""
        scan_pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        scan = _make_scan(scan_pts)
        matcher = BranchBoundScanMatcher()
        pred = Pose2(0.0, 0.0, 0.0)
        result = matcher.match(
            scan=scan,
            prediction_map=pred,
            ref_points_xy_map=np.zeros((0, 2)),
        )
        assert result.score == float("-inf")

    def test_implements_protocol(self) -> None:
        """BranchBoundScanMatcher implements the ScanMatcher protocol."""
        from slamx.core.local_matching.protocol import ScanMatcher
        matcher = BranchBoundScanMatcher()
        assert isinstance(matcher, ScanMatcher)


class TestHybridBBMatcher:
    def test_hybrid_bb_icp(self) -> None:
        """B&B coarse + ICP refine should work end-to-end."""
        ref = _l_shape_points()
        shift = np.array([0.05, 0.03])
        scan_pts = ref - shift
        scan = _make_scan(scan_pts)

        bb_cfg = BranchBoundConfig(
            resolution_m=0.03,
            n_levels=3,
            linear_window_m=0.2,
            angular_window_deg=10.0,
            angular_step_deg=1.0,
            sigma_hit_m=0.08,
        )
        icp_cfg = IcpConfig(
            max_iterations=15,
            max_correspondence_dist_m=0.5,
            min_correspondences=10,
            trim_fraction=0.1,
        )
        matcher = HybridBBScanMatcher(branch_bound=bb_cfg, icp=icp_cfg)
        result = matcher.match(
            scan=scan,
            prediction_map=Pose2(0.0, 0.0, 0.0),
            ref_points_xy_map=ref,
        )

        assert result.score > float("-inf")
        assert "hybrid_bb" in result.diagnostics

    def test_implements_protocol(self) -> None:
        """HybridBBScanMatcher implements the ScanMatcher protocol."""
        from slamx.core.local_matching.protocol import ScanMatcher
        matcher = HybridBBScanMatcher()
        assert isinstance(matcher, ScanMatcher)
