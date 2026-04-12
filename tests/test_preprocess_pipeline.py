from __future__ import annotations

import numpy as np

from slamx.core.preprocess.pipeline import PreprocessConfig, preprocess_scan
from slamx.core.types import LaserScan


def _scan() -> LaserScan:
    return LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-90.0),
        angle_max=np.deg2rad(90.0),
        angle_increment=np.deg2rad(30.0),
        ranges=np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        range_min=0.1,
        range_max=10.0,
    )


def test_preprocess_applies_min_and_max_range() -> None:
    out = preprocess_scan(
        _scan(),
        PreprocessConfig(min_range=1.5, max_range=4.5),
    )
    np.testing.assert_allclose(
        out.ranges,
        np.array([np.nan, np.nan, 2.0, 3.0, 4.0, np.nan, np.nan]),
        equal_nan=True,
    )


def test_preprocess_crops_bearings_before_stride() -> None:
    out = preprocess_scan(
        _scan(),
        PreprocessConfig(min_angle_deg=-30.0, max_angle_deg=30.0, stride=2),
    )
    np.testing.assert_allclose(out.ranges, np.array([2.0, 4.0]))
    assert np.isclose(out.angle_min, np.deg2rad(-30.0))
    assert np.isclose(out.angle_increment, np.deg2rad(60.0))
    assert np.isclose(out.angle_max, np.deg2rad(30.0))


def test_preprocess_returns_empty_scan_when_crop_excludes_all_points() -> None:
    out = preprocess_scan(
        _scan(),
        PreprocessConfig(min_angle_deg=120.0, max_angle_deg=140.0),
    )
    assert out.ranges.size == 0
    assert out.angle_min == 0.0
    assert out.angle_max == 0.0


def test_preprocess_masks_only_near_side_of_steep_local_gradient() -> None:
    scan = LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-30.0),
        angle_max=np.deg2rad(30.0),
        angle_increment=np.deg2rad(15.0),
        ranges=np.array([4.0, 0.9, 4.2, 4.1, 4.0]),
        range_min=0.1,
        range_max=10.0,
    )
    out = preprocess_scan(
        scan,
        PreprocessConfig(gradient_mask_diff_m=1.0, gradient_mask_max_range=2.0),
    )
    np.testing.assert_allclose(
        out.ranges,
        np.array([4.0, np.nan, 4.2, 4.1, 4.0]),
        equal_nan=True,
    )


def test_preprocess_gradient_mask_ratio_detects_range_jump() -> None:
    scan = LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-30.0),
        angle_max=np.deg2rad(30.0),
        angle_increment=np.deg2rad(15.0),
        ranges=np.array([4.0, 1.2, 4.0, 4.0, 4.0]),
        range_min=0.1,
        range_max=10.0,
    )
    # ratio 4.0/1.2 = 3.33, threshold 3.0 → should mask the near point (1.2)
    out = preprocess_scan(
        scan,
        PreprocessConfig(gradient_mask_ratio=3.0, gradient_mask_max_range=2.0),
    )
    np.testing.assert_allclose(
        out.ranges,
        np.array([4.0, np.nan, 4.0, 4.0, 4.0]),
        equal_nan=True,
    )


def test_preprocess_gradient_mask_ratio_below_threshold_keeps_all() -> None:
    scan = LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-30.0),
        angle_max=np.deg2rad(30.0),
        angle_increment=np.deg2rad(15.0),
        ranges=np.array([4.0, 2.5, 4.0, 4.0, 4.0]),
        range_min=0.1,
        range_max=10.0,
    )
    # ratio 4.0/2.5 = 1.6, threshold 3.0 → no mask
    out = preprocess_scan(
        scan,
        PreprocessConfig(gradient_mask_ratio=3.0, gradient_mask_max_range=3.0),
    )
    np.testing.assert_allclose(out.ranges, scan.ranges)


def test_preprocess_gradient_mask_expands_to_neighbor_window() -> None:
    scan = LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-30.0),
        angle_max=np.deg2rad(30.0),
        angle_increment=np.deg2rad(15.0),
        ranges=np.array([4.0, 0.9, 4.2, 4.1, 4.0]),
        range_min=0.1,
        range_max=10.0,
    )
    out = preprocess_scan(
        scan,
        PreprocessConfig(
            gradient_mask_diff_m=1.0,
            gradient_mask_max_range=2.0,
            gradient_mask_window=1,
        ),
    )
    np.testing.assert_allclose(
        out.ranges,
        np.array([np.nan, np.nan, np.nan, 4.1, 4.0]),
        equal_nan=True,
    )
