from __future__ import annotations

import math

import numpy as np

from slamx.core.preprocess.imu_utils import quaternion_to_pitch, pitch_adjusted_min_range
from slamx.core.preprocess.pipeline import PreprocessConfig, apply_pitch_compensation, preprocess_scan
from slamx.core.types import LaserScan


def _make_scan(ranges: list[float]) -> LaserScan:
    n = len(ranges)
    return LaserScan(
        stamp_ns=1,
        frame_id="laser",
        angle_min=np.deg2rad(-90.0),
        angle_max=np.deg2rad(-90.0 + (n - 1) * 30.0),
        angle_increment=np.deg2rad(30.0),
        ranges=np.array(ranges, dtype=np.float64),
        range_min=0.0,
        range_max=30.0,
    )


def test_quaternion_to_pitch_level() -> None:
    """Identity quaternion -> pitch = 0."""
    pitch = quaternion_to_pitch(0.0, 0.0, 0.0, 1.0)
    assert abs(pitch) < 1e-9


def test_quaternion_to_pitch_tilted() -> None:
    """15 degree pitch -> correct value."""
    angle = math.radians(15.0)
    # Quaternion for pure pitch (rotation about Y axis):
    # qx=0, qy=sin(angle/2), qz=0, qw=cos(angle/2)
    qy = math.sin(angle / 2.0)
    qw = math.cos(angle / 2.0)
    pitch = quaternion_to_pitch(0.0, qy, 0.0, qw)
    assert abs(pitch - angle) < 1e-9, f"Expected {angle}, got {pitch}"


def test_quaternion_to_pitch_negative() -> None:
    """Negative pitch (-10 degrees)."""
    angle = math.radians(-10.0)
    qy = math.sin(angle / 2.0)
    qw = math.cos(angle / 2.0)
    pitch = quaternion_to_pitch(0.0, qy, 0.0, qw)
    assert abs(pitch - angle) < 1e-9


def test_pitch_adjusted_min_range_zero_pitch() -> None:
    """Zero pitch -> floor range = 0 (no filtering)."""
    r = pitch_adjusted_min_range(0.0, 0.5, 0.0)
    assert r == 0.0


def test_pitch_adjusted_min_range_nonzero() -> None:
    """10 degree pitch, 0.5m height -> range = 0.5 / sin(10deg) ~ 2.88m."""
    pitch = math.radians(10.0)
    r = pitch_adjusted_min_range(pitch, 0.5, 0.0)
    expected = 0.5 / math.sin(pitch)
    assert abs(r - expected) < 1e-6


def test_pitch_compensation_removes_floor_rays() -> None:
    """With pitch=10deg, near-range rays below the floor threshold should be removed."""
    pitch = math.radians(10.0)
    # floor range ~ 0.5 / sin(10deg) ~ 2.88m
    # with margin 1.5 -> threshold ~ 4.32m
    # Rays below that threshold should be NaN-ed
    scan = _make_scan([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0])
    cfg = PreprocessConfig(
        pitch_compensation_enabled=True,
        pitch_sensor_height_m=0.5,
        pitch_floor_margin=1.5,
    )
    result = apply_pitch_compensation(scan, cfg, pitch)
    # threshold ~ 2.88 * 1.5 ~ 4.32
    # rays 1.0, 2.0, 3.0 are below threshold -> NaN
    # rays 5.0, 8.0, 10.0, 15.0 are above -> kept
    assert np.isnan(result.ranges[0])
    assert np.isnan(result.ranges[1])
    assert np.isnan(result.ranges[2])
    assert result.ranges[3] == 5.0
    assert result.ranges[4] == 8.0
    assert result.ranges[5] == 10.0
    assert result.ranges[6] == 15.0


def test_pitch_compensation_disabled_is_noop() -> None:
    """When disabled, scan is unchanged even if pitch is provided."""
    pitch = math.radians(10.0)
    scan = _make_scan([1.0, 2.0, 3.0, 5.0])
    cfg = PreprocessConfig(
        pitch_compensation_enabled=False,
        pitch_sensor_height_m=0.5,
        pitch_floor_margin=1.5,
    )
    result = preprocess_scan(scan, cfg, pitch_rad=pitch)
    np.testing.assert_array_equal(result.ranges, scan.ranges)


def test_pitch_compensation_zero_pitch_is_noop() -> None:
    """When pitch is zero, no rays should be removed."""
    scan = _make_scan([1.0, 2.0, 3.0, 5.0])
    cfg = PreprocessConfig(
        pitch_compensation_enabled=True,
        pitch_sensor_height_m=0.5,
        pitch_floor_margin=1.5,
    )
    result = preprocess_scan(scan, cfg, pitch_rad=0.0)
    np.testing.assert_array_equal(result.ranges, scan.ranges)


def test_pitch_compensation_integrated_in_preprocess_scan() -> None:
    """Verify that preprocess_scan applies pitch compensation when enabled."""
    pitch = math.radians(10.0)
    scan = _make_scan([1.0, 2.0, 3.0, 5.0, 8.0])
    cfg = PreprocessConfig(
        pitch_compensation_enabled=True,
        pitch_sensor_height_m=0.5,
        pitch_floor_margin=1.5,
    )
    result = preprocess_scan(scan, cfg, pitch_rad=pitch)
    # threshold ~ 4.32; rays 1.0, 2.0, 3.0 removed
    assert np.isnan(result.ranges[0])
    assert np.isnan(result.ranges[1])
    assert np.isnan(result.ranges[2])
    assert result.ranges[3] == 5.0
    assert result.ranges[4] == 8.0


def test_preprocess_scan_backward_compatible() -> None:
    """preprocess_scan still works without pitch_rad argument."""
    scan = _make_scan([1.0, 2.0, 3.0, 5.0])
    cfg = PreprocessConfig()
    result = preprocess_scan(scan, cfg)
    np.testing.assert_array_equal(result.ranges, scan.ranges)
