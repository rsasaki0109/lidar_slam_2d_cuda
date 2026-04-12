"""Tests for ScanContext descriptor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slamx.core.local_matching.scan_context import ScanContextConfig, ScanContextDescriptor
from slamx.core.types import LaserScan


def _make_scan(
    ranges: np.ndarray,
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
) -> LaserScan:
    """Helper to build a LaserScan from a 1D range array."""
    n = len(ranges)
    if n < 2:
        inc = 0.0
    else:
        inc = (angle_max - angle_min) / (n - 1)
    return LaserScan(
        stamp_ns=0,
        frame_id="laser",
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=inc,
        ranges=np.asarray(ranges, dtype=np.float64),
        range_min=0.01,
        range_max=30.0,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_descriptor_same_scan_distance_zero():
    """Same scan should produce distance 0."""
    cfg = ScanContextConfig(n_sectors=60, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    ranges = np.full(360, 5.0)
    scan = _make_scan(ranges)
    desc = sc.compute(scan)

    dist = sc.distance(desc, desc)
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_descriptor_different_scans():
    """Different scans should produce nonzero distance."""
    cfg = ScanContextConfig(n_sectors=60, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    scan_a = _make_scan(np.full(360, 5.0))
    scan_b = _make_scan(np.linspace(1.0, 9.0, 360))

    desc_a = sc.compute(scan_a)
    desc_b = sc.compute(scan_b)

    dist = sc.distance(desc_a, desc_b)
    assert dist > 0.01, f"Expected nonzero distance, got {dist}"


def test_descriptor_rotation_invariance():
    """A rotated scan should match via column shift."""
    n_rays = 360
    cfg = ScanContextConfig(n_sectors=60, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    # Create a scan with a distinctive pattern
    ranges = np.full(n_rays, 5.0)
    ranges[0:90] = 2.0  # first quadrant is close

    scan_a = _make_scan(ranges, angle_min=-math.pi, angle_max=math.pi)

    # Rotate by 90 degrees: shift ranges by 90 entries
    ranges_rot = np.roll(ranges, 90)
    scan_b = _make_scan(ranges_rot, angle_min=-math.pi, angle_max=math.pi)

    desc_a = sc.compute(scan_a)
    desc_b = sc.compute(scan_b)

    dist = sc.distance(desc_a, desc_b)
    # With rotation invariance, the distance should be small
    assert dist < 0.05, f"Expected small distance for rotated scan, got {dist}"


def test_descriptor_empty_scan():
    """Empty scan should produce zero descriptor without crash."""
    cfg = ScanContextConfig(n_sectors=60, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    # All-NaN ranges = no valid points
    ranges = np.full(100, np.nan)
    scan = _make_scan(ranges)

    desc = sc.compute(scan)
    assert desc.shape == (60,)
    assert np.allclose(desc, 0.0)

    # Distance between two zero descriptors should be 1.0 (undefined similarity)
    dist = sc.distance(desc, desc)
    assert dist == pytest.approx(1.0, abs=1e-9)


def test_best_shift_returns_rotation():
    """Best shift should correspond to the rotation angle."""
    n_sectors = 60
    cfg = ScanContextConfig(n_sectors=n_sectors, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    # Build two descriptors directly (bypass scan creation)
    rng = np.random.RandomState(42)
    desc_a = rng.uniform(0.1, 1.0, size=n_sectors)

    # Shift by 15 sectors = 90 degrees
    shift_amount = 15
    desc_b = np.roll(desc_a, -shift_amount)

    dist, best_shift = sc.best_shift(desc_a, desc_b)
    assert dist == pytest.approx(0.0, abs=1e-6), f"Expected ~0 distance, got {dist}"
    assert best_shift == shift_amount, (
        f"Expected shift {shift_amount}, got {best_shift}"
    )


def test_descriptor_shape():
    """Descriptor should have shape (n_sectors,)."""
    for n_sec in [20, 60, 120]:
        cfg = ScanContextConfig(n_sectors=n_sec, max_range=10.0)
        sc = ScanContextDescriptor(cfg)
        scan = _make_scan(np.full(360, 5.0))
        desc = sc.compute(scan)
        assert desc.shape == (n_sec,), f"Expected ({n_sec},), got {desc.shape}"


def test_descriptor_values_normalized():
    """Descriptor values should be in [0, 1] when ranges <= max_range."""
    cfg = ScanContextConfig(n_sectors=60, max_range=10.0)
    sc = ScanContextDescriptor(cfg)

    ranges = np.linspace(0.5, 10.0, 360)
    scan = _make_scan(ranges)
    desc = sc.compute(scan)

    assert np.all(desc >= 0.0)
    assert np.all(desc <= 1.0)
