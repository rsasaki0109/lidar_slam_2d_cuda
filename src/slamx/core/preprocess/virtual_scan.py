"""Virtual 2D scan generator from 3D PointCloud2 data.

Slices a 3D point cloud horizontally (by z-band) and projects into a 2D
LaserScan.  This makes the 2D observation pitch-invariant, closing the
"ramp" gap where physical tilt corrupts native 2D LaserScan data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.io.bag import PointCloud3D
from slamx.core.types import LaserScan


@dataclass
class VirtualScanConfig:
    """Parameters for converting a 3D cloud to a virtual 2D scan."""

    z_min: float = -0.1  # height band lower bound (sensor frame)
    z_max: float = 0.3  # height band upper bound (sensor frame)
    angle_min_deg: float = -180.0
    angle_max_deg: float = 180.0
    angular_resolution_deg: float = 0.5  # output scan resolution
    max_range: float = 30.0


def pointcloud_to_virtual_scan(
    cloud: PointCloud3D,
    cfg: VirtualScanConfig,
) -> LaserScan:
    """Convert a 3D point cloud to a virtual 2D scan by horizontal slicing.

    Algorithm
    ---------
    1. Filter points by ``z_min <= z <= z_max``.
    2. Project to 2D using the remaining (x, y).
    3. Compute bearing ``atan2(y, x)`` and range ``sqrt(x**2 + y**2)``.
    4. Bin into angular cells at *angular_resolution_deg*.
    5. For each cell take the minimum range (closest obstacle).
    6. Return a :class:`LaserScan` with the binned ranges.
    """
    pts = cloud.points_xyz
    angle_min = np.deg2rad(cfg.angle_min_deg)
    angle_max = np.deg2rad(cfg.angle_max_deg)
    ang_res = np.deg2rad(cfg.angular_resolution_deg)
    n_bins = max(1, int(round((angle_max - angle_min) / ang_res)))
    # Re-derive exact angle_max to avoid floating-point edge issues
    angle_max_exact = angle_min + n_bins * ang_res

    # Initialize all bins to +inf so np.minimum.at works correctly;
    # bins that remain +inf are converted to NaN before returning.
    ranges = np.full(n_bins, np.inf, dtype=np.float64)

    def _make_scan(r: np.ndarray) -> LaserScan:
        """Build a LaserScan from the range array, converting +inf to NaN."""
        out = r.copy()
        out[~np.isfinite(out)] = np.nan
        return LaserScan(
            stamp_ns=cloud.stamp_ns,
            frame_id=cloud.frame_id,
            angle_min=float(angle_min),
            angle_max=float(angle_max_exact - ang_res),
            angle_increment=float(ang_res),
            ranges=out,
            range_min=0.0,
            range_max=cfg.max_range,
        )

    if pts.size == 0:
        return _make_scan(ranges)

    # 1. Filter by z band
    z = pts[:, 2]
    z_mask = (z >= cfg.z_min) & (z <= cfg.z_max)
    filtered = pts[z_mask]

    if filtered.shape[0] == 0:
        return _make_scan(ranges)

    # 2. Project to 2D
    x = filtered[:, 0]
    y = filtered[:, 1]

    # 3. Bearing and range
    bearings = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)

    # Filter by max range and positive range
    range_mask = (r > 0.0) & (r <= cfg.max_range)
    bearings = bearings[range_mask]
    r = r[range_mask]

    if r.size == 0:
        return _make_scan(ranges)

    # 4. Bin into angular cells
    bin_idx = ((bearings - angle_min) / ang_res).astype(int)
    # Clamp to valid range (handles edge cases at boundaries)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid]
    r = r[valid]

    # 5. For each cell, take the minimum range
    # Use np.minimum.at for efficient in-place reduction
    np.minimum.at(ranges, bin_idx, r)

    return _make_scan(ranges)
