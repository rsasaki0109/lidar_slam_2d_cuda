from __future__ import annotations

import math

import numpy as np
import pytest

from slamx.core.scan_ba import Tsdf2DConfig, align_scan_to_tsdf
from slamx.core.scan_ba.tsdf import build_tsdf_from_signed_distance
from slamx.core.types import Pose2


def _l_room_sdf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L-room: walls at y=5 (lower side free) and x=5 (left side free).

    For points with x < 5 and y < 5 (the free quadrant), phi = min(5 - x, 5 - y).
    Elsewhere we extend the SDF as the (negative) distance to the wall set so
    that gradients still point back toward the free region.
    """
    # signed distance with sign + on free side; negative outside
    # the wall set is the union of segments {x=5, y<=5} and {y=5, x<=5}.
    # For the free quadrant, distance to either wall is positive.
    dx = 5.0 - x  # >0 in free region
    dy = 5.0 - y
    free = (dx > 0) & (dy > 0)
    free_dist = np.minimum(dx, dy)
    # for x>=5 or y>=5, compute negative distance to the wall set (approx)
    # distance to point (5,5) when both x>=5 and y>=5
    both_outside = (dx <= 0) & (dy <= 0)
    only_x_outside = (dx <= 0) & (dy > 0)
    only_y_outside = (dy <= 0) & (dx > 0)
    out = np.zeros_like(x)
    out[free] = free_dist[free]
    out[only_x_outside] = -(-dx[only_x_outside])  # = dx (which is <=0)
    out[only_y_outside] = -(-dy[only_y_outside])
    if np.any(both_outside):
        out[both_outside] = -np.hypot(-dx[both_outside], -dy[both_outside])
    return out


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


def _raycast_scan(pose: Pose2, n_beams: int = 360, range_max: float = 8.0) -> np.ndarray:
    """Ray-cast a scan from `pose` against the L-room walls.

    Returns points in the sensor (scan) frame.
    """
    angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
    cos_a = np.cos(angles + pose.theta)
    sin_a = np.sin(angles + pose.theta)
    # intersection with wall x=5: t = (5 - pose.x) / cos_a, require t>0 and resulting y<=5
    # intersection with wall y=5: t = (5 - pose.y) / sin_a, require t>0 and resulting x<=5
    t_x = np.where(cos_a > 1e-9, (5.0 - pose.x) / np.where(cos_a != 0, cos_a, 1.0), np.inf)
    t_y = np.where(sin_a > 1e-9, (5.0 - pose.y) / np.where(sin_a != 0, sin_a, 1.0), np.inf)
    # validate: hit point must lie on segment (the other coordinate ≤ 5)
    hit_x_y = pose.y + t_x * sin_a
    hit_y_x = pose.x + t_y * cos_a
    t_x = np.where((t_x > 0) & (hit_x_y <= 5.0), t_x, np.inf)
    t_y = np.where((t_y > 0) & (hit_y_x <= 5.0), t_y, np.inf)
    t = np.minimum(t_x, t_y)
    valid = np.isfinite(t) & (t <= range_max) & (t > 0.05)

    # sensor-frame points
    sx = np.cos(angles) * t
    sy = np.sin(angles) * t
    return np.column_stack((sx[valid], sy[valid]))


def test_align_recovers_ground_truth_pose():
    tsdf = _make_l_room_tsdf()
    gt = Pose2(x=2.0, y=1.5, theta=0.3)
    scan_xy = _raycast_scan(gt)
    assert scan_xy.shape[0] > 50, "L-room raycast should produce many beams"

    # perturbed initial guess
    init = Pose2(x=gt.x + 0.18, y=gt.y - 0.12, theta=gt.theta - 0.07)
    result = align_scan_to_tsdf(
        tsdf=tsdf,
        scan_xy=scan_xy,
        pose_init=init,
        max_iters=40,
    )

    assert result.converged, f"failed to converge: {result.diagnostics}"
    assert result.num_inliers > 50
    assert math.hypot(result.pose.x - gt.x, result.pose.y - gt.y) < 0.02
    assert abs(result.pose.theta - gt.theta) < 0.01


def test_align_returns_unconverged_when_outside_tsdf():
    tsdf = _make_l_room_tsdf()
    # scan that lies far outside the TSDF bounded region
    scan_xy = np.array([[100.0, 100.0], [101.0, 100.5]], dtype=np.float64)
    result = align_scan_to_tsdf(
        tsdf=tsdf,
        scan_xy=scan_xy,
        pose_init=Pose2(0.0, 0.0, 0.0),
        max_iters=5,
    )
    assert not result.converged
    assert result.num_inliers == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
