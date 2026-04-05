from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LaserScan:
    stamp_ns: int | None
    frame_id: str
    angle_min: float
    angle_max: float
    angle_increment: float
    ranges: np.ndarray
    range_min: float = 0.0
    range_max: float = float("inf")

    def __post_init__(self) -> None:
        self.ranges = np.asarray(self.ranges, dtype=np.float64).reshape(-1)

    def valid_mask(self) -> np.ndarray:
        r = self.ranges
        return np.isfinite(r) & (r >= self.range_min) & (r <= self.range_max)

    def bearings(self) -> np.ndarray:
        n = self.ranges.size
        return self.angle_min + np.arange(n, dtype=np.float64) * self.angle_increment

    def points_xy(self) -> np.ndarray:
        """Points in scan (sensor) frame, shape (N, 2)."""
        b = self.bearings()
        r = self.ranges
        m = self.valid_mask()
        cb, sb = np.cos(b[m]), np.sin(b[m])
        return np.column_stack((r[m] * cb, r[m] * sb))


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    theta: float

    def as_se2(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s, self.x], [s, c, self.y], [0, 0, 1]], dtype=np.float64)

    def compose(self, local: Pose2) -> Pose2:
        t = self.as_se2() @ local.as_se2()
        return Pose2(float(t[0, 2]), float(t[1, 2]), float(np.arctan2(t[1, 0], t[0, 0])))

    def inverse(self) -> Pose2:
        c, s = np.cos(self.theta), np.sin(self.theta)
        xi = -c * self.x - s * self.y
        yi = s * self.x - c * self.y
        return Pose2(float(xi), float(yi), float(-self.theta))


def transform_points_xy(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE2 (3x3) to Nx2 points."""
    if pts.size == 0:
        return pts
    h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ h.T).T
    return out[:, :2]


@dataclass
class MatchResult:
    pose_map: Pose2
    score: float
    candidates: list[tuple[float, float, float, float]]  # x, y, theta, score
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Submap:
    id: int
    node_id: int
    origin: Pose2
    hit_xy_map: np.ndarray  # (K, 2) in map frame
