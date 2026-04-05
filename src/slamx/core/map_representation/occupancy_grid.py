from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import LaserScan, Pose2


def _world_to_grid(x: float, y: float, *, origin_x: float, origin_y: float, res: float) -> tuple[int, int]:
    gx = int(np.floor((x - origin_x) / res))
    gy = int(np.floor((y - origin_y) / res))
    return gx, gy


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


@dataclass
class OccupancyGridConfig:
    resolution_m: float = 0.05
    size_x_m: float = 40.0
    size_y_m: float = 40.0
    # map frame origin (bottom-left) in meters
    origin_x_m: float = -20.0
    origin_y_m: float = -20.0
    # log-odds updates
    lo_occ: float = 0.85
    lo_free: float = -0.4
    lo_min: float = -5.0
    lo_max: float = 5.0


class OccupancyGridMap:
    """Simple occupancy grid with ray casting (CPU, deterministic).

    Stores log-odds in a fixed-size grid.
    """

    def __init__(self, cfg: OccupancyGridConfig | None = None) -> None:
        self.cfg = cfg or OccupancyGridConfig()
        self.res = float(self.cfg.resolution_m)
        self.w = int(np.ceil(self.cfg.size_x_m / self.res))
        self.h = int(np.ceil(self.cfg.size_y_m / self.res))
        self.origin_x = float(self.cfg.origin_x_m)
        self.origin_y = float(self.cfg.origin_y_m)
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)

    def update(self, *, pose_map: Pose2, scan: LaserScan) -> None:
        # sensor origin in map frame is the pose translation (no extrinsics in Phase 1)
        sx, sy = pose_map.x, pose_map.y
        sgi, sgj = _world_to_grid(sx, sy, origin_x=self.origin_x, origin_y=self.origin_y, res=self.res)
        if not (0 <= sgi < self.w and 0 <= sgj < self.h):
            return

        pts_s = scan.points_xy()
        if pts_s.size == 0:
            return

        c, s = float(np.cos(pose_map.theta)), float(np.sin(pose_map.theta))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts_m = (R @ pts_s.T).T + np.array([pose_map.x, pose_map.y], dtype=np.float64)

        for hx, hy in pts_m:
            gi, gj = _world_to_grid(hx, hy, origin_x=self.origin_x, origin_y=self.origin_y, res=self.res)
            if not (0 <= gi < self.w and 0 <= gj < self.h):
                continue
            cells = list(_bresenham(sgi, sgj, gi, gj))
            if len(cells) <= 1:
                continue
            # free along the ray excluding last cell, occupied at end
            for cx, cy in cells[:-1]:
                if 0 <= cx < self.w and 0 <= cy < self.h:
                    self.logodds[cy, cx] = np.clip(
                        self.logodds[cy, cx] + self.cfg.lo_free, self.cfg.lo_min, self.cfg.lo_max
                    )
            cx, cy = cells[-1]
            self.logodds[cy, cx] = np.clip(
                self.logodds[cy, cx] + self.cfg.lo_occ, self.cfg.lo_min, self.cfg.lo_max
            )

    def to_8bit_occupancy(self) -> np.ndarray:
        """Return uint8 image in [0,255], 0=occupied, 254=free, 205=unknown-ish."""
        p = 1.0 / (1.0 + np.exp(-self.logodds.astype(np.float64)))
        # ROS-ish: 0..100 with -1 unknown; here just map to grayscale
        img = (254.0 * (1.0 - p)).astype(np.uint8)
        return img

