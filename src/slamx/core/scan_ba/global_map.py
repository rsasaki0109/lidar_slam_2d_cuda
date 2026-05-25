"""Persistent global TSDF map for the scan-BA frontend (P-map).

The fixed-lag tracker deliberately runs against a *crisp local submap* rebuilt each
scan (a persistent global accumulation blurs surfaces as the robot moves and makes the
tracker lag -- see `engine._rebuild_local_map`). That local map is throwaway, so the
SLAM never produced a single coherent map of the whole run.

`GlobalTsdfMap` is that missing artifact: one large persistent TSDF that every accepted
scan is folded into at its current pose. Crucially it is kept *separate* from the
tracking map, so building it cannot feed back into and degrade tracking. When loop
closure corrects the trajectory, the global map is rebuilt from scratch at the
corrected poses (`rebuild`) so the map stays consistent with the optimized graph --
the online incremental fold alone would bake in the pre-correction drift.
"""
from __future__ import annotations

import numpy as np

from slamx.core.scan_ba.tsdf import Tsdf2D, Tsdf2DConfig
from slamx.core.scan_ba.tsdf_update import update_tsdf_from_scan
from slamx.core.types import Pose2


class GlobalTsdfMap:
    """A persistent, loop-closure-consistent global TSDF accumulated over all scans."""

    def __init__(self, cfg: Tsdf2DConfig, *, weight_inc: float = 1.0, weight_max: float = 100.0):
        self.cfg = cfg
        self.weight_inc = float(weight_inc)
        self.weight_max = float(weight_max)
        self.tsdf = Tsdf2D.zeros(cfg)

    def integrate(self, pose: Pose2, pts_sensor: np.ndarray) -> None:
        """Fold one scan (sensor-frame (N,2) points) into the map at `pose`."""
        if pts_sensor.shape[0] == 0:
            return
        update_tsdf_from_scan(
            self.tsdf,
            pose_map=pose,
            points_sensor=pts_sensor,
            weight_inc=self.weight_inc,
            weight_max=self.weight_max,
        )

    def rebuild(self, poses: list[Pose2], scans: list[np.ndarray]) -> None:
        """Clear and re-fold every scan at its (corrected) pose -- call after a loop
        closure / final pose-graph optimization so the map matches the optimized graph."""
        self.tsdf.phi[:] = 0.0
        self.tsdf.weight[:] = 0.0
        for pose, pts in zip(poses, scans):
            self.integrate(pose, pts)

    def to_occupancy_u8(self, *, occupied_band_m: float | None = None) -> np.ndarray:
        """Render the TSDF as a ROS-style 8-bit occupancy image (north-up).

        254 = free, 0 = occupied (on/inside the zero-crossing surface), 205 = unknown
        (never observed). `occupied_band_m` defaults to one cell: cells whose signed
        distance is within +band of the surface (or negative) are marked occupied.
        """
        band = float(occupied_band_m if occupied_band_m is not None else self.cfg.resolution_m)
        phi = self.tsdf.phi
        wt = self.tsdf.weight
        img = np.full(phi.shape, 205, dtype=np.uint8)  # unknown
        seen = wt > 0
        occupied = seen & (phi <= band)
        free = seen & (phi > band)
        img[free] = 254
        img[occupied] = 0
        # ROS map_server convention: image row 0 is the top (max y), so flip vertically.
        return np.flipud(img)
