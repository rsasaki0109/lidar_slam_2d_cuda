from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import LaserScan


@dataclass
class PreprocessConfig:
    max_range: float | None = None
    stride: int = 1  # keep every k-th ray


def preprocess_scan(scan: LaserScan, cfg: PreprocessConfig) -> LaserScan:
    r = scan.ranges.copy()
    if cfg.max_range is not None:
        r = np.where(np.isfinite(r) & (r > cfg.max_range), np.nan, r)
    if cfg.stride > 1:
        r = r[:: cfg.stride]
        inc = scan.angle_increment * cfg.stride
        n = r.size
        ang_max = scan.angle_min + inc * max(0, n - 1)
        return LaserScan(
            stamp_ns=scan.stamp_ns,
            frame_id=scan.frame_id,
            angle_min=scan.angle_min,
            angle_max=ang_max,
            angle_increment=inc,
            ranges=r,
            range_min=scan.range_min,
            range_max=scan.range_max,
        )
    return LaserScan(
        stamp_ns=scan.stamp_ns,
        frame_id=scan.frame_id,
        angle_min=scan.angle_min,
        angle_max=scan.angle_max,
        angle_increment=scan.angle_increment,
        ranges=r,
        range_min=scan.range_min,
        range_max=scan.range_max,
    )
