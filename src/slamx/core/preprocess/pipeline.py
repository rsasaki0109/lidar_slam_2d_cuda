from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import LaserScan


@dataclass
class PreprocessConfig:
    min_range: float | None = None
    max_range: float | None = None
    stride: int = 1  # keep every k-th ray
    min_angle_deg: float | None = None
    max_angle_deg: float | None = None
    gradient_mask_diff_m: float | None = None
    gradient_mask_max_range: float | None = None
    gradient_mask_window: int = 0


def _apply_gradient_mask(ranges: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    threshold = cfg.gradient_mask_diff_m
    if threshold is None or not np.isfinite(threshold) or threshold <= 0.0 or ranges.size < 2:
        return ranges

    valid_pairs = np.isfinite(ranges[:-1]) & np.isfinite(ranges[1:])
    jump_idx = np.flatnonzero(valid_pairs & (np.abs(ranges[1:] - ranges[:-1]) >= float(threshold)))
    if jump_idx.size == 0:
        return ranges

    max_range = cfg.gradient_mask_max_range
    window = max(0, int(cfg.gradient_mask_window))
    mask = np.zeros(ranges.shape, dtype=bool)
    for idx in jump_idx:
        left = float(ranges[idx])
        right = float(ranges[idx + 1])
        mask_center = idx if left <= right else idx + 1
        if max_range is not None and (
            not np.isfinite(ranges[mask_center]) or ranges[mask_center] > float(max_range)
        ):
            continue
        lo = max(0, mask_center - window)
        hi = min(ranges.size, mask_center + window + 1)
        mask[lo:hi] = True

    if not np.any(mask):
        return ranges
    out = ranges.copy()
    out[mask] = np.nan
    return out


def preprocess_scan(scan: LaserScan, cfg: PreprocessConfig) -> LaserScan:
    r = scan.ranges.copy()
    if cfg.min_range is not None:
        r = np.where(np.isfinite(r) & (r < cfg.min_range), np.nan, r)
    if cfg.max_range is not None:
        r = np.where(np.isfinite(r) & (r > cfg.max_range), np.nan, r)
    angle_min = scan.angle_min
    angle_max = scan.angle_max
    angle_increment = scan.angle_increment
    if cfg.min_angle_deg is not None or cfg.max_angle_deg is not None:
        bearings = scan.bearings()
        min_angle = np.deg2rad(cfg.min_angle_deg) if cfg.min_angle_deg is not None else float(bearings[0])
        max_angle = np.deg2rad(cfg.max_angle_deg) if cfg.max_angle_deg is not None else float(bearings[-1])
        tol = max(abs(float(angle_increment)) * 0.5, 1e-12)
        keep = (bearings >= (min_angle - tol)) & (bearings <= (max_angle + tol))
        if np.any(keep):
            first = int(np.argmax(keep))
            last = int(r.size - np.argmax(keep[::-1]))
            r = r[first:last]
            angle_min = float(bearings[first])
            angle_max = float(bearings[last - 1])
        else:
            r = r[:0]
            angle_min = 0.0
            angle_max = 0.0
    r = _apply_gradient_mask(r, cfg)
    if cfg.stride > 1:
        r = r[:: cfg.stride]
        inc = angle_increment * cfg.stride
        n = r.size
        ang_max = angle_min + inc * max(0, n - 1)
        return LaserScan(
            stamp_ns=scan.stamp_ns,
            frame_id=scan.frame_id,
            angle_min=angle_min,
            angle_max=ang_max,
            angle_increment=inc,
            ranges=r,
            range_min=scan.range_min,
            range_max=scan.range_max,
        )
    return LaserScan(
        stamp_ns=scan.stamp_ns,
        frame_id=scan.frame_id,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
        ranges=r,
        range_min=scan.range_min,
        range_max=scan.range_max,
    )
