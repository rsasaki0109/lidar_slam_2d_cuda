from __future__ import annotations

import numpy as np


def compute_range_weights(scan_ranges: np.ndarray, mode: str, min_m: float) -> np.ndarray | None:
    """Compute per-point weights based on sensor range.

    Modes:
      - "none": no weighting (returns None)
      - "linear": w = clamp(range / min_m, 0, 1)
      - "sigmoid": smooth sigmoid transition around min_m
    """
    if mode == "none" or min_m <= 0.0:
        return None
    if mode == "linear":
        w = np.clip(scan_ranges / min_m, 0.0, 1.0)
    elif mode == "sigmoid":
        w = 1.0 / (1.0 + np.exp(-5.0 * (scan_ranges / min_m - 1.0)))
    else:
        return None
    if float(np.sum(w)) < 1e-12:
        return None
    return w
