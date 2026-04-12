"""Ring-free 1D scan context descriptor for relocalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import LaserScan


@dataclass
class ScanContextConfig:
    n_sectors: int = 60  # number of angular bins
    max_range: float = 10.0  # clip ranges at this value


class ScanContextDescriptor:
    """Ring-free 1D scan context: per-sector mean range.

    Each sector covers ``2*pi / n_sectors`` radians.  For each sector the
    mean valid range is recorded (0 if no valid rays fall into the sector).
    The descriptor is normalised by ``max_range`` so values lie in [0, 1].

    Rotation invariance is achieved by trying all circular shifts when
    comparing two descriptors.
    """

    def __init__(self, cfg: ScanContextConfig | None = None) -> None:
        self.cfg = cfg or ScanContextConfig()

    # ------------------------------------------------------------------ #
    # Descriptor computation
    # ------------------------------------------------------------------ #

    def compute(self, scan: LaserScan) -> np.ndarray:
        """Compute descriptor: ``(n_sectors,)`` array of normalised ranges per sector."""
        n_sectors = int(self.cfg.n_sectors)
        max_range = float(self.cfg.max_range)

        bearings = scan.bearings()
        ranges = scan.ranges.copy()
        valid = scan.valid_mask()

        if not np.any(valid):
            return np.zeros(n_sectors, dtype=np.float64)

        bearings_v = bearings[valid]
        ranges_v = ranges[valid]

        # Clip ranges
        ranges_v = np.clip(ranges_v, 0.0, max_range)

        # Map bearings to [0, 2*pi)
        bearings_norm = bearings_v % (2.0 * np.pi)

        # Sector width
        sector_width = 2.0 * np.pi / n_sectors

        # Bin indices
        bin_idx = np.floor(bearings_norm / sector_width).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_sectors - 1)

        # Accumulate mean range per sector
        desc = np.zeros(n_sectors, dtype=np.float64)
        counts = np.zeros(n_sectors, dtype=np.float64)
        np.add.at(desc, bin_idx, ranges_v)
        np.add.at(counts, bin_idx, 1.0)

        mask = counts > 0
        desc[mask] /= counts[mask]

        # Normalise by max_range
        desc /= max_range

        return desc

    # ------------------------------------------------------------------ #
    # Distance / matching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def distance(self, desc_a: np.ndarray, desc_b: np.ndarray) -> float:
        """Cosine distance between two descriptors, invariant to rotation.

        Tries all column shifts (rotations) and returns the minimum distance.

        ``distance = 1 - max_shift(cosine_similarity(a, shift(b)))``
        """
        _, shift = self.best_shift(desc_a, desc_b)
        # Re-use best_shift which already computes the answer
        best_sim = self._cosine_similarity(desc_a, np.roll(desc_b, shift))
        return float(1.0 - best_sim)

    def best_shift(self, desc_a: np.ndarray, desc_b: np.ndarray) -> tuple[float, int]:
        """Return ``(distance, best_shift_sectors)`` for rotation estimation.

        The shift is the number of sectors ``desc_b`` must be rolled (via
        ``np.roll``) to best align with ``desc_a``.
        """
        n = len(desc_a)
        na = np.linalg.norm(desc_a)
        nb = np.linalg.norm(desc_b)
        if na < 1e-12 or nb < 1e-12:
            return (1.0, 0)

        # Use circular cross-correlation via FFT for efficiency
        fa = np.fft.rfft(desc_a)
        fb = np.fft.rfft(desc_b)
        cross = np.fft.irfft(fa * np.conj(fb), n=n)
        # cross[k] = sum_i a[i] * b[i-k]  (i.e. b rolled by k)
        best_k = int(np.argmax(cross))
        best_sim = float(cross[best_k] / (na * nb))
        best_sim = min(max(best_sim, -1.0), 1.0)
        return (1.0 - best_sim, best_k)
