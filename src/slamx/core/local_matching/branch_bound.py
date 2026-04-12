"""Cartographer-style multi-resolution branch-and-bound correlative scan matcher.

Implements the key algorithm from Hess et al., 2016:
1. Build multi-resolution probability grid (likelihood field) from reference points
2. For each angular candidate, rotate scan points
3. Branch-and-bound search over (x, y) translations with pruning
4. Return best (x, y, theta) with score
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from slamx.core.local_matching.icp import IcpConfig, IcpScanMatcher
from slamx.core.local_matching.hybrid import HybridRefinementConfig
from slamx.core.types import LaserScan, MatchResult, Pose2


@dataclass
class BranchBoundConfig:
    resolution_m: float = 0.05
    n_levels: int = 4
    linear_window_m: float = 0.3
    angular_window_deg: float = 20.0
    angular_step_deg: float = 1.0
    sigma_hit_m: float = 0.10
    min_score: float = -0.5


class ProbabilityGrid:
    """Multi-resolution probability grid (likelihood field) from reference points."""

    def __init__(
        self,
        ref_points_xy: np.ndarray,
        resolution: float,
        sigma: float,
        origin: tuple[float, float],
        size: tuple[int, int],
        n_levels: int,
    ) -> None:
        self.resolution = resolution
        self.origin = origin
        self.n_levels = n_levels
        self.grids: list[np.ndarray] = []

        ny, nx = size

        # Level 0: rasterize reference points onto binary grid, then blur
        grid0 = np.zeros((ny, nx), dtype=np.float32)
        if ref_points_xy.size > 0:
            ref = ref_points_xy.reshape(-1, 2)
            ix = ((ref[:, 0] - origin[0]) / resolution).astype(np.int32)
            iy = ((ref[:, 1] - origin[1]) / resolution).astype(np.int32)
            valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            grid0[iy[valid], ix[valid]] = 1.0

            # Gaussian blur (sigma in grid cells)
            sigma_cells = max(sigma / resolution, 0.5)
            grid0 = gaussian_filter(grid0, sigma=sigma_cells, mode="constant")

            # Normalize to [0, 1]
            mx = grid0.max()
            if mx > 0:
                grid0 /= mx

        self.grids.append(grid0)

        # Levels 1..n_levels-1: 2x2 max-pooling (upper bounds for B&B)
        for _level in range(1, n_levels):
            prev = self.grids[-1]
            h, w = prev.shape
            # Pad to even dimensions if needed
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                prev = np.pad(prev, ((0, pad_h), (0, pad_w)), mode="constant")
                h, w = prev.shape
            h2, w2 = h // 2, w // 2
            coarse = np.maximum(
                np.maximum(prev[0::2, 0::2][:h2, :w2], prev[0::2, 1::2][:h2, :w2]),
                np.maximum(prev[1::2, 0::2][:h2, :w2], prev[1::2, 1::2][:h2, :w2]),
            )
            self.grids.append(coarse)

    def score_at(self, points_xy: np.ndarray, level: int = 0) -> float:
        """Score scan points against probability grid at given level.

        Returns mean probability across all points.
        Points outside grid get score 0.
        """
        res = self.resolution * (2**level)
        grid = self.grids[level]
        # Convert points to grid coordinates
        ix = ((points_xy[:, 0] - self.origin[0]) / res).astype(np.int32)
        iy = ((points_xy[:, 1] - self.origin[1]) / res).astype(np.int32)
        # Bounds check
        valid = (ix >= 0) & (ix < grid.shape[1]) & (iy >= 0) & (iy < grid.shape[0])
        if not np.any(valid):
            return 0.0
        scores = grid[iy[valid], ix[valid]]
        # Out-of-bound points score 0; average over all points
        total = float(np.sum(scores))
        return total / len(points_xy)


class BranchBoundScanMatcher:
    """Multi-resolution branch-and-bound correlative scan matcher.

    Implements the key algorithm from Cartographer (Hess et al., 2016):
    1. Build multi-resolution probability grid from reference points
    2. For each angular candidate, rotate scan points
    3. Branch-and-bound search over (x, y) translations:
       - Start at coarsest level
       - Compute upper bound score from max-pooled grid
       - Prune branches where upper_bound < current_best
       - Refine surviving branches to finer levels
    4. Return best (x, y, theta) with score
    """

    def __init__(self, cfg: BranchBoundConfig | None = None) -> None:
        self.cfg = cfg or BranchBoundConfig()

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        pts_s = scan.points_xy()
        if pts_s.size == 0 or ref_points_xy_map.size == 0:
            return MatchResult(
                pose_map=prediction_map,
                score=float("-inf"),
                candidates=[],
                diagnostics={"reason": "empty"},
            )

        ref = ref_points_xy_map.reshape(-1, 2)

        # 1. Compute grid bounds from prediction + search window + scan extent
        px, py = prediction_map.x, prediction_map.y
        window = self.cfg.linear_window_m
        scan_extent = float(np.max(np.linalg.norm(pts_s, axis=1))) if pts_s.size > 0 else 0.0
        margin = scan_extent + window + 0.5

        x_min = px - margin
        y_min = py - margin
        x_max = px + margin
        y_max = py + margin

        nx = int(np.ceil((x_max - x_min) / self.cfg.resolution_m)) + 1
        ny = int(np.ceil((y_max - y_min) / self.cfg.resolution_m)) + 1

        # 2. Build probability grid
        prob_grid = ProbabilityGrid(
            ref,
            self.cfg.resolution_m,
            self.cfg.sigma_hit_m,
            origin=(x_min, y_min),
            size=(ny, nx),
            n_levels=self.cfg.n_levels,
        )

        # 3. Angular candidates
        ang = np.deg2rad(
            np.arange(
                -self.cfg.angular_window_deg,
                self.cfg.angular_window_deg + 1e-9,
                self.cfg.angular_step_deg,
            )
        )

        best_score = float("-inf")
        best_pose = prediction_map

        # 4. For each angle, branch-and-bound over (x, y)
        for dth in ang:
            theta = prediction_map.theta + float(dth)
            c, s = np.cos(theta), np.sin(theta)
            # Rotate scan points once per angle
            rot = np.array([[c, -s], [s, c]], dtype=np.float64)
            pts_rot = pts_s @ rot.T

            # Branch and bound from coarsest to finest level
            score, tx, ty = self._branch_and_bound(
                prob_grid, pts_rot, px, py, window, best_score
            )

            if score > best_score:
                best_score = score
                best_pose = Pose2(tx, ty, theta)

        return MatchResult(
            pose_map=best_pose,
            score=best_score,
            candidates=[(best_pose.x, best_pose.y, best_pose.theta, best_score)],
            diagnostics={
                "branch_bound": {
                    "n_levels": self.cfg.n_levels,
                    "resolution_m": self.cfg.resolution_m,
                    "window_m": self.cfg.linear_window_m,
                }
            },
        )

    def _branch_and_bound(
        self,
        prob_grid: ProbabilityGrid,
        pts_rot: np.ndarray,
        cx: float,
        cy: float,
        window: float,
        global_best: float,
    ) -> tuple[float, float, float]:
        """Run B&B search over translations for rotated scan points."""
        top_level = prob_grid.n_levels - 1
        coarsest_res = prob_grid.resolution * (2**top_level)

        # Generate initial candidates at coarsest level
        n_steps = int(np.ceil(window / coarsest_res))
        candidates: list[tuple[float, float, float]] = []  # (upper_bound, tx, ty)
        for ix in range(-n_steps, n_steps + 1):
            for iy in range(-n_steps, n_steps + 1):
                tx = cx + ix * coarsest_res
                ty = cy + iy * coarsest_res
                translated = pts_rot + np.array([tx, ty])
                upper = prob_grid.score_at(translated, level=top_level)
                if upper > global_best:
                    candidates.append((upper, tx, ty))

        # Sort by upper bound (best first)
        candidates.sort(key=lambda c: -c[0])

        best_score = global_best
        best_tx, best_ty = cx, cy

        # Refine through levels (coarse to fine)
        for level in range(top_level - 1, -1, -1):
            res = prob_grid.resolution * (2**level)
            half_res = res * 0.5
            next_candidates: list[tuple[float, float, float]] = []
            for upper, tx, ty in candidates:
                if upper <= best_score:
                    break  # pruned (sorted, so all remaining are worse)
                # Expand to 4 children
                for dx in (-half_res, half_res):
                    for dy in (-half_res, half_res):
                        child_tx = tx + dx
                        child_ty = ty + dy
                        translated = pts_rot + np.array([child_tx, child_ty])
                        child_score = prob_grid.score_at(translated, level=level)
                        if child_score > best_score:
                            if level == 0:
                                best_score = child_score
                                best_tx, best_ty = child_tx, child_ty
                            else:
                                next_candidates.append(
                                    (child_score, child_tx, child_ty)
                                )

            if level > 0:
                next_candidates.sort(key=lambda c: -c[0])
                candidates = next_candidates

        return best_score, best_tx, best_ty


class HybridBBScanMatcher:
    """Branch-and-bound coarse alignment followed by ICP refinement."""

    def __init__(
        self,
        branch_bound: BranchBoundConfig | None = None,
        icp: IcpConfig | None = None,
        refinement: HybridRefinementConfig | None = None,
    ) -> None:
        self._coarse = BranchBoundScanMatcher(branch_bound)
        self._refine = IcpScanMatcher(icp)
        self._refinement_cfg = refinement or HybridRefinementConfig()

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        coarse = self._coarse.match(
            scan=scan,
            prediction_map=prediction_map,
            ref_points_xy_map=ref_points_xy_map,
        )

        # Refine the best coarse result with ICP
        refined = self._refine.match(
            scan=scan,
            prediction_map=coarse.pose_map,
            ref_points_xy_map=ref_points_xy_map,
        )

        return MatchResult(
            pose_map=refined.pose_map,
            score=refined.score,
            candidates=refined.candidates,
            diagnostics={
                "hybrid_bb": {
                    "coarse_score": float(coarse.score),
                    "refined_score": float(refined.score),
                },
                "coarse": coarse.diagnostics,
                "refined": refined.diagnostics,
            },
        )
