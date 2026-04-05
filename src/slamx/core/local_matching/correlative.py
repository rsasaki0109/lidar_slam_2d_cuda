from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import LaserScan, MatchResult, Pose2, transform_points_xy


@dataclass
class CorrelativeGridConfig:
    linear_step_m: float = 0.05
    angular_step_deg: float = 2.0
    linear_window_m: float = 0.5
    angular_window_deg: float = 30.0
    sigma_hit_m: float = 0.15  # Gaussian on ref points for scoring


class CorrelativeScanMatcher:
    """Coarse grid search + Gaussian hit-cost against reference points (map frame).

    Not Cartographer CSM: intentionally small, deterministic, and easy to instrument.
    """

    def __init__(self, cfg: CorrelativeGridConfig | None = None) -> None:
        self.cfg = cfg or CorrelativeGridConfig()

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        pts_s = scan.points_xy()
        if pts_s.size == 0:
            return MatchResult(
                pose_map=prediction_map,
                score=float("-inf"),
                candidates=[],
                diagnostics={"reason": "empty scan"},
            )

        if ref_points_xy_map.size == 0:
            return MatchResult(
                pose_map=prediction_map,
                score=0.0,
                candidates=[
                    (prediction_map.x, prediction_map.y, prediction_map.theta, 0.0)
                ],
                diagnostics={"reason": "no reference"},
            )

        ref = np.asarray(ref_points_xy_map, dtype=np.float64).reshape(-1, 2)
        sig = max(self.cfg.sigma_hit_m, 1e-6)

        lin = np.arange(
            -self.cfg.linear_window_m,
            self.cfg.linear_window_m + 1e-9,
            self.cfg.linear_step_m,
        )
        ang = np.deg2rad(
            np.arange(
                -self.cfg.angular_window_deg,
                self.cfg.angular_window_deg + 1e-9,
                self.cfg.angular_step_deg,
            )
        )

        best: tuple[float, Pose2] | None = None
        candidates: list[tuple[float, float, float, float]] = []

        for dy in lin:
            for dx in lin:
                for dth in ang:
                    cand = Pose2(
                        prediction_map.x + dx,
                        prediction_map.y + dy,
                        prediction_map.theta + dth,
                    )
                    T_ms = cand.as_se2()
                    pts_m = transform_points_xy(T_ms, pts_s)

                    # Score: mean log-likelihood under NN distance to refs (soft)
                    diff = pts_m[:, None, :] - ref[None, :, :]
                    d2 = np.sum(diff * diff, axis=2)
                    nn = np.min(d2, axis=1)
                    score = float(np.mean(-nn / (2 * sig**2)))

                    candidates.append((cand.x, cand.y, cand.theta, score))
                    if best is None or score > best[0]:
                        best = (score, cand)

        assert best is not None
        # stable sort by score for telemetry
        candidates.sort(key=lambda t: t[3], reverse=True)
        top_k = candidates[: min(50, len(candidates))]
        return MatchResult(
            pose_map=best[1],
            score=best[0],
            candidates=top_k,
            diagnostics={"grid": {"lin": lin.size, "ang": ang.size}},
        )
