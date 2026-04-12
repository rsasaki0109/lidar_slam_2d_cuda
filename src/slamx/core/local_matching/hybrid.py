from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from slamx.core.local_matching.correlative import CorrelativeGridConfig, CorrelativeScanMatcher
from slamx.core.local_matching.icp import IcpConfig, IcpScanMatcher
from slamx.core.types import LaserScan, MatchResult, Pose2


@dataclass
class HybridFallbackConfig:
    enabled: bool = False
    trigger_score: float = -0.01
    min_score_gain: float = 0.0
    correlative: CorrelativeGridConfig = field(
        default_factory=lambda: CorrelativeGridConfig(
            linear_step_m=0.05,
            angular_step_deg=2.0,
            linear_window_m=0.20,
            angular_window_deg=15.0,
            sigma_hit_m=0.10,
        )
        )


@dataclass
class HybridRefinementConfig:
    top_k: int = 1
    min_linear_dist_m: float = 0.0
    min_angular_dist_deg: float = 0.0


def _candidate_pose(candidate: tuple[float, float, float, float]) -> Pose2:
    return Pose2(float(candidate[0]), float(candidate[1]), float(candidate[2]))


def _angular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


class HybridScanMatcher:
    """Correlative coarse alignment followed by ICP refinement."""

    def __init__(
        self,
        correlative: CorrelativeGridConfig | None = None,
        icp: IcpConfig | None = None,
        refinement: HybridRefinementConfig | None = None,
        fallback: HybridFallbackConfig | None = None,
    ) -> None:
        self._coarse = CorrelativeScanMatcher(correlative)
        self._refine = IcpScanMatcher(icp)
        self._refinement_cfg = refinement or HybridRefinementConfig()
        self._fallback_cfg = fallback or HybridFallbackConfig()
        self._fallback_coarse = (
            CorrelativeScanMatcher(self._fallback_cfg.correlative)
            if self._fallback_cfg.enabled
            else None
        )

    def _select_refinement_predictions(self, coarse: MatchResult) -> list[Pose2]:
        top_k = max(1, int(self._refinement_cfg.top_k))
        min_linear = max(0.0, float(self._refinement_cfg.min_linear_dist_m))
        min_angular = math.radians(max(0.0, float(self._refinement_cfg.min_angular_dist_deg)))

        selected: list[Pose2] = [coarse.pose_map]
        if top_k <= 1:
            return selected

        for cand in coarse.candidates:
            pose = _candidate_pose(cand)
            duplicate = False
            for prev in selected:
                if (
                    math.hypot(pose.x - prev.x, pose.y - prev.y) <= min_linear
                    and _angular_distance(pose.theta, prev.theta) <= min_angular
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(pose)
            if len(selected) >= top_k:
                break
        return selected

    def _refine_candidates(
        self,
        *,
        scan: LaserScan,
        ref_points_xy_map: np.ndarray,
        predictions: list[Pose2],
    ) -> tuple[MatchResult, list[dict[str, object]], int]:
        best_idx = 0
        best_mr: MatchResult | None = None
        diags: list[dict[str, object]] = []
        for idx, pred in enumerate(predictions):
            mr = self._refine.match(
                scan=scan,
                prediction_map=pred,
                ref_points_xy_map=ref_points_xy_map,
            )
            icp_diag = mr.diagnostics.get("icp", {}) if isinstance(mr.diagnostics, dict) else {}
            diags.append(
                {
                    "prediction": {"x": pred.x, "y": pred.y, "theta": pred.theta},
                    "score": float(mr.score),
                    "final_rms": icp_diag.get("final_rms"),
                }
            )
            if best_mr is None or mr.score > best_mr.score:
                best_idx = idx
                best_mr = mr
        assert best_mr is not None
        return best_mr, diags, best_idx

    def _match_once(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
        coarse_matcher: CorrelativeScanMatcher,
    ) -> tuple[MatchResult, MatchResult, list[dict[str, object]], int]:
        coarse = coarse_matcher.match(
            scan=scan,
            prediction_map=prediction_map,
            ref_points_xy_map=ref_points_xy_map,
        )
        predictions = self._select_refinement_predictions(coarse)
        refined, refined_diags, best_idx = self._refine_candidates(
            scan=scan,
            ref_points_xy_map=ref_points_xy_map,
            predictions=predictions,
        )
        return coarse, refined, refined_diags, best_idx

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        coarse, refined, refined_diags, best_idx = self._match_once(
            scan=scan,
            prediction_map=prediction_map,
            ref_points_xy_map=ref_points_xy_map,
            coarse_matcher=self._coarse,
        )
        used_fallback = False
        fallback_diag: dict[str, object] = {"attempted": False}

        if (
            self._fallback_coarse is not None
            and refined.score <= float(self._fallback_cfg.trigger_score)
        ):
            fallback_diag["attempted"] = True
            fallback_coarse, fallback_refined, fallback_diags, fallback_best_idx = self._match_once(
                scan=scan,
                prediction_map=prediction_map,
                ref_points_xy_map=ref_points_xy_map,
                coarse_matcher=self._fallback_coarse,
            )
            fallback_diag.update(
                {
                    "trigger_score": float(self._fallback_cfg.trigger_score),
                    "primary_refined_score": float(refined.score),
                    "fallback_coarse_score": float(fallback_coarse.score),
                    "fallback_refined_score": float(fallback_refined.score),
                    "min_score_gain": float(self._fallback_cfg.min_score_gain),
                    "fallback_best_candidate_index": int(fallback_best_idx),
                    "fallback_refined_candidates": fallback_diags,
                }
            )
            if fallback_refined.score > refined.score + float(self._fallback_cfg.min_score_gain):
                coarse = fallback_coarse
                refined = fallback_refined
                refined_diags = fallback_diags
                best_idx = fallback_best_idx
                used_fallback = True

        return MatchResult(
            pose_map=refined.pose_map,
            score=refined.score,
            candidates=refined.candidates,
            diagnostics={
                "hybrid": {
                    "coarse_score": float(coarse.score),
                    "refined_score": float(refined.score),
                    "used_fallback": used_fallback,
                    "n_refined_candidates": len(refined_diags),
                    "best_candidate_index": int(best_idx),
                },
                "coarse": coarse.diagnostics,
                "refined": refined.diagnostics,
                "refined_candidates": refined_diags,
                "fallback": fallback_diag,
            },
        )
