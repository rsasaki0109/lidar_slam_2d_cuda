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
    selection_translation_weight: float = 0.0
    selection_rotation_weight: float = 0.0
    # Opt-in narrow tiebreaker: when scores are essentially tied and ICP RMS
    # is not worse, prefer the refined candidate with the smallest prediction
    # yaw delta. Default OFF (enabled=False).
    prediction_yaw_tiebreak_enabled: bool = False
    tiebreak_score_eps: float = 0.0
    tiebreak_rms_eps: float = 0.0
    tiebreak_yaw_margin_rad: float = 0.0


def _candidate_pose(candidate: tuple[float, float, float, float]) -> Pose2:
    return Pose2(float(candidate[0]), float(candidate[1]), float(candidate[2]))


def _angular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _refinement_selection_score(
    cfg: HybridRefinementConfig,
    *,
    match_score: float,
    pose: Pose2,
    prediction: Pose2,
) -> tuple[float, float, float]:
    trans_delta = math.hypot(pose.x - prediction.x, pose.y - prediction.y)
    yaw_delta = _angular_distance(pose.theta, prediction.theta)
    score = (
        float(match_score)
        - float(cfg.selection_translation_weight) * trans_delta
        - float(cfg.selection_rotation_weight) * yaw_delta
    )
    return score, trans_delta, yaw_delta


def _final_rms_of(diag: dict[str, object]) -> float | None:
    rms = diag.get("final_rms")
    if rms is None:
        return None
    try:
        return float(rms)
    except (TypeError, ValueError):
        return None


def _apply_prediction_yaw_tiebreak(
    cfg: HybridRefinementConfig,
    diags: list[dict[str, object]],
    best_idx: int,
) -> tuple[int, dict[str, object]]:
    """If scores are tied and ICP RMS is not worse, swap to the candidate with
    the smallest prediction yaw delta. Returns (new_best_idx, tiebreak_diag)."""
    tiebreak_diag: dict[str, object] = {
        "enabled": bool(cfg.prediction_yaw_tiebreak_enabled),
        "applied": False,
        "from_index": int(best_idx),
        "to_index": int(best_idx),
    }
    if not cfg.prediction_yaw_tiebreak_enabled:
        return best_idx, tiebreak_diag
    if len(diags) < 2:
        return best_idx, tiebreak_diag

    score_eps = float(cfg.tiebreak_score_eps)
    rms_eps = float(cfg.tiebreak_rms_eps)
    yaw_margin = float(cfg.tiebreak_yaw_margin_rad)

    best_diag = diags[best_idx]
    best_sel_score = float(best_diag.get("selection_score", float("-inf")))
    best_yaw = float(best_diag.get("prediction_delta_yaw_rad", 0.0))
    best_rms = _final_rms_of(best_diag)

    chosen_idx = best_idx
    chosen_yaw = best_yaw
    for idx, d in enumerate(diags):
        if idx == best_idx:
            continue
        sel = float(d.get("selection_score", float("-inf")))
        if best_sel_score - sel > score_eps:
            continue
        cand_rms = _final_rms_of(d)
        if best_rms is not None and cand_rms is not None and cand_rms - best_rms > rms_eps:
            continue
        cand_yaw = float(d.get("prediction_delta_yaw_rad", 0.0))
        if chosen_yaw - cand_yaw <= yaw_margin:
            continue
        chosen_idx = idx
        chosen_yaw = cand_yaw

    if chosen_idx != best_idx:
        tiebreak_diag["applied"] = True
        tiebreak_diag["to_index"] = int(chosen_idx)
    return chosen_idx, tiebreak_diag


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
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
        predictions: list[Pose2],
    ) -> tuple[MatchResult, list[dict[str, object]], int, dict[str, object]]:
        best_idx = 0
        best_selection_score = float("-inf")
        mrs: list[MatchResult] = []
        diags: list[dict[str, object]] = []
        for idx, pred in enumerate(predictions):
            mr = self._refine.match(
                scan=scan,
                prediction_map=pred,
                ref_points_xy_map=ref_points_xy_map,
            )
            icp_diag = mr.diagnostics.get("icp", {}) if isinstance(mr.diagnostics, dict) else {}
            selection_score, pred_delta_m, pred_delta_yaw = _refinement_selection_score(
                self._refinement_cfg,
                match_score=float(mr.score),
                pose=mr.pose_map,
                prediction=prediction_map,
            )
            diags.append(
                {
                    "prediction": {"x": pred.x, "y": pred.y, "theta": pred.theta},
                    "score": float(mr.score),
                    "selection_score": float(selection_score),
                    "prediction_delta_m": float(pred_delta_m),
                    "prediction_delta_yaw_rad": float(pred_delta_yaw),
                    "final_rms": icp_diag.get("final_rms"),
                }
            )
            mrs.append(mr)
            if idx == 0 or selection_score > best_selection_score:
                best_idx = idx
                best_selection_score = selection_score
        assert mrs, "predictions must be non-empty"
        best_idx, tiebreak_diag = _apply_prediction_yaw_tiebreak(
            self._refinement_cfg, diags, best_idx
        )
        return mrs[best_idx], diags, best_idx, tiebreak_diag

    def _match_once(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
        coarse_matcher: CorrelativeScanMatcher,
    ) -> tuple[MatchResult, MatchResult, list[dict[str, object]], int, dict[str, object]]:
        coarse = coarse_matcher.match(
            scan=scan,
            prediction_map=prediction_map,
            ref_points_xy_map=ref_points_xy_map,
        )
        predictions = self._select_refinement_predictions(coarse)
        refined, refined_diags, best_idx, tiebreak_diag = self._refine_candidates(
            scan=scan,
            prediction_map=prediction_map,
            ref_points_xy_map=ref_points_xy_map,
            predictions=predictions,
        )
        return coarse, refined, refined_diags, best_idx, tiebreak_diag

    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult:
        coarse, refined, refined_diags, best_idx, tiebreak_diag = self._match_once(
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
            (
                fallback_coarse,
                fallback_refined,
                fallback_diags,
                fallback_best_idx,
                fallback_tiebreak_diag,
            ) = self._match_once(
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
                    "fallback_tiebreak": fallback_tiebreak_diag,
                }
            )
            if fallback_refined.score > refined.score + float(self._fallback_cfg.min_score_gain):
                coarse = fallback_coarse
                refined = fallback_refined
                refined_diags = fallback_diags
                best_idx = fallback_best_idx
                tiebreak_diag = fallback_tiebreak_diag
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
                "tiebreak": tiebreak_diag,
                "fallback": fallback_diag,
            },
        )
