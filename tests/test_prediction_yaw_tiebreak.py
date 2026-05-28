"""Unit tests for the narrow prediction-yaw tiebreak in HybridRefinementConfig."""

from __future__ import annotations

import math

import pytest

from slamx.core.local_matching.hybrid import (
    HybridRefinementConfig,
    _apply_prediction_yaw_tiebreak,
)


def _diag(
    *,
    selection_score: float,
    prediction_delta_yaw_rad: float,
    final_rms: float | None = 0.01,
) -> dict[str, object]:
    return {
        "selection_score": float(selection_score),
        "prediction_delta_yaw_rad": float(prediction_delta_yaw_rad),
        "final_rms": final_rms,
    }


def _tb_cfg(**overrides) -> HybridRefinementConfig:
    base = dict(
        prediction_yaw_tiebreak_enabled=True,
        tiebreak_score_eps=0.002,
        tiebreak_rms_eps=0.005,
        tiebreak_yaw_margin_rad=math.radians(0.5),
    )
    base.update(overrides)
    return HybridRefinementConfig(**base)


class TestApplyPredictionYawTiebreak:
    def test_disabled_keeps_best_idx(self) -> None:
        cfg = HybridRefinementConfig(prediction_yaw_tiebreak_enabled=False)
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.01),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 0
        assert td["applied"] is False
        assert td["enabled"] is False

    def test_single_candidate_is_noop(self) -> None:
        cfg = _tb_cfg()
        diags = [_diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10)]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 0
        assert td["applied"] is False

    def test_swaps_when_scores_tied_and_yaw_clearly_better(self) -> None:
        cfg = _tb_cfg()
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10, final_rms=0.005),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.01, final_rms=0.006),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 1
        assert td["applied"] is True
        assert td["from_index"] == 0
        assert td["to_index"] == 1

    def test_no_swap_when_score_gap_too_large(self) -> None:
        cfg = _tb_cfg(tiebreak_score_eps=0.0005)
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10),
            _diag(selection_score=-0.02, prediction_delta_yaw_rad=0.01),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 0
        assert td["applied"] is False

    def test_no_swap_when_rms_clearly_worse(self) -> None:
        cfg = _tb_cfg(tiebreak_rms_eps=0.001)
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10, final_rms=0.003),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.01, final_rms=0.020),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 0
        assert td["applied"] is False

    def test_no_swap_when_yaw_improvement_below_margin(self) -> None:
        cfg = _tb_cfg(tiebreak_yaw_margin_rad=math.radians(2.0))
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.05),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.045),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 0
        assert td["applied"] is False

    def test_missing_rms_does_not_block_swap(self) -> None:
        cfg = _tb_cfg()
        diags = [
            _diag(selection_score=-0.01, prediction_delta_yaw_rad=0.10, final_rms=None),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.01, final_rms=None),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 1
        assert td["applied"] is True

    def test_picks_overall_smallest_yaw_among_qualifying(self) -> None:
        cfg = _tb_cfg()
        diags = [
            _diag(selection_score=-0.010, prediction_delta_yaw_rad=0.12),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.05),
            _diag(selection_score=-0.011, prediction_delta_yaw_rad=0.02),
        ]
        idx, td = _apply_prediction_yaw_tiebreak(cfg, diags, best_idx=0)
        assert idx == 2
        assert td["applied"] is True


class TestMatcherIntegrationDiagnostics:
    """End-to-end sanity: tiebreak diagnostics show up in MatchResult.diagnostics."""

    def test_hybrid_bb_emits_tiebreak_block(self) -> None:
        import numpy as np

        from slamx.core.local_matching.branch_bound import (
            BranchBoundConfig,
            HybridBBScanMatcher,
        )
        from slamx.core.local_matching.icp import IcpConfig
        from slamx.core.types import LaserScan, Pose2

        # Tiny synthetic scan against a rectangle of reference points.
        ranges = np.linspace(1.0, 1.5, 60)
        scan = LaserScan(
            stamp_ns=0,
            frame_id="t",
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
            angle_increment=math.pi / 59,
            ranges=ranges,
            range_min=0.0,
            range_max=10.0,
        )
        ref = np.array(
            [[1.0 + 0.05 * i, 0.05 * j] for i in range(20) for j in range(-5, 6)],
            dtype=float,
        )
        matcher = HybridBBScanMatcher(
            branch_bound=BranchBoundConfig(
                resolution_m=0.1,
                n_levels=2,
                linear_window_m=0.2,
                angular_window_deg=10.0,
                angular_step_deg=2.0,
                sigma_hit_m=0.1,
            ),
            icp=IcpConfig(max_iterations=5),
            refinement=HybridRefinementConfig(
                top_k=3,
                min_angular_dist_deg=1.0,
                prediction_yaw_tiebreak_enabled=True,
                tiebreak_score_eps=0.001,
                tiebreak_rms_eps=0.001,
                tiebreak_yaw_margin_rad=math.radians(0.5),
            ),
        )
        mr = matcher.match(
            scan=scan,
            prediction_map=Pose2(0.0, 0.0, 0.0),
            ref_points_xy_map=ref,
        )
        assert "tiebreak" in mr.diagnostics
        td = mr.diagnostics["tiebreak"]
        assert td["enabled"] is True
        assert "applied" in td
        assert "from_index" in td
        assert "to_index" in td
