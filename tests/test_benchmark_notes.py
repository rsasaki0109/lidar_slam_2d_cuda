from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_iilabs_benchmark_note_tracks_timestamp_aligned_public_claims() -> None:
    rep = json.loads(
        (ROOT / "notes" / "benchmark_iilabs_vs_cartographer_sampled_vlp16.json").read_text(
            encoding="utf-8"
        )
    )

    assert rep["evaluation_protocol"]["max_dt_ms"] == 50

    comparisons = rep["comparisons"]
    for name in ("slippage_full", "nav_a_omni_2k", "nav_a_diff_2k", "ramp_2k_vscan_bb"):
        comp = comparisons[name]
        assert comp["gt_coverage"] == "continuous"
        assert comp["align_se2_rmse_m"]["slamx"] < comp["align_se2_rmse_m"]["sampled_cartographer"]
        assert comp["matched_gt_segments"] == {"matched": 1, "total": 1}

    loop = comparisons["loop_2k_prefix"]
    assert loop["gt_coverage"] == "segmented_prefix_only"
    assert loop["align_se2_rmse_m"]["slamx"] < loop["align_se2_rmse_m"]["sampled_cartographer"]
    assert loop["matched_gt_segments"] == {"matched": 1, "total": 6}

    elevator = comparisons["elevator_2k_prefix"]
    assert elevator["gt_coverage"] == "segmented_prefix_only"
    assert elevator["align_se2_rmse_m"]["slamx"] < elevator["align_se2_rmse_m"]["sampled_cartographer"]
    assert elevator["matched_gt_segments"] == {"matched": 1, "total": 2}
    assert any("Only 954/2000" in warning for warning in elevator["warnings"])
