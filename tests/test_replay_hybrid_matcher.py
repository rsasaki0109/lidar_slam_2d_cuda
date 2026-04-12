from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import _engine_from_config, app


def test_replay_accepts_hybrid_matcher(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text(
        "\n".join(
            [
                "slam:",
                "  preprocess:",
                "    stride: 2",
                "  local_matching:",
                "    type: hybrid",
                "    correlative_grid:",
                "      linear_step_m: 0.05",
                "      angular_step_deg: 2.0",
                "      linear_window_m: 0.10",
                "      angular_window_deg: 10.0",
                "      sigma_hit_m: 0.10",
                "    icp:",
                "      max_iterations: 8",
                "      max_correspondence_dist_m: 0.75",
                "      min_correspondences: 20",
                "      trim_fraction: 0.10",
                "    hybrid_refinement:",
                "      top_k: 3",
                "      min_linear_dist_m: 0.05",
                "      min_angular_dist_deg: 2.0",
                "    hybrid_fallback:",
                "      enabled: true",
                "      trigger_score: -0.01",
                "      correlative_grid:",
                "        linear_step_m: 0.05",
                "        angular_step_deg: 2.0",
                "        linear_window_m: 0.20",
                "        angular_window_deg: 15.0",
                "        sigma_hit_m: 0.10",
                "  prediction:",
                "    mode: constant_velocity",
                "    gain: 1.0",
                "  submap:",
                "    max_submap_scans: 20",
                "    downsample_stride: 1",
                "  optimize_every_n_keyframes: 10",
                "  loop_detection:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(out),
            "--deterministic",
            "--seed",
            "0",
            "--no-write-map",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr

    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    assert len(traj) > 0


def test_engine_from_config_reads_gradient_mask_preprocess_options() -> None:
    eng = _engine_from_config(
        {
            "slam": {
                "preprocess": {
                    "gradient_mask_diff_m": 0.8,
                    "gradient_mask_max_range": 2.5,
                    "gradient_mask_window": 1,
                }
            }
        },
        telemetry=None,
    )

    assert eng.cfg.preprocess.gradient_mask_diff_m == 0.8
    assert eng.cfg.preprocess.gradient_mask_max_range == 2.5
    assert eng.cfg.preprocess.gradient_mask_window == 1


def test_replay_can_optimize_once_at_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = tmp_path / "final-opt.yaml"
    cfg.write_text(
        "\n".join(
            [
                "slam:",
                "  preprocess:",
                "    stride: 2",
                "  local_matching:",
                "    type: hybrid",
                "    correlative_grid:",
                "      linear_step_m: 0.05",
                "      angular_step_deg: 2.0",
                "      linear_window_m: 0.10",
                "      angular_window_deg: 10.0",
                "      sigma_hit_m: 0.10",
                "    icp:",
                "      max_iterations: 8",
                "      max_correspondence_dist_m: 0.75",
                "      min_correspondences: 20",
                "      trim_fraction: 0.10",
                "  optimize_every_n_keyframes: 0",
                "  final_optimize_at_end: true",
                "  loop_detection:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(out),
            "--deterministic",
            "--seed",
            "0",
            "--no-write-map",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr

    telem = [
        json.loads(line)
        for line in (out / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    optimizations = [ev for ev in telem if ev.get("type") == "optimization"]

    assert optimizations
    assert optimizations[-1]["node"] == len(traj) - 1
