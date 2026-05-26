from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.cloud_analyzer import CloudAnalyzer
from slamx.cli.main import app


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    looped = tmp_path / "looped"
    baseline.mkdir()
    looped.mkdir()

    _write_json(
        baseline / "trajectory.json",
        [
            {"i": 0, "x": 0.0, "y": 0.0, "theta": 0.0},
            {"i": 1, "x": 1.0, "y": 0.0, "theta": 0.0},
            {"i": 2, "x": 1.2, "y": 0.0, "theta": 0.0},
        ],
    )
    _write_json(
        looped / "trajectory.json",
        [
            {"i": 0, "x": 0.0, "y": 0.0, "theta": 0.0},
            {"i": 1, "x": 1.0, "y": 0.0, "theta": 0.0},
            {"i": 2, "x": 0.02, "y": 0.0, "theta": 0.0},
        ],
    )
    _write_jsonl(
        looped / "telemetry.jsonl",
        [
            {
                "type": "keyframe",
                "schema_version": 1,
                "node": 0,
                "stamp_ns": 0,
                "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "prediction": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "scan_match_score": 0.0,
                "pose_jump": 0.0,
            },
            {
                "type": "keyframe",
                "schema_version": 1,
                "node": 1,
                "stamp_ns": 1,
                "pose": {"x": 1.0, "y": 0.0, "theta": 0.0},
                "prediction": {"x": 1.0, "y": 0.0, "theta": 0.0},
                "scan_match_score": 0.0,
                "pose_jump": 0.0,
            },
            {
                "type": "loop_closure_candidates",
                "schema_version": 1,
                "node": 2,
                "candidates": [{"i": 0, "j": 2, "score": -0.01, "accepted": True}],
            },
            {
                "type": "loop_closure_accepted",
                "schema_version": 1,
                "node": 2,
                "i": 0,
                "j": 2,
                "score": -0.01,
                "rel_ij": {"x": 0.02, "y": 0.0, "theta": 0.0},
            },
            {
                "type": "optimization",
                "schema_version": 1,
                "node": 2,
                "final": True,
                "success": True,
                "cost": 0.0,
                "n_poses": 3,
                "n_edges": 3,
                "nfev": 5,
                "max_nfev": 32,
                "residual_rms_before": 0.2,
                "residual_rms_after": 0.01,
            },
        ],
    )
    return baseline, looped


def test_cloud_analyzer_detects_loop_and_odometry_drift(tmp_path: Path) -> None:
    baseline, looped = _make_run(tmp_path)

    rep = CloudAnalyzer(looped, baseline_run=baseline).analyze()

    assert rep["ok"] is True
    assert rep["inference"]["loop_closure"]["status"] == "closed"
    assert rep["inference"]["loop_closure"]["can_close"] == "yes"
    assert rep["inference"]["lidar_odometry"]["status"] == "failing_drift_corrected_by_loop"
    assert rep["facts"]["telemetry"]["loop_candidates"]["accepted"] == 1
    assert rep["facts"]["baseline_vs_loop"]["baseline_start_end_gap_m"] == 1.2
    assert rep["facts"]["baseline_vs_loop"]["loop_start_end_gap_m"] == 0.02


def test_cloud_analyze_cli_outputs_json(tmp_path: Path) -> None:
    baseline, looped = _make_run(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["cloud-analyze", str(looped), "--baseline-run", str(baseline)],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    rep = json.loads(result.stdout)
    assert rep["inference"]["loop_closure"]["status"] == "closed"
    assert rep["inference"]["lidar_odometry"]["status"] == "failing_drift_corrected_by_loop"
