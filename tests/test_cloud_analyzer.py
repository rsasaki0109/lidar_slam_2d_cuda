from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.cloud_analyzer import CloudAnalyzer, CloudHotspotAnalyzer
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
    _write_jsonl(
        baseline / "telemetry.jsonl",
        [
            {
                "type": "keyframe",
                "schema_version": 1,
                "node": i,
                "stamp_ns": i,
                "pose": {"x": float(i) * 0.6, "y": 0.0, "theta": 0.0},
                "prediction": {"x": float(i) * 0.6, "y": 0.0, "theta": 0.0},
                "scan_match_score": 0.0,
                "pose_jump": 0.0,
            }
            for i in range(3)
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
    assert (
        rep["inference"]["lidar_odometry"]["failure_mode"]
        == "accumulated_drift_without_local_jump"
    )
    assert rep["inference"]["lidar_odometry"]["telemetry_source"] == "baseline_run"
    assert rep["facts"]["telemetry"]["loop_candidates"]["accepted"] == 1
    assert rep["facts"]["baseline_telemetry"]["prediction_error"]["translation_m"]["max"] == 0.0
    assert rep["facts"]["baseline_vs_loop"]["baseline_start_end_gap_m"] == 1.2
    assert rep["facts"]["baseline_vs_loop"]["loop_start_end_gap_m"] == 0.02
    assert rep["facts"]["baseline_vs_loop"]["max_translation_node"] == 2
    assert rep["facts"]["baseline_vs_loop"]["drift_growth"]["threshold_crossings"] == {
        "0.25m": 2,
        "0.50m": 2,
        "1.00m": 2,
    }
    end_decomp = rep["facts"]["baseline_vs_loop"]["drift_decomposition"]["end"]
    assert end_decomp["dominant_translation_component"] == "longitudinal"
    assert end_decomp["longitudinal_m"] == -1.18
    assert end_decomp["lateral_m"] == 0.0
    assert end_decomp["yaw_rad"] == 0.0
    hotspots = rep["facts"]["baseline_vs_loop"]["hotspots"]
    assert hotspots[0]["start_node"] == 0
    assert hotspots[0]["end_node"] == 2
    assert hotspots[0]["growth_m"] == 1.18
    assert hotspots[0]["loop_events"]["accepted"] == 1
    assert hotspots[0]["failure_mode"] == "drift_growth_without_local_jump"


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
    assert rep["inference"]["lidar_odometry"]["telemetry_source"] == "baseline_run"
    assert "drift_growth" in rep["facts"]["baseline_vs_loop"]
    assert rep["facts"]["baseline_vs_loop"]["hotspots"][0]["growth_m"] == 1.18


def test_cloud_analyze_cli_outputs_hotspot_markdown(tmp_path: Path) -> None:
    baseline, looped = _make_run(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["cloud-analyze", str(looped), "--baseline-run", str(baseline), "--markdown"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "## Drift Hotspots" in result.stdout
    assert "| 1 | 0->2 | 1.180 | -1.180 | 0.000 | 0.00 |" in result.stdout


def test_cloud_hotspot_reports_node_table(tmp_path: Path) -> None:
    baseline, looped = _make_run(tmp_path)

    rep = CloudHotspotAnalyzer(
        looped,
        baseline_run=baseline,
        start_node=1,
        end_node=2,
    ).analyze()

    assert rep["ok"] is True
    assert rep["summary"]["drift_growth_m"] == 1.18
    assert rep["summary"]["delta"]["dominant_translation_component"] == "longitudinal"
    assert rep["summary"]["loop_events"]["accepted"] == 1
    assert rep["summary"]["telemetry_source"] == "baseline_run"
    assert [r["node"] for r in rep["table"]] == [1, 2]
    assert rep["table"][1]["correction_m"] == 1.18
    assert rep["table"][1]["longitudinal_m"] == -1.18
    assert rep["table"][1]["loop_event_label"] == "cand:1 acc:1"


def test_cloud_hotspot_cli_outputs_markdown(tmp_path: Path) -> None:
    baseline, looped = _make_run(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "cloud-hotspot",
            str(looped),
            "--baseline-run",
            str(baseline),
            "--start-node",
            "1",
            "--end-node",
            "2",
            "--markdown",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "# Cloud Hotspot" in result.stdout
    assert "Drift growth: 1.180 m" in result.stdout
    assert "| 2 | 1.180 | -1.180 | 0.000 | 0.00 |" in result.stdout
