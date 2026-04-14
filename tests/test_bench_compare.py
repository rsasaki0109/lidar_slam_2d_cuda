from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_export_slamx_trajectory_csv(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"
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
    assert r.exit_code == 0

    csv_path = tmp_path / "traj.csv"
    e = runner.invoke(app, ["export-slamx-trajectory", str(out), "--out", str(csv_path)])
    assert e.exit_code == 0, e.stdout + e.stderr
    assert csv_path.exists()


def test_sample_trajectory_to_timestamps_exports_nearest_matches(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "stamp_ns,x,y\n10000000,1.0,0.0\n20000000,2.0,0.0\n40000000,4.0,0.0\n",
        encoding="utf-8",
    )
    target = tmp_path / "target.csv"
    target.write_text(
        "stamp_ns,x,y\n11000000,0.0,0.0\n18000000,0.0,0.0\n55000000,0.0,0.0\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "sampled.csv"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "sample-trajectory-to-timestamps",
            str(source),
            str(target),
            "--out",
            str(out_csv),
            "--max-dt-ms",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["rows"] == 2

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"stamp_ns": "11000000", "x": "1.0", "y": "0.0"},
        {"stamp_ns": "18000000", "x": "2.0", "y": "0.0"},
    ]


def test_export_telemetry_keyframes_exports_keyframe_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    telemetry = run_dir / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                '{"type":"scan_match_candidates","node":0}',
                '{"type":"keyframe","stamp_ns":10,"pose":{"x":1.5,"y":2.5,"theta":0.1}}',
                '{"type":"keyframe","stamp_ns":20,"pose":{"x":3.5,"y":4.5,"theta":0.2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "telemetry.csv"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-telemetry-keyframes",
            str(run_dir),
            "--out",
            str(out_csv),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["rows"] == 2

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"stamp_ns": "10", "x": "1.5", "y": "2.5"},
        {"stamp_ns": "20", "x": "3.5", "y": "4.5"},
    ]
