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
