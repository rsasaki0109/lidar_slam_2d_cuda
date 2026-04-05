from __future__ import annotations

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

