from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_diff_identical_runs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"

    runner = CliRunner()
    for name in ("a", "b"):
        r = runner.invoke(
            app,
            [
                "replay",
                str(fixture),
                "--config",
                str(cfg),
                "--out",
                str(tmp_path / name),
                "--deterministic",
                "--seed",
                "7",
            ],
        )
        assert r.exit_code == 0, r.stdout + r.stderr

    d = runner.invoke(app, ["diff", str(tmp_path / "a"), str(tmp_path / "b")])
    assert d.exit_code == 0, d.stdout + d.stderr
    rep = json.loads(d.stdout)
    assert rep["trajectory"]["max_translation_m"] == 0.0
    assert rep["trajectory"]["max_rotation_rad"] == 0.0


def test_doctor_ok_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(tmp_path / "r"),
            "--deterministic",
        ],
    )
    assert r.exit_code == 0
    d = runner.invoke(app, ["doctor", str(tmp_path / "r")])
    assert d.exit_code == 0, d.stdout + d.stderr
