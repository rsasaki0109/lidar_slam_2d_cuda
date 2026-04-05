from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_replay_fixture_jsonl(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"
    out = tmp_path / "r1"

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
            "1",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    traj_path = out / "trajectory.json"
    assert traj_path.exists()
    poses = json.loads(traj_path.read_text(encoding="utf-8"))
    assert len(poses) == 3

    out2 = tmp_path / "r2"
    r2 = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(out2),
            "--deterministic",
            "--seed",
            "1",
        ],
    )
    assert r2.exit_code == 0
    assert traj_path.read_bytes() == (out2 / "trajectory.json").read_bytes()
