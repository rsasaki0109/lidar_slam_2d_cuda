from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_replay_writes_map_files(tmp_path: Path) -> None:
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
            "--write-map",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert (out / "map.pgm").exists()
    assert (out / "map.yaml").exists()

