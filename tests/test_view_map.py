from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_view_map_ascii(tmp_path: Path) -> None:
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
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr

    v = runner.invoke(app, ["view-map", str(out), "--width", "40"])
    assert v.exit_code == 0, v.stdout + v.stderr
    assert "map.pgm" in v.stdout

