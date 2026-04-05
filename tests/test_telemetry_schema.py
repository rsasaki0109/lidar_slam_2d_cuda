from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app
from slamx.cli.doctor_lib import load_jsonl
from slamx.telemetry.schema import validate_event


def test_telemetry_events_validate(tmp_path: Path) -> None:
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

    evs = load_jsonl(out / "telemetry.jsonl")
    assert evs, "telemetry must not be empty"
    for e in evs:
        issues = validate_event(e)
        assert not [i for i in issues if i.level == "error"], (e, issues)

