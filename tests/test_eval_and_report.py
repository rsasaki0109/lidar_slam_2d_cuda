from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_eval_ate_and_report(tmp_path: Path) -> None:
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
    assert r.exit_code == 0, r.stdout + r.stderr

    # Create a trivial GT identical to slam trajectory (so RMSE should be ~0 after alignment)
    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    gt = tmp_path / "gt.csv"
    gt.write_text("stamp_ns,x,y\n" + "\n".join(f"{row['stamp_ns']},{row['x']},{row['y']}" for row in traj) + "\n", encoding="utf-8")

    e = runner.invoke(app, ["eval", "ate", str(out), "--gt", str(gt), "--max-dt-ms", "1"])
    assert e.exit_code == 0, e.stdout + e.stderr
    rep = json.loads(e.stdout)
    assert rep["ok"] is True
    assert rep["rmse_m"] <= 1e-9

    notes = tmp_path / "notes.md"
    rr = runner.invoke(app, ["report", str(out), "--gt", str(gt), "--out", str(notes)])
    assert rr.exit_code == 0, rr.stdout + rr.stderr
    assert notes.exists()
    md = notes.read_text(encoding="utf-8")
    assert "## 結論" in md
    assert "## 確認済み事実" in md
    assert "## 未確認/要確認項目" in md
    assert "## 次アクション" in md

