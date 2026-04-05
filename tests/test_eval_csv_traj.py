from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_eval_ate_accepts_csv_traj(tmp_path: Path) -> None:
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

    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text(
        "stamp_ns,x,y\n" + "\n".join(f"{row['stamp_ns']},{row['x']},{row['y']}" for row in traj) + "\n",
        encoding="utf-8",
    )
    est_csv = tmp_path / "est.csv"
    est_csv.write_text(gt_csv.read_text(encoding="utf-8"), encoding="utf-8")

    e = runner.invoke(app, ["eval", "ate", str(est_csv), "--gt", str(gt_csv), "--max-dt-ms", "1"])
    assert e.exit_code == 0, e.stdout + e.stderr
    rep = json.loads(e.stdout)
    assert rep["ok"] is True
    assert rep["rmse_m"] <= 1e-9

