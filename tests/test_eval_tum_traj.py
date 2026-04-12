from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def _write_tum(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        ts_sec = float(row["stamp_ns"]) / 1_000_000_000.0
        lines.append(f"{ts_sec:.9f} {float(row['x'])} {float(row['y'])} 0.0 0.0 0.0 0.0 1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_eval_ate_accepts_tum_gt_and_traj(tmp_path: Path) -> None:
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

    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    gt_tum = tmp_path / "gt.tum"
    est_tum = tmp_path / "est.tum"
    _write_tum(gt_tum, traj)
    _write_tum(est_tum, traj)

    e_run = runner.invoke(app, ["eval", "ate", str(out), "--gt", str(gt_tum), "--max-dt-ms", "1"])
    assert e_run.exit_code == 0, e_run.stdout + e_run.stderr
    rep_run = json.loads(e_run.stdout)
    assert rep_run["ok"] is True
    assert rep_run["rmse_m"] <= 1e-9

    e_tum = runner.invoke(app, ["eval", "ate", str(est_tum), "--gt", str(gt_tum), "--max-dt-ms", "1"])
    assert e_tum.exit_code == 0, e_tum.stdout + e_tum.stderr
    rep_tum = json.loads(e_tum.stdout)
    assert rep_tum["ok"] is True
    assert rep_tum["rmse_m"] <= 1e-9
