from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_sweep_can_toggle_loop_enabled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    base_cfg = root / "configs" / "default.yaml"
    axes = root / "examples" / "axes_loop.yaml"

    # Create GT from baseline trajectory
    out_gt = tmp_path / "gt_run"
    runner = CliRunner()
    rr = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(base_cfg),
            "--out",
            str(out_gt),
            "--deterministic",
            "--seed",
            "0",
            "--no-write-map",
        ],
    )
    assert rr.exit_code == 0
    traj = json.loads((out_gt / "trajectory.json").read_text(encoding="utf-8"))
    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text(
        "stamp_ns,x,y\n" + "\n".join(f"{row['stamp_ns']},{row['x']},{row['y']}" for row in traj) + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "sweep"
    r = runner.invoke(
        app,
        [
            "sweep",
            str(axes),
            str(fixture),
            "--base-config",
            str(base_cfg),
            "--out-root",
            str(out_root),
            "--gt",
            str(gt_csv),
            "--max-dt-ms",
            "1",
            "--deterministic",
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert len(rep["results"]) == 4

