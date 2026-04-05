from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_sweep_minimal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    base_cfg = root / "configs" / "default.yaml"

    axes_yaml = tmp_path / "axes.yaml"
    axes_yaml.write_text(
        "\n".join(
            [
                "axes:",
                "  - key: slam.local_matching.correlative_grid.linear_window_m",
                "    values: [0.4, 0.6]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "sweep"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "sweep",
            str(axes_yaml),
            str(fixture),
            "--base-config",
            str(base_cfg),
            "--out-root",
            str(out_root),
            "--deterministic",
            "--seed",
            "3",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr

    rep = json.loads(r.stdout)
    assert Path(rep["baseline"]).exists()
    assert (out_root / "results.json").exists()
    assert len(rep["results"]) == 2


def test_sweep_with_gt_ate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    base_cfg = root / "configs" / "default.yaml"

    axes_yaml = tmp_path / "axes.yaml"
    axes_yaml.write_text(
        "\n".join(
            [
                "axes:",
                "  - key: slam.local_matching.correlative_grid.linear_window_m",
                "    values: [0.4, 0.6]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Generate baseline run and use it as GT (so RMSE ~ 0)
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

    gt_csv = tmp_path / "gt.csv"
    traj = json.loads((out_gt / "trajectory.json").read_text(encoding="utf-8"))
    gt_csv.write_text(
        "stamp_ns,x,y\n" + "\n".join(f"{row['stamp_ns']},{row['x']},{row['y']}" for row in traj) + "\n",
        encoding="utf-8",
    )

    out_root = tmp_path / "sweep_gt"
    r = runner.invoke(
        app,
        [
            "sweep",
            str(axes_yaml),
            str(fixture),
            "--base-config",
            str(base_cfg),
            "--out-root",
            str(out_root),
            "--gt",
            str(gt_csv),
            "--max-dt-ms",
            "1",
            "--write-reports",
            "--deterministic",
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["baseline_ate"]["ok"] is True
    assert len(rep["results"]) == 2
    assert "ate" in rep["results"][0]
    assert (out_root / "notes" / "baseline.md").exists()
    # At least one per-run report exists
    assert any((out_root / "notes").glob("run_*.md"))

