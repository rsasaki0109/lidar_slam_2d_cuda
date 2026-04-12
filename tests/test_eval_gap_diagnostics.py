from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_eval_ate_reports_gt_gap_and_partial_coverage(tmp_path: Path) -> None:
    est_csv = tmp_path / "est.csv"
    gt_csv = tmp_path / "gt.csv"

    est_csv.write_text(
        "\n".join(
            [
                "stamp_ns,x,y",
                "0,0.0,0.0",
                "1000000000,1.0,0.0",
                "2000000000,2.0,0.0",
                "3000000000,3.0,0.0",
                "4000000000,4.0,0.0",
                "5000000000,5.0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gt_csv.write_text(
        "\n".join(
            [
                "stamp_ns,x,y",
                "0,0.0,0.0",
                "1000000000,1.0,0.0",
                "2000000000,2.0,0.0",
                "100000000000,100.0,0.0",
                "101000000000,101.0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    r = runner.invoke(app, ["eval", "ate", str(est_csv), "--gt", str(gt_csv), "--max-dt-ms", "1"])
    assert r.exit_code == 0, r.stdout + r.stderr

    rep = json.loads(r.stdout)
    assert rep["ok"] is True
    assert rep["n"] == 3
    assert rep["gt_time_axis"]["segment_count"] == 2
    assert rep["gt_time_axis"]["largest_segment_gap_s"] > 90.0
    assert rep["association"]["matched_pairs"] == 3
    assert abs(rep["association"]["matched_traj_point_ratio"] - 0.5) < 1e-9
    assert abs(rep["association"]["matched_traj_span_ratio"] - 0.4) < 1e-9
    warnings = rep.get("warnings") or []
    assert any("multiple time segments" in w for w in warnings)
    assert any("trajectory time span" in w for w in warnings)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trajectory.json").write_text(
        json.dumps(
            [
                {"stamp_ns": 0, "x": 0.0, "y": 0.0},
                {"stamp_ns": 1000000000, "x": 1.0, "y": 0.0},
                {"stamp_ns": 2000000000, "x": 2.0, "y": 0.0},
                {"stamp_ns": 3000000000, "x": 3.0, "y": 0.0},
                {"stamp_ns": 4000000000, "x": 4.0, "y": 0.0},
                {"stamp_ns": 5000000000, "x": 5.0, "y": 0.0},
            ]
        ),
        encoding="utf-8",
    )
    notes = tmp_path / "report.md"
    rr = runner.invoke(app, ["report", str(run_dir), "--gt", str(gt_csv), "--out", str(notes)])
    assert rr.exit_code == 0, rr.stdout + rr.stderr
    md = notes.read_text(encoding="utf-8")
    assert "ATE coverage" in md
    assert "ATE time coverage" in md
