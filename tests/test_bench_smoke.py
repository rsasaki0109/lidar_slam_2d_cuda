from __future__ import annotations

import json
from pathlib import Path

from slamx.cli import bench_lib


def test_agreement_metric_smoke(tmp_path: Path) -> None:
    # Create two identical csv trajectories and ensure agreement ~0
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    csv1.write_text("stamp_ns,x,y\n0,0,0\n1,1,0\n", encoding="utf-8")
    csv2.write_text("stamp_ns,x,y\n0,0,0\n1,1,0\n", encoding="utf-8")

    rep = bench_lib.compute_agreement(slamx_traj=csv1, carto_csv=csv2, max_dt_ns=0, align=True)
    assert rep["ok"] is True
    assert rep["rmse_m"] == 0.0


def test_bench_markdown_contains_steps() -> None:
    md = bench_lib.render_bench_markdown(
        {
            "bag_path": "/abs/path/to/bag.bag",
            "slamx_run": "/abs/path/to/runs/slamx_run",
            "scan_topic": "/scan",
            "agreement": None,
            "cartographer_traj_csv": None,
        }
    )
    assert "## 結論" in md
    assert "rosbag play" in md

