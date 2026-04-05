from __future__ import annotations

import csv
import json
from pathlib import Path


def export_trajectory_csv(run_dir: Path, out_csv: Path) -> Path:
    """Export slamx run trajectory.json to CSV (stamp_ns,x,y)."""
    traj = run_dir / "trajectory.json"
    if not traj.exists():
        raise FileNotFoundError(traj)
    rows = json.loads(traj.read_text(encoding="utf-8"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp_ns", "x", "y"])
        for r in rows:
            s = r.get("stamp_ns")
            if s is None:
                continue
            w.writerow([int(s), float(r["x"]), float(r["y"])])
    return out_csv

