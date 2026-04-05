from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slamx.cli import datasets, report_lib
from slamx.core.evaluation.ate import associate_by_time, compute_ate_rmse, load_estimated_trajectory, load_gt


@dataclass(frozen=True)
class BenchResult:
    bag_path: str
    slamx_run: str
    cartographer_traj_csv: str | None
    agreement: dict[str, Any] | None
    notes_path: str | None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_bench_markdown(rep: dict[str, Any]) -> str:
    md = []
    md.append("## 結論")
    agree = rep.get("agreement")
    if isinstance(agree, dict) and agree.get("ok"):
        md.append(f"- slamx vs cartographer agreement RMSE: **{agree.get('rmse_m'):.3f} m** (n={agree.get('n')})")
    else:
        md.append("- Cartographer比較: **未実施**（cartographer軌跡が未提供）")
    md.append("")
    md.append("## 確認済み事実")
    md.append(f"- bag: `{rep.get('bag_path')}`")
    md.append(f"- slamx_run: `{rep.get('slamx_run')}`")
    if rep.get("scan_topic"):
        md.append(f"- scan_topic: `{rep.get('scan_topic')}`")
    if rep.get("cartographer_traj_csv"):
        md.append(f"- cartographer_traj_csv: `{rep.get('cartographer_traj_csv')}`")
    md.append("")
    md.append("## 未確認/要確認項目")
    md.append("- TFフレーム（map/base_link等）が正しいか")
    md.append("- Cartographer側のrecordに /tf が含まれているか")
    md.append("")
    md.append("## 次アクション")
    md.append("### Cartographer実行（テンプレ）")
    md.append("- bagを再生し、Cartographerを起動しつつ /tf をrecordする。")
    md.append("- 例（ROS1想定、環境により調整）:")
    md.append("")
    md.append("```bash")
    md.append(f"rosbag play \"{rep.get('bag_path')}\"")
    md.append("# 別端末で:")
    md.append("# cartographer_ros の 2D 設定で起動（例: backpack_2d）")
    md.append("# roslaunch cartographer_ros demo_backpack_2d.launch")
    md.append("# さらに別端末で /tf を含めて記録:")
    md.append("rosbag record -O carto_recorded.bag /tf /tf_static")
    md.append("```")
    md.append("")
    md.append("### TFから軌跡CSV化 → 一致度計算")
    md.append("- `slamx export-tf-trajectory <carto_recorded_bag> --parent map --child base_link --out carto.csv`")
    md.append("- `slamx eval ate <slamx_run_or_csv> --gt carto.csv` で一致度を再計算（同一max_dt_ms）")
    md.append("")
    return "\n".join(md)


def compute_agreement(
    *,
    slamx_traj: Path,
    carto_csv: Path,
    max_dt_ns: int,
    align: bool,
) -> dict[str, Any]:
    # Treat cartographer trajectory as "gt" for agreement
    slam = load_estimated_trajectory(slamx_traj)
    gt = load_gt(carto_csv)
    slam_xy, gt_xy = associate_by_time(slam, gt, max_dt_ns=max_dt_ns)
    rep = compute_ate_rmse(slam_xy, gt_xy, align=align)
    rep["traj"] = str(slamx_traj)
    rep["carto_csv"] = str(carto_csv)
    rep["max_dt_ns"] = int(max_dt_ns)
    return rep


def ensure_cartographer_bag_downloaded(bag_name: str, out_dir: Path) -> Path:
    spec = datasets.cartographer_backpack2d_bag(bag_name)
    return datasets.download(spec, out_dir)

