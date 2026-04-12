from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml

from slamx.cli import doctor_lib
from slamx.core.evaluation.ate import (
    build_ate_report,
    load_gt,
    load_slam_trajectory,
)


def build_report(
    *,
    run_dir: Path,
    gt_path: Path | None,
    max_dt_ns: int,
    align: bool,
    segment_gap_ns: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"run_dir": str(run_dir)}

    cfg_path = run_dir / "config_resolved.yaml"
    cfg = None
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = None
    out["config_resolved_path"] = str(cfg_path) if cfg_path.exists() else None
    out["config"] = cfg if isinstance(cfg, dict) else None

    # facts from artifacts
    traj_path = run_dir / "trajectory.json"
    telem_path = run_dir / "telemetry.jsonl"
    out["artifacts"] = {
        "trajectory": str(traj_path) if traj_path.exists() else None,
        "telemetry": str(telem_path) if telem_path.exists() else None,
        "map_pgm": str(run_dir / "map.pgm") if (run_dir / "map.pgm").exists() else None,
        "map_yaml": str(run_dir / "map.yaml") if (run_dir / "map.yaml").exists() else None,
    }

    out["doctor"] = doctor_lib.diagnose_run(run_dir)

    if gt_path is not None and traj_path.exists():
        slam = load_slam_trajectory(traj_path)
        gt = load_gt(gt_path)
        out["ate"] = build_ate_report(
            slam,
            gt,
            max_dt_ns=max_dt_ns,
            align=align,
            segment_gap_ns=segment_gap_ns,
        )
        out["ate"]["gt_path"] = str(gt_path)
    else:
        out["ate"] = None

    out["generated_at"] = _dt.datetime.now().isoformat()
    return out


def render_markdown(rep: dict[str, Any]) -> str:
    run_dir = rep.get("run_dir", "")
    doctor = rep.get("doctor") or {}
    findings = doctor.get("findings") or []
    suggestions = doctor.get("suggestions") or []
    ate = rep.get("ate")

    # 結論: fact-based
    conc_lines: list[str] = []
    if ate and ate.get("ok"):
        conc_lines.append(f"- ATE RMSE: **{ate.get('rmse_m'):.3f} m** (n={ate.get('n')})")
        assoc = ate.get("association") or {}
        matched_pairs = assoc.get("matched_pairs")
        traj_points = assoc.get("traj_points")
        span_ratio = assoc.get("matched_traj_span_ratio")
        if matched_pairs is not None and traj_points:
            conc_lines.append(f"- ATE coverage: **{matched_pairs}/{traj_points}** trajectory points")
        if isinstance(span_ratio, (float, int)) and span_ratio < 0.999:
            conc_lines.append(f"- ATE time coverage: **{span_ratio * 100.0:.1f}%** of trajectory span")
    else:
        conc_lines.append("- ATE: **未計測**（GT未指定 or マッチ失敗）")
    if findings:
        conc_lines.append(f"- Doctor: **{len(findings)}** 件の所見（warn/error）")
    else:
        conc_lines.append("- Doctor: 所見なし")

    # 確認済み事実
    facts: list[str] = []
    facts.append(f"- run_dir: `{run_dir}`")
    art = rep.get("artifacts") or {}
    for k in ("trajectory", "telemetry", "map_pgm", "map_yaml"):
        if art.get(k):
            facts.append(f"- {k}: `{art.get(k)}`")

    if ate:
        facts.append(f"- ate: `{json.dumps(ate, ensure_ascii=False)}`")
        for warning in ate.get("warnings") or []:
            facts.append(f"- ate_warning: `{warning}`")

    # 未確認/要確認項目
    unknown: list[str] = []
    if not ate or not (ate.get("ok") if isinstance(ate, dict) else False):
        unknown.append("- GTが正しい座標系/時刻同期か（ATE算出条件）")
    elif any("coverage" in w.lower() or "segments" in w.lower() for w in (ate.get("warnings") or [])):
        unknown.append("- GT の欠測区間や区間分断で、評価が軌跡全体を代表しているか")
    if any(f.get("level") == "error" for f in findings):
        unknown.append("- run成果物の欠落が意図通りか（telemetry/trajectory）")

    # 次アクション
    actions: list[str] = []
    if suggestions:
        for s in suggestions[:5]:
            why = s.get("why", "why?")
            tr = s.get("try", [])
            actions.append(f"- **{why}**")
            if isinstance(tr, list):
                for t in tr[:5]:
                    actions.append(f"  - {t}")
    else:
        actions.append("- `slamx diff --focus worst runA runB` で崩れ点のイベントを比較")

    md = []
    md.append("## 結論")
    md.extend(conc_lines)
    md.append("")
    md.append("## 確認済み事実")
    md.extend(facts)
    md.append("")
    md.append("## 未確認/要確認項目")
    md.extend(unknown if unknown else ["- （なし）"])
    md.append("")
    md.append("## 次アクション")
    md.extend(actions)
    md.append("")
    return "\n".join(md)


def write_notes_markdown(notes_path: Path, content: str) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(content, encoding="utf-8")
