from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from slamx.telemetry.schema import validate_event


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_events_by_node(telem_path: Path, node_min: int, node_max: int) -> list[dict[str, Any]]:
    evs = load_jsonl(telem_path)
    out: list[dict[str, Any]] = []
    for e in evs:
        n = e.get("node")
        if n is None:
            continue
        try:
            ni = int(n)
        except Exception:
            continue
        if node_min <= ni <= node_max and e.get("type") in {
            "keyframe",
            "scan_match_candidates",
            "optimization",
            "loop_closure_candidates",
        }:
            out.append(e)
    return out


def diagnose_run(run_dir: Path) -> dict[str, Any]:
    telem = run_dir / "telemetry.jsonl"
    traj = run_dir / "trajectory.json"
    out: dict[str, Any] = {"run_dir": str(run_dir), "findings": [], "suggestions": []}

    if not telem.exists():
        out["findings"].append({"level": "error", "message": "missing telemetry.jsonl"})
    if not traj.exists():
        out["findings"].append({"level": "error", "message": "missing trajectory.json"})

    if telem.exists():
        evs = load_jsonl(telem)
        schema_errors = 0
        unknown_types = 0
        for e in evs:
            issues = validate_event(e)
            for it in issues:
                if it.level == "error":
                    schema_errors += 1
                if it.level == "warn" and "unknown event type" in it.message:
                    unknown_types += 1
        low_scores: list[int] = []
        big_jumps: list[int] = []
        ambiguous: list[int] = []
        for e in evs:
            if e.get("type") != "keyframe":
                continue
            node = int(e.get("node", -1))
            sc = float(e.get("scan_match_score", 0.0))
            if sc < -5.0:  # heuristic threshold for this prototype scorer
                low_scores.append(node)
            pj = float(e.get("pose_jump", 0.0))
            if pj > 1.0:
                big_jumps.append(node)
        for e in evs:
            if e.get("type") != "scan_match_candidates":
                continue
            node = int(e.get("node", -1))
            top = e.get("top") or []
            if isinstance(top, list) and len(top) >= 2:
                try:
                    s0 = float(top[0][3])
                    s1 = float(top[1][3])
                    if abs(s0 - s1) < 0.02:  # flat / ambiguous grid for this scorer
                        ambiguous.append(node)
                except Exception:
                    pass
        if low_scores:
            out["findings"].append(
                {
                    "level": "warn",
                    "message": "keyframes with very low scan-match scores",
                    "nodes": low_scores[:20],
                }
            )
        if big_jumps:
            out["findings"].append(
                {
                    "level": "warn",
                    "message": "keyframes with large pose jump vs prediction",
                    "nodes": big_jumps[:20],
                }
            )
        if ambiguous:
            out["findings"].append(
                {
                    "level": "warn",
                    "message": "scan matching appears ambiguous (top-2 scores too close)",
                    "nodes": ambiguous[:20],
                }
            )

        n_opt = sum(1 for e in evs if e.get("type") == "optimization")
        out["stats"] = {
            "events": len(evs),
            "optimization_events": n_opt,
            "telemetry_schema_errors": schema_errors,
            "telemetry_unknown_types": unknown_types,
        }

        # Suggestions (actionable knobs). Uses resolved config if present.
        cfg_path = run_dir / "config_resolved.yaml"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}

        slam = (cfg or {}).get("slam", {}) if isinstance(cfg, dict) else {}
        preprocess = slam.get("preprocess", {}) if isinstance(slam, dict) else {}
        lm = slam.get("local_matching", {}) if isinstance(slam, dict) else {}
        matcher_type = str(lm.get("type", "correlative"))

        if low_scores:
            out["suggestions"].append(
                {
                    "why": "scan_match_score is very low on some keyframes",
                    "try": [
                        "increase slam.local_matching.correlative_grid.linear_window_m / angular_window_deg",
                        "reduce preprocess.stride (denser points) or lower preprocess.max_range",
                        "if using icp: increase slam.local_matching.icp.max_correspondence_dist_m",
                    ],
                    "context": {"matcher_type": matcher_type, "preprocess": preprocess},
                }
            )
        if ambiguous:
            out["suggestions"].append(
                {
                    "why": "scan_match_candidates top-2 scores too close (flat search)",
                    "try": [
                        "decrease correlative_grid.linear_step_m / angular_step_deg (finer grid)",
                        "increase sigma_hit_m slightly to smooth scoring",
                        "switch matcher: slam.local_matching.type = icp (local refinement)",
                    ],
                    "context": {"matcher_type": matcher_type},
                }
            )
        if big_jumps:
            out["suggestions"].append(
                {
                    "why": "pose_jump large vs prediction (instability)",
                    "try": [
                        "increase optimize_every_n_keyframes frequency (smaller N)",
                        "reduce preprocess.stride, ensure sufficient overlap",
                        "tighten max_range to avoid far noisy returns",
                    ],
                    "context": {"optimize_every_n_keyframes": slam.get("optimize_every_n_keyframes")},
                }
            )

    return out


def trajectory_max_se2_delta(a_path: Path, b_path: Path) -> dict[str, Any]:
    def load_traj(p: Path) -> list[dict[str, float]]:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [
            {
                "i": int(row["i"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "theta": float(row["theta"]),
            }
            for row in data
        ]

    a = load_traj(a_path)
    b = load_traj(b_path)
    n = min(len(a), len(b))
    if n == 0:
        return {"ok": False, "message": "empty trajectory"}
    max_trans = 0.0
    max_rot = 0.0
    worst_i = 0
    for i in range(n):
        dx = a[i]["x"] - b[i]["x"]
        dy = a[i]["y"] - b[i]["y"]
        dt = math.atan2(
            math.sin(a[i]["theta"] - b[i]["theta"]),
            math.cos(a[i]["theta"] - b[i]["theta"]),
        )
        t = math.hypot(dx, dy)
        if t > max_trans or abs(dt) > max_rot:
            worst_i = i
        max_trans = max(max_trans, t)
        max_rot = max(max_rot, abs(dt))
    return {
        "ok": True,
        "compared_poses": n,
        "max_translation_m": max_trans,
        "max_rotation_rad": max_rot,
        "worst_index": worst_i,
        "len_a": len(a),
        "len_b": len(b),
    }


def telemetry_keyframe_series(telem_path: Path) -> dict[str, list[float]]:
    evs = load_jsonl(telem_path)
    score: list[float] = []
    jump: list[float] = []
    node: list[int] = []
    for e in evs:
        if e.get("type") != "keyframe":
            continue
        node.append(int(e.get("node", -1)))
        score.append(float(e.get("scan_match_score", 0.0)))
        jump.append(float(e.get("pose_jump", 0.0)))
    return {"node": node, "scan_match_score": score, "pose_jump": jump}


def series_summary(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    xs2 = sorted(xs)
    n = len(xs2)
    p50 = xs2[n // 2]
    p90 = xs2[int(0.9 * (n - 1))]
    return {"n": n, "min": xs2[0], "p50": p50, "p90": p90, "max": xs2[-1]}


def series_diff(a: list[float], b: list[float]) -> dict[str, Any]:
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0}
    d = [float(a[i] - b[i]) for i in range(n)]
    # worst by absolute delta
    worst_i = int(max(range(n), key=lambda i: abs(d[i])))
    return {
        "n": n,
        "max_abs_delta": float(max(abs(x) for x in d)),
        "worst_index": worst_i,
        "worst_delta": float(d[worst_i]),
    }
