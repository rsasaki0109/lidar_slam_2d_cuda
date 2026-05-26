from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slamx.cli.doctor_lib import load_jsonl, series_summary


def _wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def _run_file(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def _load_trajectory(path: Path) -> list[dict[str, float]]:
    traj_path = _run_file(path, "trajectory.json")
    rows = json.loads(traj_path.read_text(encoding="utf-8"))
    out: list[dict[str, float]] = []
    for k, row in enumerate(rows):
        out.append(
            {
                "i": int(row.get("i", k)),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "theta": float(row.get("theta", 0.0)),
            }
        )
    return out


def _path_length(traj: list[dict[str, float]]) -> float:
    total = 0.0
    for a, b in zip(traj, traj[1:]):
        total += math.hypot(float(b["x"] - a["x"]), float(b["y"] - a["y"]))
    return float(total)


def _start_end_gap(traj: list[dict[str, float]]) -> float:
    if len(traj) < 2:
        return 0.0
    return float(math.hypot(traj[-1]["x"] - traj[0]["x"], traj[-1]["y"] - traj[0]["y"]))


def _trajectory_summary(traj: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "poses": len(traj),
        "path_length_m": _path_length(traj),
        "start_end_gap_m": _start_end_gap(traj),
    }


def _trajectory_drift_components(
    baseline: list[dict[str, float]],
    looped: list[dict[str, float]],
    n: int,
) -> dict[str, list[float] | list[int]]:
    nodes: list[int] = []
    trans: list[float] = []
    longitudinal: list[float] = []
    lateral: list[float] = []
    yaw: list[float] = []
    for i in range(n):
        node = int(looped[i].get("i", i))
        # Correction vector that moves the no-loop baseline pose to the loop-closed pose.
        dx = looped[i]["x"] - baseline[i]["x"]
        dy = looped[i]["y"] - baseline[i]["y"]
        th = looped[i]["theta"]
        c = math.cos(th)
        s = math.sin(th)
        nodes.append(node)
        trans.append(float(math.hypot(dx, dy)))
        longitudinal.append(float(c * dx + s * dy))
        lateral.append(float(-s * dx + c * dy))
        yaw.append(_wrap_pi(looped[i]["theta"] - baseline[i]["theta"]))
    return {
        "nodes": nodes,
        "translation_m": trans,
        "longitudinal_m": longitudinal,
        "lateral_m": lateral,
        "yaw_rad": yaw,
    }


def _sample_drift_curve(
    components: dict[str, list[float] | list[int]], limit: int = 24
) -> list[dict[str, float | int]]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    longitudinal = components["longitudinal_m"]
    lateral = components["lateral_m"]
    yaw = components["yaw_rad"]
    n = len(nodes)
    if n == 0:
        return []
    if n <= limit:
        idxs = list(range(n))
    else:
        idxs = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [
        {
            "node": int(nodes[i]),
            "translation_m": float(trans[i]),
            "longitudinal_m": float(longitudinal[i]),
            "lateral_m": float(lateral[i]),
            "yaw_rad": float(yaw[i]),
        }
        for i in idxs
    ]


def _drift_threshold_crossings(
    components: dict[str, list[float] | list[int]],
    thresholds: tuple[float, ...] = (0.25, 0.5, 1.0),
) -> dict[str, int | None]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    out: dict[str, int | None] = {}
    for threshold in thresholds:
        hit = next((int(nodes[i]) for i, v in enumerate(trans) if float(v) >= threshold), None)
        out[f"{threshold:.2f}m"] = hit
    return out


def _max_window_growth(
    components: dict[str, list[float] | list[int]],
    window_nodes: int = 100,
) -> dict[str, float | int | None]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    n = len(nodes)
    if n < 2:
        return {"window_nodes": window_nodes, "growth_m": 0.0, "start_node": None, "end_node": None}
    step = min(window_nodes, n - 1)
    best_i = step
    best_growth = float(trans[step]) - float(trans[0])
    for i in range(step, n):
        growth = float(trans[i]) - float(trans[i - step])
        if growth > best_growth:
            best_growth = growth
            best_i = i
    return {
        "window_nodes": int(step),
        "growth_m": float(best_growth),
        "start_node": int(nodes[best_i - step]),
        "end_node": int(nodes[best_i]),
    }


def _event_counts_in_range(
    loop_events: dict[int, dict[str, Any]],
    start_node: int,
    end_node: int,
) -> dict[str, int]:
    selected = [
        ev for node, ev in loop_events.items() if start_node <= int(node) <= end_node
    ]
    return {
        "candidate_count": sum(int(ev.get("candidate_count", 0)) for ev in selected),
        "accepted": sum(int(ev.get("accepted", 0)) for ev in selected),
        "rejected": sum(int(ev.get("rejected", 0)) for ev in selected),
    }


def _metrics_in_range(
    metrics: dict[int, dict[str, Any]],
    start_node: int,
    end_node: int,
    key: str,
) -> dict[str, Any]:
    vals = [
        float(row[key])
        for node, row in metrics.items()
        if start_node <= int(node) <= end_node and row.get(key) is not None
    ]
    return series_summary(vals)


def _optimization_summary_in_range(
    evs: list[dict[str, Any]],
    start_node: int,
    end_node: int,
) -> dict[str, Any]:
    selected = [
        e
        for e in evs
        if e.get("type") == "optimization"
        and start_node <= int(e.get("node", -1)) <= end_node
    ]
    reductions: list[float] = []
    after: list[float] = []
    max_nfev_hits = 0
    success = 0
    nodes: list[int] = []
    for e in selected:
        nodes.append(int(e.get("node", -1)))
        success += 1 if bool(e.get("success", False)) else 0
        before = e.get("residual_rms_before")
        after_raw = e.get("residual_rms_after")
        if after_raw is not None:
            after.append(float(after_raw))
        if before is not None and after_raw is not None and float(after_raw) > 0.0:
            reductions.append(float(before) / float(after_raw))
        nfev = e.get("nfev")
        max_nfev = e.get("max_nfev")
        if nfev is not None and max_nfev is not None and int(nfev) >= int(max_nfev):
            max_nfev_hits += 1
    return {
        "events": len(selected),
        "success": success,
        "nodes": _node_sample(sorted(set(nodes)), limit=10),
        "residual_rms_after": series_summary(after),
        "residual_rms_reduction": series_summary(reductions),
        "max_nfev_hits": max_nfev_hits,
    }


def _monotonic_growth_fraction(xs: list[float]) -> float:
    if len(xs) < 2:
        return 1.0
    monotonic = sum(1 for a, b in zip(xs, xs[1:]) if float(b) >= float(a))
    return monotonic / float(len(xs) - 1)


def _hotspot_failure_mode(
    row: dict[str, Any],
    thresholds: CloudAnalyzerThresholds,
) -> str:
    pose_jump = row["pose_jump"]
    score = row["scan_match_score"]
    max_jump = float(pose_jump.get("max", 0.0)) if pose_jump.get("n", 0) else 0.0
    min_score = float(score.get("min", 0.0)) if score.get("n", 0) else 0.0
    if max_jump > thresholds.pose_jump_fail:
        return "large_local_jump"
    if min_score <= thresholds.low_score:
        return "low_score_scan_match"
    if max_jump <= thresholds.pose_jump_warn:
        return "drift_growth_without_local_jump"
    return "mixed_local_odometry_signal"


def _loop_effect_in_window(
    components: dict[str, list[float] | list[int]],
    loop_events: dict[int, dict[str, Any]],
    start_i: int,
    end_i: int,
) -> dict[str, Any]:
    nodes = [int(n) for n in components["nodes"]]
    trans = [float(v) for v in components["translation_m"]]
    start_node = nodes[start_i]
    end_node = nodes[end_i]
    events = [
        (int(node), ev)
        for node, ev in loop_events.items()
        if start_node <= int(node) <= end_node
    ]
    if not events:
        return {"verdict": "no_loop_events", "accepted_nodes": []}

    accepted_nodes = sorted(
        node for node, ev in events if int(ev.get("accepted", 0)) > 0
    )
    rejected_nodes = sorted(
        node for node, ev in events if int(ev.get("rejected", 0)) > 0
    )
    candidate_nodes = sorted(
        node for node, ev in events if int(ev.get("candidate_count", 0)) > 0
    )
    if not accepted_nodes:
        return {
            "verdict": "candidates_without_acceptance",
            "candidate_nodes": _node_sample(candidate_nodes, limit=10),
            "rejected_nodes": _node_sample(rejected_nodes, limit=10),
            "accepted_nodes": [],
        }

    node_index = {node: i for i, node in enumerate(nodes)}
    accepted_idx = [
        node_index[node]
        for node in accepted_nodes
        if start_i <= node_index.get(node, -1) <= end_i
    ]
    if not accepted_idx:
        return {
            "verdict": "accepted_events_without_pose_sample",
            "accepted_nodes": _node_sample(accepted_nodes, limit=10),
        }

    first_i = min(accepted_idx)
    last_i = max(accepted_idx)
    start_corr = trans[start_i]
    first_corr = trans[first_i]
    last_corr = trans[last_i]
    end_corr = trans[end_i]
    after_first = trans[first_i : end_i + 1]
    after_last = trans[last_i : end_i + 1]
    reduction_after_first = first_corr - min(after_first)
    reduction_after_last = last_corr - min(after_last)
    growth_after_last = end_corr - last_corr
    if last_i == end_i:
        verdict = "accepted_at_window_end"
    elif growth_after_last > 0.05:
        verdict = "accepted_loops_but_correction_keeps_growing"
    elif max(reduction_after_first, reduction_after_last) > 0.05:
        verdict = "accepted_loops_reduce_correction"
    else:
        verdict = "accepted_loops_hold_correction"
    return {
        "verdict": verdict,
        "accepted_nodes": _node_sample(accepted_nodes, limit=10),
        "first_accepted_node": nodes[first_i],
        "last_accepted_node": nodes[last_i],
        "start_correction_m": start_corr,
        "first_accepted_correction_m": first_corr,
        "last_accepted_correction_m": last_corr,
        "end_correction_m": end_corr,
        "growth_before_first_accept_m": first_corr - start_corr,
        "growth_after_last_accept_m": growth_after_last,
        "reduction_after_first_accept_m": reduction_after_first,
        "reduction_after_last_accept_m": reduction_after_last,
    }


def _hotspot_debug_target(
    row: dict[str, Any],
    thresholds: CloudAnalyzerThresholds,
) -> str:
    loops = row["loop_events"]
    effect = row["loop_effect"]
    delta = row["delta"]
    opt = row["optimization"]
    if loops["candidate_count"] == 0 and loops["accepted"] == 0 and loops["rejected"] == 0:
        return "loop_candidate_search"
    if loops["accepted"] == 0:
        return "candidate_scoring_or_acceptance_gate"
    if abs(float(delta["yaw_deg"])) > 1.0:
        return "yaw_refinement"
    reductions = opt.get("residual_rms_reduction", {})
    max_reduction = float(reductions.get("max", 0.0)) if reductions.get("n", 0) else 0.0
    if opt.get("events", 0) == 0 or max_reduction < thresholds.residual_reduction_good:
        return "pose_graph_edge_weight_or_optimizer"
    if effect.get("verdict") == "accepted_loops_but_correction_keeps_growing":
        return "odometry_bias_after_loop"
    return "loop_closure_effective"


def _make_hotspot_row(
    components: dict[str, list[float] | list[int]],
    metrics: dict[int, dict[str, Any]],
    loop_events: dict[int, dict[str, Any]],
    opt_evs: list[dict[str, Any]],
    start_i: int,
    end_i: int,
    *,
    window_nodes: int,
    thresholds: CloudAnalyzerThresholds,
) -> dict[str, Any]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    longitudinal = components["longitudinal_m"]
    lateral = components["lateral_m"]
    yaw = components["yaw_rad"]
    start_node = int(nodes[start_i])
    end_node = int(nodes[end_i])
    start_trans = float(trans[start_i])
    end_trans = float(trans[end_i])
    d_long = float(longitudinal[end_i]) - float(longitudinal[start_i])
    d_lat = float(lateral[end_i]) - float(lateral[start_i])
    d_yaw = _wrap_pi(float(yaw[end_i]) - float(yaw[start_i]))
    dominant = "longitudinal" if abs(d_long) >= abs(d_lat) else "lateral"
    correction = [float(v) for v in trans[start_i : end_i + 1]]
    row = {
        "window_nodes": int(window_nodes),
        "start_node": start_node,
        "end_node": end_node,
        "start_correction_m": start_trans,
        "end_correction_m": end_trans,
        "growth_m": float(end_trans - start_trans),
        "delta": {
            "longitudinal_m": d_long,
            "lateral_m": d_lat,
            "yaw_rad": d_yaw,
            "yaw_deg": math.degrees(d_yaw),
            "dominant_translation_component": dominant,
        },
        "correction_m": series_summary(correction),
        "monotonic_growth_fraction": _monotonic_growth_fraction(correction),
        "loop_events": _event_counts_in_range(loop_events, start_node, end_node),
        "loop_effect": _loop_effect_in_window(components, loop_events, start_i, end_i),
        "optimization": _optimization_summary_in_range(opt_evs, start_node, end_node),
        "pose_jump": _metrics_in_range(metrics, start_node, end_node, "pose_jump"),
        "scan_match_score": _metrics_in_range(
            metrics, start_node, end_node, "scan_match_score"
        ),
        "prediction_error_m": _metrics_in_range(
            metrics, start_node, end_node, "prediction_error_m"
        ),
        "odometry_step_m": _metrics_in_range(
            metrics, start_node, end_node, "odometry_step_m"
        ),
    }
    row["failure_mode"] = _hotspot_failure_mode(row, thresholds)
    row["debug_target"] = _hotspot_debug_target(row, thresholds)
    return row


def _detect_drift_hotspots(
    components: dict[str, list[float] | list[int]],
    metrics: dict[int, dict[str, Any]],
    loop_events: dict[int, dict[str, Any]],
    opt_evs: list[dict[str, Any]],
    *,
    thresholds: CloudAnalyzerThresholds,
    window_nodes: int = 100,
    limit: int = 5,
    min_growth_m: float = 0.25,
) -> list[dict[str, Any]]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    n = len(nodes)
    if n < 2 or limit <= 0:
        return []

    step = max(1, min(int(window_nodes), n - 1))
    candidates: list[tuple[float, int, int]] = []
    for end_i in range(step, n):
        start_i = end_i - step
        growth = float(trans[end_i]) - float(trans[start_i])
        if growth >= min_growth_m:
            candidates.append((growth, start_i, end_i))
    candidates.sort(reverse=True)

    selected: list[tuple[int, int]] = []
    out: list[dict[str, Any]] = []
    for _, start_i, end_i in candidates:
        if any(not (end_i < s or start_i > e) for s, e in selected):
            continue
        selected.append((start_i, end_i))
        row = _make_hotspot_row(
            components,
            metrics,
            loop_events,
            opt_evs,
            start_i,
            end_i,
            window_nodes=step,
            thresholds=thresholds,
        )
        row["rank"] = len(out) + 1
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _hotspot_diagnostics(
    hotspots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    for h in hotspots[:3]:
        target = h.get("debug_target", "unknown")
        effect = h.get("loop_effect") or {}
        nodes = {"start": h.get("start_node"), "end": h.get("end_node")}
        common = {
            "nodes": nodes,
            "growth_m": h.get("growth_m"),
            "rank": h.get("rank"),
        }
        if target == "candidate_scoring_or_acceptance_gate":
            findings.append(
                {
                    "level": "warn",
                    "message": "drift hotspot has loop candidates but no accepted loops",
                    **common,
                }
            )
            suggestions.append(
                {
                    "why": "candidate scoring or acceptance gates block this hotspot",
                    "try": [
                        "inspect rejected loop candidates in this node window",
                        "compare yaw/refined score before and after ICP",
                        "relax acceptance gates only around this diagnostic window",
                    ],
                }
            )
        elif target == "yaw_refinement":
            findings.append(
                {
                    "level": "warn",
                    "message": "drift hotspot is dominated by yaw-sensitive loop correction",
                    **common,
                }
            )
            suggestions.append(
                {
                    "why": "accepted loops coincide with a yaw correction hotspot",
                    "try": [
                        "increase yaw refinement hypotheses around accepted loop candidates",
                        "verify candidate ranking after yaw refinement",
                        "inspect whether lateral drift changes sign across the window",
                    ],
                }
            )
        elif target == "pose_graph_edge_weight_or_optimizer":
            findings.append(
                {
                    "level": "warn",
                    "message": "accepted loops have weak optimization evidence in hotspot",
                    **common,
                }
            )
            suggestions.append(
                {
                    "why": "loop edges are accepted but pose-graph residual reduction is weak",
                    "try": [
                        "inspect loop edge information weights for this node window",
                        "raise loop-edge weight or lower odometry-edge confidence in a test run",
                        "check optimizer nfev/max_nfev and final residuals",
                    ],
                }
            )
        elif target == "odometry_bias_after_loop":
            findings.append(
                {
                    "level": "warn",
                    "message": "correction keeps growing after accepted loop closures",
                    "growth_after_last_accept_m": effect.get("growth_after_last_accept_m"),
                    **common,
                }
            )
            suggestions.append(
                {
                    "why": "loop closure works but odometry bias continues after the loop event",
                    "try": [
                        "focus local odometry bias tuning after the last accepted loop node",
                        "compare submap size and yaw bias on this exact interval",
                        "use cloud-hotspot on the reported node window for node-level metrics",
                    ],
                }
            )
    return findings, suggestions


def _drift_decomposition_summary(
    components: dict[str, list[float] | list[int]],
) -> dict[str, Any]:
    nodes = components["nodes"]
    trans = components["translation_m"]
    longitudinal = components["longitudinal_m"]
    lateral = components["lateral_m"]
    yaw = components["yaw_rad"]
    if not nodes:
        return {}

    abs_long = [abs(float(v)) for v in longitudinal]
    abs_lat = [abs(float(v)) for v in lateral]
    abs_yaw = [abs(float(v)) for v in yaw]
    max_long_i = int(max(range(len(nodes)), key=lambda i: abs_long[i]))
    max_lat_i = int(max(range(len(nodes)), key=lambda i: abs_lat[i]))
    max_yaw_i = int(max(range(len(nodes)), key=lambda i: abs_yaw[i]))
    end_long = float(longitudinal[-1])
    end_lat = float(lateral[-1])
    dominant = "longitudinal" if abs(end_long) >= abs(end_lat) else "lateral"

    return {
        "longitudinal_abs_m": series_summary(abs_long),
        "lateral_abs_m": series_summary(abs_lat),
        "yaw_abs_rad": series_summary(abs_yaw),
        "end": {
            "translation_m": float(trans[-1]),
            "longitudinal_m": end_long,
            "lateral_m": end_lat,
            "yaw_rad": float(yaw[-1]),
            "yaw_deg": math.degrees(float(yaw[-1])),
            "dominant_translation_component": dominant,
        },
        "max_abs_longitudinal": {
            "node": int(nodes[max_long_i]),
            "value_m": float(longitudinal[max_long_i]),
        },
        "max_abs_lateral": {
            "node": int(nodes[max_lat_i]),
            "value_m": float(lateral[max_lat_i]),
        },
        "max_abs_yaw": {
            "node": int(nodes[max_yaw_i]),
            "value_rad": float(yaw[max_yaw_i]),
            "value_deg": math.degrees(float(yaw[max_yaw_i])),
        },
    }


def _trajectory_compare(
    baseline: list[dict[str, float]],
    looped: list[dict[str, float]],
) -> dict[str, Any]:
    n = min(len(baseline), len(looped))
    if n == 0:
        return {"ok": False, "message": "empty trajectory"}

    components = _trajectory_drift_components(baseline, looped, n)
    nodes = components["nodes"]
    trans = components["translation_m"]
    yaw = components["yaw_rad"]
    rot = [abs(float(v)) for v in yaw]
    worst_i = int(max(range(n), key=lambda i: trans[i]))
    return {
        "ok": True,
        "compared_poses": n,
        "max_translation_m": float(trans[worst_i]),
        "max_translation_index": worst_i,
        "max_translation_node": int(nodes[worst_i]),
        "end_translation_m": float(trans[-1]),
        "translation_m": series_summary(trans),
        "max_rotation_rad": max(rot),
        "baseline_start_end_gap_m": _start_end_gap(baseline[:n]),
        "loop_start_end_gap_m": _start_end_gap(looped[:n]),
        "drift_growth": {
            "samples": _sample_drift_curve(components),
            "threshold_crossings": _drift_threshold_crossings(components),
            "max_growth_per_100_nodes": _max_window_growth(components, window_nodes=100),
        },
        "drift_decomposition": _drift_decomposition_summary(components),
    }


def _score_margins(evs: list[dict[str, Any]]) -> list[float]:
    margins: list[float] = []
    for e in evs:
        if e.get("type") != "scan_match_candidates":
            continue
        top = e.get("top") or []
        if isinstance(top, list) and len(top) >= 2:
            try:
                margins.append(abs(float(top[0][3]) - float(top[1][3])))
            except Exception:
                pass
    return margins


def _node_sample(nodes: list[int], limit: int = 20) -> list[int]:
    if len(nodes) <= limit:
        return nodes
    return nodes[: limit // 2] + nodes[-(limit // 2) :]


def _top_metric_nodes(
    nodes: list[int], values: list[float], limit: int = 8
) -> list[dict[str, Any]]:
    pairs = sorted(zip(nodes, values, strict=False), key=lambda p: p[1], reverse=True)
    return [{"node": int(n), "value": float(v)} for n, v in pairs[:limit]]


def _keyframe_metrics_by_node(evs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    last_pose: dict[str, Any] | None = None
    for e in evs:
        if e.get("type") != "keyframe":
            continue
        node = int(e.get("node", -1))
        pose = e.get("pose") or {}
        pred = e.get("prediction") or {}
        pred_error_m = None
        pred_error_yaw_rad = None
        step_m = None
        step_yaw_rad = None
        if isinstance(pose, dict) and isinstance(pred, dict):
            try:
                pred_error_m = float(
                    math.hypot(
                        float(pose["x"]) - float(pred["x"]),
                        float(pose["y"]) - float(pred["y"]),
                    )
                )
                pred_error_yaw_rad = abs(_wrap_pi(float(pose["theta"]) - float(pred["theta"])))
            except Exception:
                pass
        if isinstance(pose, dict) and last_pose is not None:
            try:
                step_m = float(
                    math.hypot(
                        float(pose["x"]) - float(last_pose["x"]),
                        float(pose["y"]) - float(last_pose["y"]),
                    )
                )
                step_yaw_rad = abs(_wrap_pi(float(pose["theta"]) - float(last_pose["theta"])))
            except Exception:
                pass
        rows[node] = {
            "pose_jump": float(e.get("pose_jump", 0.0)),
            "scan_match_score": float(e.get("scan_match_score", 0.0)),
            "prediction_error_m": pred_error_m,
            "prediction_error_yaw_rad": pred_error_yaw_rad,
            "odometry_step_m": step_m,
            "odometry_step_yaw_rad": step_yaw_rad,
        }
        if isinstance(pose, dict):
            last_pose = pose
    return rows


def _loop_events_by_node(evs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}

    def ensure(node: int) -> dict[str, Any]:
        if node not in rows:
            rows[node] = {
                "candidate_count": 0,
                "accepted": 0,
                "rejected": 0,
                "accepted_edges": [],
                "rejected_edges": [],
            }
        return rows[node]

    for e in evs:
        typ = e.get("type")
        if typ == "loop_closure_candidates":
            node = int(e.get("node", -1))
            cands = e.get("candidates")
            ensure(node)["candidate_count"] += len(cands) if isinstance(cands, list) else 0
        elif typ == "loop_closure_accepted":
            node = int(e.get("node", -1))
            row = ensure(node)
            row["accepted"] += 1
            row["accepted_edges"].append([int(e.get("i", -1)), int(e.get("j", -1))])
        elif typ == "loop_closure_rejected":
            node = int(e.get("node", -1))
            row = ensure(node)
            row["rejected"] += 1
            row["rejected_edges"].append([int(e.get("i", -1)), int(e.get("j", -1))])
    return rows


def _loop_event_label(ev: dict[str, Any] | None) -> str:
    if not ev:
        return "-"
    parts: list[str] = []
    if ev.get("candidate_count", 0):
        parts.append(f"cand:{int(ev['candidate_count'])}")
    if ev.get("accepted", 0):
        parts.append(f"acc:{int(ev['accepted'])}")
    if ev.get("rejected", 0):
        parts.append(f"rej:{int(ev['rejected'])}")
    return " ".join(parts) if parts else "-"


def _summary_or_empty(xs: list[float | None]) -> dict[str, Any]:
    return series_summary([float(x) for x in xs if x is not None])


@dataclass(frozen=True)
class CloudAnalyzerThresholds:
    odom_gap_warn_m: float = 0.5
    odom_gap_fail_m: float = 1.0
    loop_gap_good_m: float = 0.15
    loop_delta_warn_m: float = 0.5
    pose_jump_warn: float = 0.75
    pose_jump_fail: float = 1.5
    low_score: float = -5.0
    ambiguous_margin: float = 0.02
    residual_reduction_good: float = 5.0


class CloudAnalyzer:
    """Analyze run artifacts for loop-closure viability and LiDAR odometry drift."""

    def __init__(
        self,
        run_dir: Path,
        *,
        baseline_run: Path | None = None,
        thresholds: CloudAnalyzerThresholds | None = None,
        hotspot_window_nodes: int = 100,
        hotspot_limit: int = 5,
    ) -> None:
        self.run_dir = run_dir
        self.baseline_run = baseline_run
        self.thresholds = thresholds or CloudAnalyzerThresholds()
        self.hotspot_window_nodes = int(hotspot_window_nodes)
        self.hotspot_limit = int(hotspot_limit)

    def analyze(self) -> dict[str, Any]:
        traj_path = _run_file(self.run_dir, "trajectory.json")
        telem_path = _run_file(self.run_dir, "telemetry.jsonl")

        out: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "baseline_run": str(self.baseline_run) if self.baseline_run else None,
            "ok": True,
            "facts": {},
            "inference": {},
            "findings": [],
            "suggestions": [],
        }

        missing = []
        if not traj_path.exists():
            missing.append(str(traj_path))
        if not telem_path.exists():
            missing.append(str(telem_path))
        if missing:
            out["ok"] = False
            out["findings"].append(
                {"level": "error", "message": "missing required run artifacts", "paths": missing}
            )
            return out

        traj = _load_trajectory(traj_path)
        evs = load_jsonl(telem_path)

        telemetry = self._telemetry_facts(evs)
        trajectory = _trajectory_summary(traj)
        out["facts"] = {
            "trajectory": trajectory,
            "telemetry": telemetry,
        }

        baseline_traj: list[dict[str, float]] | None = None
        baseline_evs: list[dict[str, Any]] | None = None
        odom_telemetry = telemetry
        odom_telemetry_source = "run"
        if self.baseline_run is not None:
            baseline_path = _run_file(self.baseline_run, "trajectory.json")
            if baseline_path.exists():
                baseline_traj = _load_trajectory(baseline_path)
                out["facts"]["baseline_trajectory"] = _trajectory_summary(baseline_traj)
                out["facts"]["baseline_vs_loop"] = _trajectory_compare(baseline_traj, traj)
            else:
                out["findings"].append(
                    {
                        "level": "warn",
                        "message": "baseline trajectory missing; odometry verdict is weaker",
                        "path": str(baseline_path),
                    }
                )
            baseline_telem_path = _run_file(self.baseline_run, "telemetry.jsonl")
            if baseline_telem_path.exists():
                baseline_evs = load_jsonl(baseline_telem_path)
                baseline_telemetry = self._telemetry_facts(baseline_evs)
                out["facts"]["baseline_telemetry"] = baseline_telemetry
                odom_telemetry = baseline_telemetry
                odom_telemetry_source = "baseline_run"
            if baseline_traj is not None:
                n = min(len(baseline_traj), len(traj))
                components = _trajectory_drift_components(baseline_traj, traj, n)
                metric_evs = baseline_evs if baseline_evs is not None else evs
                out["facts"]["baseline_vs_loop"]["hotspots"] = _detect_drift_hotspots(
                    components,
                    _keyframe_metrics_by_node(metric_evs),
                    _loop_events_by_node(evs),
                    evs,
                    thresholds=self.thresholds,
                    window_nodes=self.hotspot_window_nodes,
                    limit=self.hotspot_limit,
                )
                out["facts"]["baseline_vs_loop"]["hotspot_telemetry_source"] = (
                    odom_telemetry_source
                )

        loop_inf = self._infer_loop_closure(telemetry, trajectory)
        odom_inf = self._infer_odometry(
            odom_telemetry,
            trajectory,
            baseline_traj,
            traj,
            telemetry_source=odom_telemetry_source,
        )
        out["inference"] = {
            "loop_closure": loop_inf,
            "lidar_odometry": odom_inf,
        }

        out["findings"].extend(loop_inf["findings"])
        out["findings"].extend(odom_inf["findings"])
        out["suggestions"].extend(loop_inf["suggestions"])
        out["suggestions"].extend(odom_inf["suggestions"])
        hotspots = out.get("facts", {}).get("baseline_vs_loop", {}).get("hotspots", [])
        hotspot_findings, hotspot_suggestions = _hotspot_diagnostics(hotspots)
        out["findings"].extend(hotspot_findings)
        out["suggestions"].extend(hotspot_suggestions)
        return out

    def _telemetry_facts(self, evs: list[dict[str, Any]]) -> dict[str, Any]:
        keyframes = [e for e in evs if e.get("type") == "keyframe"]
        pose_jumps = [float(e.get("pose_jump", 0.0)) for e in keyframes]
        scores = [float(e.get("scan_match_score", 0.0)) for e in keyframes]
        keyframe_nodes = [int(e.get("node", i)) for i, e in enumerate(keyframes)]

        pred_trans: list[float] = []
        pred_rot: list[float] = []
        pred_nodes: list[int] = []
        step_trans: list[float] = []
        step_rot: list[float] = []
        step_nodes: list[int] = []
        last_pose: dict[str, Any] | None = None
        for e in keyframes:
            node = int(e.get("node", -1))
            pose = e.get("pose") or {}
            pred = e.get("prediction") or {}
            if isinstance(pose, dict) and isinstance(pred, dict):
                try:
                    pred_trans.append(
                        math.hypot(
                            float(pose["x"]) - float(pred["x"]),
                            float(pose["y"]) - float(pred["y"]),
                        )
                    )
                    pred_rot.append(abs(_wrap_pi(float(pose["theta"]) - float(pred["theta"]))))
                    pred_nodes.append(node)
                except Exception:
                    pass
            if isinstance(pose, dict) and last_pose is not None:
                try:
                    step_trans.append(
                        math.hypot(
                            float(pose["x"]) - float(last_pose["x"]),
                            float(pose["y"]) - float(last_pose["y"]),
                        )
                    )
                    step_rot.append(abs(_wrap_pi(float(pose["theta"]) - float(last_pose["theta"]))))
                    step_nodes.append(node)
                except Exception:
                    pass
            if isinstance(pose, dict):
                last_pose = pose

        low_score_nodes = [
            int(e.get("node", -1))
            for e in keyframes
            if float(e.get("scan_match_score", 0.0)) < self.thresholds.low_score
        ]
        jump_warn_nodes = [
            int(e.get("node", -1))
            for e in keyframes
            if float(e.get("pose_jump", 0.0)) > self.thresholds.pose_jump_warn
        ]
        jump_fail_nodes = [
            int(e.get("node", -1))
            for e in keyframes
            if float(e.get("pose_jump", 0.0)) > self.thresholds.pose_jump_fail
        ]

        candidate_events = [e for e in evs if e.get("type") == "loop_closure_candidates"]
        accepted = [e for e in evs if e.get("type") == "loop_closure_accepted"]
        rejected = [e for e in evs if e.get("type") == "loop_closure_rejected"]
        total_candidates = 0
        for e in candidate_events:
            cands = e.get("candidates")
            if isinstance(cands, list):
                total_candidates += len(cands)

        margins = _score_margins(evs)
        ambiguous_nodes = []
        for e in evs:
            if e.get("type") != "scan_match_candidates":
                continue
            top = e.get("top") or []
            if isinstance(top, list) and len(top) >= 2:
                try:
                    if abs(float(top[0][3]) - float(top[1][3])) < self.thresholds.ambiguous_margin:
                        ambiguous_nodes.append(int(e.get("node", -1)))
                except Exception:
                    pass

        opt = [e for e in evs if e.get("type") == "optimization"]
        final_opt = next((e for e in reversed(opt) if e.get("final")), opt[-1] if opt else None)
        final_opt_summary = None
        if final_opt:
            before = final_opt.get("residual_rms_before")
            after = final_opt.get("residual_rms_after")
            reduction = None
            if before is not None and after is not None and float(after) > 0:
                reduction = float(before) / float(after)
            final_opt_summary = {
                "node": final_opt.get("node"),
                "final": bool(final_opt.get("final", False)),
                "success": bool(final_opt.get("success", False)),
                "n_poses": final_opt.get("n_poses"),
                "n_edges": final_opt.get("n_edges"),
                "nfev": final_opt.get("nfev"),
                "max_nfev": final_opt.get("max_nfev"),
                "residual_rms_before": before,
                "residual_rms_after": after,
                "residual_rms_reduction": reduction,
            }

        accepted_nodes = [int(e.get("node", -1)) for e in accepted]
        accept_den = len(accepted) + len(rejected)
        return {
            "events": len(evs),
            "keyframes": len(keyframes),
            "scan_match_score": series_summary(scores),
            "pose_jump": series_summary(pose_jumps),
            "prediction_error": {
                "translation_m": series_summary(pred_trans),
                "rotation_rad": series_summary(pred_rot),
                "worst_translation_nodes": _top_metric_nodes(pred_nodes, pred_trans),
            },
            "odometry_step": {
                "translation_m": series_summary(step_trans),
                "rotation_rad": series_summary(step_rot),
                "worst_translation_nodes": _top_metric_nodes(step_nodes, step_trans),
            },
            "low_score_nodes": _node_sample(low_score_nodes),
            "large_pose_jump_nodes": _node_sample(jump_warn_nodes),
            "very_large_pose_jump_nodes": _node_sample(jump_fail_nodes),
            "ambiguous_scan_match_nodes": _node_sample(ambiguous_nodes),
            "worst_pose_jump_nodes": _top_metric_nodes(keyframe_nodes, pose_jumps),
            "scan_match_top2_margin": series_summary(margins),
            "loop_candidates": {
                "events": len(candidate_events),
                "total_candidates": total_candidates,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "accept_ratio": (len(accepted) / accept_den) if accept_den else None,
                "unique_accepted_nodes": len(set(accepted_nodes)),
                "first_accepted_node": min(accepted_nodes) if accepted_nodes else None,
                "last_accepted_node": max(accepted_nodes) if accepted_nodes else None,
            },
            "optimization": {
                "events": len(opt),
                "final_or_last": final_opt_summary,
            },
        }

    def _infer_loop_closure(
        self,
        telemetry: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> dict[str, Any]:
        t = self.thresholds
        loop = telemetry["loop_candidates"]
        opt = telemetry["optimization"]["final_or_last"]
        accepted = int(loop["accepted"])
        rejected = int(loop["rejected"])
        candidates = int(loop["total_candidates"])

        findings: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        status = "unknown"
        can_close = "unknown"
        evidence: list[str] = []

        if accepted > 0:
            status = "closed"
            can_close = "yes"
            evidence.append(f"{accepted} loop closures were accepted")
            findings.append(
                {
                    "level": "info",
                    "message": "loop closure is observed in telemetry",
                    "accepted": accepted,
                    "first_node": loop["first_accepted_node"],
                    "last_node": loop["last_accepted_node"],
                }
            )
        elif candidates > 0 or rejected > 0:
            status = "candidates_rejected"
            can_close = "maybe"
            evidence.append("loop candidates were generated but none were accepted")
            findings.append(
                {
                    "level": "warn",
                    "message": "loop candidates exist but gates reject them",
                    "candidates": candidates,
                    "rejected": rejected,
                }
            )
            suggestions.append(
                {
                    "why": "loop candidates are present but not accepted",
                    "try": [
                        "inspect loop_closure_rejected nodes and ICP/correlative diagnostics",
                        "relax loop_detection.accept_score or icp_accept_rms cautiously",
                        "increase loop_detection.search_radius_m if odometry drift is large",
                    ],
                }
            )
        else:
            status = "not_observed"
            can_close = "unknown"
            evidence.append("no loop candidates were reported")
            findings.append(
                {
                    "level": "warn",
                    "message": "no loop-closure candidates observed",
                }
            )
            suggestions.append(
                {
                    "why": "no loop candidates were generated",
                    "try": [
                        "confirm the sequence actually revisits a previous place",
                        "enable loop_detection or scan_ba.loop_closure_enabled",
                        "increase search radius or reduce loop_detect_every_n for diagnosis runs",
                    ],
                }
            )

        if opt and opt.get("residual_rms_reduction") is not None:
            reduction = float(opt["residual_rms_reduction"])
            if reduction >= t.residual_reduction_good:
                evidence.append(f"pose-graph RMS improved {reduction:.1f}x")
                findings.append(
                    {
                        "level": "info",
                        "message": "pose graph optimization strongly reduced residuals",
                        "residual_rms_before": opt.get("residual_rms_before"),
                        "residual_rms_after": opt.get("residual_rms_after"),
                        "reduction": reduction,
                    }
                )

        gap = float(trajectory["start_end_gap_m"])
        if accepted > 0 and gap <= t.loop_gap_good_m:
            evidence.append(f"final start/end gap is small ({gap:.3f} m)")

        return {
            "status": status,
            "can_close": can_close,
            "evidence": evidence,
            "findings": findings,
            "suggestions": suggestions,
        }

    def _infer_odometry(
        self,
        telemetry: dict[str, Any],
        trajectory: dict[str, Any],
        baseline: list[dict[str, float]] | None,
        looped: list[dict[str, float]],
        *,
        telemetry_source: str,
    ) -> dict[str, Any]:
        t = self.thresholds
        findings: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        evidence: list[str] = []
        status = "unknown"
        failure_mode = "unknown"

        if baseline is not None:
            cmp = _trajectory_compare(baseline, looped)
            raw_gap = float(cmp["baseline_start_end_gap_m"])
            loop_gap = float(cmp["loop_start_end_gap_m"])
            max_delta = float(cmp["max_translation_m"])
            end_delta = float(cmp["end_translation_m"])
            evidence.append(f"baseline start/end gap {raw_gap:.3f} m")
            evidence.append(f"looped start/end gap {loop_gap:.3f} m")
            evidence.append(f"max baseline-vs-loop correction {max_delta:.3f} m")

            if raw_gap >= t.odom_gap_fail_m and (
                loop_gap <= t.loop_gap_good_m or max_delta >= t.loop_delta_warn_m
            ):
                status = "failing_drift_corrected_by_loop"
                findings.append(
                    {
                        "level": "warn",
                        "message": (
                            "no-loop LiDAR odometry drifts badly and loop closure corrects it"
                        ),
                        "baseline_start_end_gap_m": raw_gap,
                        "loop_start_end_gap_m": loop_gap,
                        "max_correction_m": max_delta,
                        "end_correction_m": end_delta,
                    }
                )
            elif raw_gap >= t.odom_gap_warn_m:
                status = "drifting"
                findings.append(
                    {
                        "level": "warn",
                        "message": "no-loop LiDAR odometry has measurable drift",
                        "baseline_start_end_gap_m": raw_gap,
                    }
                )
            else:
                status = "stable_by_baseline_gap"
                findings.append(
                    {
                        "level": "info",
                        "message": "baseline odometry start/end gap is small",
                        "baseline_start_end_gap_m": raw_gap,
                    }
                )
        else:
            jumps = telemetry["pose_jump"]
            max_jump = float(jumps.get("max", 0.0)) if jumps.get("n", 0) else 0.0
            p90_jump = float(jumps.get("p90", 0.0)) if jumps.get("n", 0) else 0.0
            if telemetry["very_large_pose_jump_nodes"]:
                status = "unstable_scan_matching"
                findings.append(
                    {
                        "level": "warn",
                        "message": "very large keyframe jumps detected",
                        "nodes": telemetry["very_large_pose_jump_nodes"],
                    }
                )
            elif telemetry["large_pose_jump_nodes"] or max_jump > t.pose_jump_warn:
                status = "possible_instability"
                findings.append(
                    {
                        "level": "warn",
                        "message": "large keyframe jumps detected",
                        "nodes": telemetry["large_pose_jump_nodes"],
                        "p90_pose_jump": p90_jump,
                        "max_pose_jump": max_jump,
                    }
                )
            else:
                status = "no_strong_failure_signal"
                findings.append(
                    {
                        "level": "info",
                        "message": (
                            "no large pose-jump signal; provide --baseline-run for drift verdict"
                        ),
                    }
                )
            evidence.append("baseline run not provided; drift verdict is weaker")

        source = self._infer_odometry_failure_source(telemetry, status)
        failure_mode = source["failure_mode"]
        evidence.extend(source["evidence"])
        findings.extend(source["findings"])
        suggestions.extend(source["suggestions"])

        if telemetry["low_score_nodes"]:
            findings.append(
                {
                    "level": "warn",
                    "message": "low scan-match scores may indicate local odometry trouble",
                    "nodes": telemetry["low_score_nodes"],
                }
            )
            suggestions.append(
                {
                    "why": "low scan-match scores were observed",
                    "try": [
                        "increase local matcher search window for diagnosis",
                        "reduce preprocessing stride to keep more LiDAR points",
                        "check range limits and virtual scan projection",
                    ],
                }
            )

        if telemetry["ambiguous_scan_match_nodes"]:
            findings.append(
                {
                    "level": "warn",
                    "message": "scan matching has ambiguous top candidates",
                    "nodes": telemetry["ambiguous_scan_match_nodes"],
                }
            )
            suggestions.append(
                {
                    "why": "top scan-match candidates are close in score",
                    "try": [
                        "use multi-initialization or geometric verification for loop matches",
                        "tighten acceptance around self-similar corridors",
                    ],
                }
            )

        if status.startswith("failing") or status in {"drifting", "unstable_scan_matching"}:
            suggestions.append(
                {
                    "why": "LiDAR odometry drift or instability is visible",
                    "try": [
                        "enable loop closure for full-length runs",
                        "compare no-loop and loop runs with cloud-analyze --baseline-run",
                        "record accepted/rejected loop nodes and tune gates around first failure",
                    ],
                }
            )

        return {
            "status": status,
            "failure_mode": failure_mode,
            "telemetry_source": telemetry_source,
            "evidence": evidence,
            "findings": findings,
            "suggestions": suggestions,
        }

    def _infer_odometry_failure_source(
        self, telemetry: dict[str, Any], status: str
    ) -> dict[str, Any]:
        t = self.thresholds
        findings: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        evidence: list[str] = []

        jump = telemetry["pose_jump"]
        score = telemetry["scan_match_score"]
        pred = telemetry["prediction_error"]["translation_m"]
        max_jump = float(jump.get("max", 0.0)) if jump.get("n", 0) else 0.0
        p90_jump = float(jump.get("p90", 0.0)) if jump.get("n", 0) else 0.0
        max_pred = float(pred.get("max", 0.0)) if pred.get("n", 0) else 0.0
        score_min = float(score.get("min", 0.0)) if score.get("n", 0) else 0.0
        drifting = status.startswith("failing") or status == "drifting"

        evidence.append(f"pose_jump p90/max {p90_jump:.3f}/{max_jump:.3f}")
        evidence.append(f"prediction error max {max_pred:.3f} m")
        evidence.append(f"scan_match_score min {score_min:.3f}")

        if telemetry["very_large_pose_jump_nodes"]:
            return {
                "failure_mode": "large_local_jump",
                "evidence": evidence,
                "findings": [
                    {
                        "level": "warn",
                        "message": "odometry failure source looks like local scan-match jumps",
                        "nodes": telemetry["very_large_pose_jump_nodes"],
                    }
                ],
                "suggestions": [
                    {
                        "why": "large local pose jumps dominate the odometry failure",
                        "try": [
                            "inspect the listed nodes and surrounding scans",
                            "tighten local matcher acceptance or add relocalization fallback",
                        ],
                    }
                ],
            }

        if telemetry["low_score_nodes"]:
            return {
                "failure_mode": "low_score_scan_match",
                "evidence": evidence,
                "findings": [],
                "suggestions": [],
            }

        if telemetry["ambiguous_scan_match_nodes"]:
            return {
                "failure_mode": "ambiguous_scan_match",
                "evidence": evidence,
                "findings": [],
                "suggestions": [],
            }

        if drifting and max_jump <= t.pose_jump_warn and not telemetry["large_pose_jump_nodes"]:
            findings.append(
                {
                    "level": "warn",
                    "message": "odometry drift accumulates without large local scan-match jumps",
                    "pose_jump_p90": p90_jump,
                    "pose_jump_max": max_jump,
                    "scan_match_score_min": score_min,
                }
            )
            suggestions.append(
                {
                    "why": "drift appears to be accumulated small odometry bias, not a single jump",
                    "try": [
                        "measure closed-loop drift on no-loop runs as a primary odometry metric",
                        "tune local scan-to-submap matching for systematic yaw/translation bias",
                        "keep loop closure enabled for long runs and inspect first loop node",
                    ],
                }
            )
            return {
                "failure_mode": "accumulated_drift_without_local_jump",
                "evidence": evidence,
                "findings": findings,
                "suggestions": suggestions,
            }

        return {
            "failure_mode": "no_clear_local_failure_source",
            "evidence": evidence,
            "findings": findings,
            "suggestions": suggestions,
        }


class CloudHotspotAnalyzer:
    """Inspect a concrete drift-growth node interval."""

    def __init__(
        self,
        run_dir: Path,
        *,
        baseline_run: Path,
        start_node: int,
        end_node: int,
        thresholds: CloudAnalyzerThresholds | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.baseline_run = baseline_run
        self.start_node = min(int(start_node), int(end_node))
        self.end_node = max(int(start_node), int(end_node))
        self.thresholds = thresholds or CloudAnalyzerThresholds()

    def analyze(self) -> dict[str, Any]:
        traj_path = _run_file(self.run_dir, "trajectory.json")
        telem_path = _run_file(self.run_dir, "telemetry.jsonl")
        baseline_traj_path = _run_file(self.baseline_run, "trajectory.json")
        baseline_telem_path = _run_file(self.baseline_run, "telemetry.jsonl")

        out: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "baseline_run": str(self.baseline_run),
            "start_node": self.start_node,
            "end_node": self.end_node,
            "ok": True,
            "summary": {},
            "table": [],
            "findings": [],
            "suggestions": [],
        }

        missing = [
            str(p)
            for p in (traj_path, telem_path, baseline_traj_path)
            if not p.exists()
        ]
        if missing:
            out["ok"] = False
            out["findings"].append(
                {
                    "level": "error",
                    "message": "missing required hotspot artifacts",
                    "paths": missing,
                }
            )
            return out

        loop_traj = _load_trajectory(traj_path)
        baseline_traj = _load_trajectory(baseline_traj_path)
        n = min(len(loop_traj), len(baseline_traj))
        components = _trajectory_drift_components(baseline_traj, loop_traj, n)
        loop_evs = load_jsonl(telem_path)
        metric_evs = load_jsonl(baseline_telem_path) if baseline_telem_path.exists() else loop_evs
        telemetry_source = "baseline_run" if baseline_telem_path.exists() else "run"
        metrics = _keyframe_metrics_by_node(metric_evs)
        loop_events = _loop_events_by_node(loop_evs)

        table = self._build_table(components, metrics, loop_events)
        out["table"] = table
        if not table:
            out["ok"] = False
            out["findings"].append(
                {
                    "level": "error",
                    "message": "no trajectory nodes overlap requested hotspot interval",
                }
            )
            return out

        out["summary"] = self._summarize(table, telemetry_source=telemetry_source)
        findings, suggestions = self._findings(table, out["summary"])
        out["findings"].extend(findings)
        out["suggestions"].extend(suggestions)
        return out

    def _build_table(
        self,
        components: dict[str, list[float] | list[int]],
        metrics: dict[int, dict[str, Any]],
        loop_events: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        nodes = components["nodes"]
        trans = components["translation_m"]
        longitudinal = components["longitudinal_m"]
        lateral = components["lateral_m"]
        yaw = components["yaw_rad"]
        rows: list[dict[str, Any]] = []
        for i, node_raw in enumerate(nodes):
            node = int(node_raw)
            if not (self.start_node <= node <= self.end_node):
                continue
            m = metrics.get(node, {})
            ev = loop_events.get(node)
            rows.append(
                {
                    "node": node,
                    "correction_m": float(trans[i]),
                    "longitudinal_m": float(longitudinal[i]),
                    "lateral_m": float(lateral[i]),
                    "yaw_rad": float(yaw[i]),
                    "yaw_deg": math.degrees(float(yaw[i])),
                    "pose_jump": m.get("pose_jump"),
                    "scan_match_score": m.get("scan_match_score"),
                    "prediction_error_m": m.get("prediction_error_m"),
                    "odometry_step_m": m.get("odometry_step_m"),
                    "loop_event": ev or {},
                    "loop_event_label": _loop_event_label(ev),
                }
            )
        return rows

    def _summarize(self, table: list[dict[str, Any]], *, telemetry_source: str) -> dict[str, Any]:
        start = table[0]
        end = table[-1]
        loop_accepted = sum(int(r["loop_event"].get("accepted", 0)) for r in table)
        loop_rejected = sum(int(r["loop_event"].get("rejected", 0)) for r in table)
        loop_candidates = sum(int(r["loop_event"].get("candidate_count", 0)) for r in table)
        correction = [float(r["correction_m"]) for r in table]
        monotonic_steps = sum(
            1 for a, b in zip(correction, correction[1:]) if float(b) >= float(a)
        )
        dominant = "longitudinal"
        if abs(float(end["lateral_m"]) - float(start["lateral_m"])) > abs(
            float(end["longitudinal_m"]) - float(start["longitudinal_m"])
        ):
            dominant = "lateral"
        return {
            "telemetry_source": telemetry_source,
            "node_count": len(table),
            "node_span": {"start": int(start["node"]), "end": int(end["node"])},
            "drift_growth_m": float(end["correction_m"] - start["correction_m"]),
            "start_correction": self._correction_view(start),
            "end_correction": self._correction_view(end),
            "delta": {
                "correction_m": float(end["correction_m"] - start["correction_m"]),
                "longitudinal_m": float(end["longitudinal_m"] - start["longitudinal_m"]),
                "lateral_m": float(end["lateral_m"] - start["lateral_m"]),
                "yaw_rad": float(end["yaw_rad"] - start["yaw_rad"]),
                "yaw_deg": float(end["yaw_deg"] - start["yaw_deg"]),
                "dominant_translation_component": dominant,
            },
            "correction_m": series_summary(correction),
            "pose_jump": _summary_or_empty([r["pose_jump"] for r in table]),
            "scan_match_score": _summary_or_empty([r["scan_match_score"] for r in table]),
            "prediction_error_m": _summary_or_empty([r["prediction_error_m"] for r in table]),
            "odometry_step_m": _summary_or_empty([r["odometry_step_m"] for r in table]),
            "monotonic_growth_fraction": (
                monotonic_steps / max(1, len(correction) - 1)
            ),
            "loop_events": {
                "candidate_count": loop_candidates,
                "accepted": loop_accepted,
                "rejected": loop_rejected,
            },
        }

    @staticmethod
    def _correction_view(row: dict[str, Any]) -> dict[str, float | int]:
        return {
            "node": int(row["node"]),
            "correction_m": float(row["correction_m"]),
            "longitudinal_m": float(row["longitudinal_m"]),
            "lateral_m": float(row["lateral_m"]),
            "yaw_rad": float(row["yaw_rad"]),
            "yaw_deg": float(row["yaw_deg"]),
        }

    def _findings(
        self, table: list[dict[str, Any]], summary: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        t = self.thresholds
        findings: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        growth = float(summary["drift_growth_m"])
        pose_jump = summary["pose_jump"]
        score = summary["scan_match_score"]
        pred = summary["prediction_error_m"]
        loops = summary["loop_events"]
        delta = summary["delta"]

        if growth > 0.25:
            findings.append(
                {
                    "level": "warn",
                    "message": "drift grows inside hotspot interval",
                    "growth_m": growth,
                    "nodes": summary["node_span"],
                }
            )
        if loops["accepted"] == 0 and loops["rejected"] == 0 and loops["candidate_count"] == 0:
            findings.append(
                {
                    "level": "info",
                    "message": "no loop-closure events occur inside hotspot interval",
                }
            )
        else:
            findings.append(
                {
                    "level": "info",
                    "message": "loop-closure events occur inside hotspot interval",
                    **loops,
                }
            )
        if pose_jump.get("max", 0.0) <= t.pose_jump_warn and score.get("min", 0.0) > t.low_score:
            findings.append(
                {
                    "level": "warn",
                    "message": "drift grows while local scan-match metrics look normal",
                    "pose_jump_max": pose_jump.get("max", 0.0),
                    "scan_match_score_min": score.get("min", 0.0),
                    "prediction_error_max_m": pred.get("max", 0.0),
                }
            )
            suggestions.append(
                {
                    "why": "hotspot does not look like a single local matcher jump",
                    "try": [
                        "inspect systematic yaw/translation bias over this interval",
                        "compare local submap window sizes on only this node range",
                        "try stronger yaw refinement or candidate re-ranking",
                    ],
                }
            )
        if abs(float(delta["yaw_deg"])) > 1.0:
            findings.append(
                {
                    "level": "warn",
                    "message": "yaw correction changes across hotspot",
                    "delta_yaw_deg": delta["yaw_deg"],
                }
            )
        if abs(float(delta["longitudinal_m"])) >= abs(float(delta["lateral_m"])):
            findings.append(
                {
                    "level": "info",
                    "message": "hotspot correction is dominated by longitudinal drift",
                    "delta_longitudinal_m": delta["longitudinal_m"],
                    "delta_lateral_m": delta["lateral_m"],
                }
            )
        else:
            findings.append(
                {
                    "level": "info",
                    "message": "hotspot correction is dominated by lateral drift",
                    "delta_longitudinal_m": delta["longitudinal_m"],
                    "delta_lateral_m": delta["lateral_m"],
                }
            )
        return findings, suggestions


def render_hotspot_markdown(rep: dict[str, Any]) -> str:
    summary = rep.get("summary", {})
    loops = summary.get("loop_events", {})
    delta = summary.get("delta", {})
    lines = [
        "# Cloud Hotspot",
        "",
        f"- Run: `{rep.get('run_dir')}`",
        f"- Baseline: `{rep.get('baseline_run')}`",
        f"- Nodes: {rep.get('start_node')} -> {rep.get('end_node')}",
        f"- Telemetry source: `{summary.get('telemetry_source', 'unknown')}`",
        f"- Drift growth: {float(summary.get('drift_growth_m', 0.0)):.3f} m",
        (
            "- Delta local frame: "
            f"longitudinal {float(delta.get('longitudinal_m', 0.0)):.3f} m, "
            f"lateral {float(delta.get('lateral_m', 0.0)):.3f} m, "
            f"yaw {float(delta.get('yaw_deg', 0.0)):.2f} deg"
        ),
        (
            "- Loop events: "
            f"cand {loops.get('candidate_count', 0)}, "
            f"acc {loops.get('accepted', 0)}, rej {loops.get('rejected', 0)}"
        ),
        "",
        "## Findings",
    ]
    for f in rep.get("findings", []):
        lines.append(f"- `{f.get('level')}` {f.get('message')}")
    lines.extend(
        [
            "",
            "## Node Table",
            "",
            (
                "| node | corr_m | long_m | lat_m | yaw_deg | pose_jump | "
                "score | pred_err_m | step_m | loop |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in rep.get("table", []):
        lines.append(
            "| "
            f"{int(r['node'])} | "
            f"{float(r['correction_m']):.3f} | "
            f"{float(r['longitudinal_m']):.3f} | "
            f"{float(r['lateral_m']):.3f} | "
            f"{float(r['yaw_deg']):.2f} | "
            f"{_fmt_optional(r.get('pose_jump'))} | "
            f"{_fmt_optional(r.get('scan_match_score'))} | "
            f"{_fmt_optional(r.get('prediction_error_m'))} | "
            f"{_fmt_optional(r.get('odometry_step_m'))} | "
            f"{r.get('loop_event_label', '-')} |"
        )
    lines.append("")
    if rep.get("suggestions"):
        lines.append("## Suggestions")
        for s in rep.get("suggestions", []):
            tries = "; ".join(str(x) for x in s.get("try", []))
            lines.append(f"- {s.get('why')}: {tries}")
        lines.append("")
    return "\n".join(lines)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def render_markdown(rep: dict[str, Any]) -> str:
    facts = rep.get("facts", {})
    telemetry = facts.get("telemetry", {})
    traj = facts.get("trajectory", {})
    loop = telemetry.get("loop_candidates", {})
    odom = rep.get("inference", {}).get("lidar_odometry", {})
    loop_inf = rep.get("inference", {}).get("loop_closure", {})
    lines = [
        "# CloudAnalyzer",
        "",
        f"- Run: `{rep.get('run_dir')}`",
        f"- Loop closure: **{loop_inf.get('status', 'unknown')}**",
        f"- LiDAR odometry: **{odom.get('status', 'unknown')}**",
        f"- Odometry failure mode: **{odom.get('failure_mode', 'unknown')}**",
        f"- Poses: {traj.get('poses')}",
        f"- Start/end gap: {float(traj.get('start_end_gap_m', 0.0)):.3f} m",
        f"- Loop accepted/rejected: {loop.get('accepted', 0)}/{loop.get('rejected', 0)}",
    ]
    if "baseline_vs_loop" in facts:
        cmp = facts["baseline_vs_loop"]
        growth = cmp.get("drift_growth") or {}
        crossings = growth.get("threshold_crossings") or {}
        decomp = cmp.get("drift_decomposition") or {}
        end_decomp = decomp.get("end") or {}
        growth_100 = growth.get("max_growth_per_100_nodes") or {}
        lines.extend(
            [
                f"- Baseline gap: {float(cmp.get('baseline_start_end_gap_m', 0.0)):.3f} m",
                f"- Looped gap: {float(cmp.get('loop_start_end_gap_m', 0.0)):.3f} m",
                f"- Max correction: {float(cmp.get('max_translation_m', 0.0)):.3f} m",
            ]
        )
        if end_decomp:
            lines.extend(
                [
                    (
                        "- End correction local frame: "
                        f"longitudinal {float(end_decomp.get('longitudinal_m', 0.0)):.3f} m, "
                        f"lateral {float(end_decomp.get('lateral_m', 0.0)):.3f} m, "
                        f"yaw {float(end_decomp.get('yaw_deg', 0.0)):.2f} deg"
                    ),
                    (
                        "- Dominant translation component: "
                        f"{end_decomp.get('dominant_translation_component', 'unknown')}"
                    ),
                ]
            )
        if crossings:
            lines.append(
                "- Drift crossings: "
                + ", ".join(f"{k} at node {v}" for k, v in crossings.items())
            )
        if growth_100:
            lines.append(
                "- Max 100-node drift growth: "
                f"{float(growth_100.get('growth_m', 0.0)):.3f} m "
                f"(nodes {growth_100.get('start_node')}->{growth_100.get('end_node')})"
            )
        hotspots = cmp.get("hotspots") or []
        if hotspots:
            lines.extend(
                [
                    "",
                    "## Drift Hotspots",
                    "",
                    (
                        "| rank | nodes | growth_m | long_m | lat_m | yaw_deg | "
                        "loop | effect | target | mode |"
                    ),
                    "|---:|---|---:|---:|---:|---:|---|---|---|---|",
                ]
            )
            for h in hotspots:
                h_delta = h.get("delta") or {}
                h_loop = h.get("loop_events") or {}
                h_effect = h.get("loop_effect") or {}
                loop_label = (
                    f"cand:{h_loop.get('candidate_count', 0)} "
                    f"acc:{h_loop.get('accepted', 0)} "
                    f"rej:{h_loop.get('rejected', 0)}"
                )
                lines.append(
                    "| "
                    f"{h.get('rank')} | "
                    f"{h.get('start_node')}->{h.get('end_node')} | "
                    f"{float(h.get('growth_m', 0.0)):.3f} | "
                    f"{float(h_delta.get('longitudinal_m', 0.0)):.3f} | "
                    f"{float(h_delta.get('lateral_m', 0.0)):.3f} | "
                    f"{float(h_delta.get('yaw_deg', 0.0)):.2f} | "
                    f"{loop_label} | "
                    f"{h_effect.get('verdict', 'unknown')} | "
                    f"{h.get('debug_target', 'unknown')} | "
                    f"{h.get('failure_mode', 'unknown')} |"
                )
    lines.append("")
    lines.append("## Findings")
    for f in rep.get("findings", []):
        lines.append(f"- `{f.get('level')}` {f.get('message')}")
    lines.append("")
    lines.append("## Suggestions")
    for s in rep.get("suggestions", []):
        tries = "; ".join(str(x) for x in s.get("try", []))
        lines.append(f"- {s.get('why')}: {tries}")
    lines.append("")
    return "\n".join(lines)
