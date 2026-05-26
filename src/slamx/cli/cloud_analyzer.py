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


def _trajectory_compare(
    baseline: list[dict[str, float]],
    looped: list[dict[str, float]],
) -> dict[str, Any]:
    n = min(len(baseline), len(looped))
    if n == 0:
        return {"ok": False, "message": "empty trajectory"}

    trans: list[float] = []
    rot: list[float] = []
    for i in range(n):
        trans.append(
            float(math.hypot(baseline[i]["x"] - looped[i]["x"], baseline[i]["y"] - looped[i]["y"]))
        )
        rot.append(abs(_wrap_pi(baseline[i]["theta"] - looped[i]["theta"])))
    worst_i = int(max(range(n), key=lambda i: trans[i]))
    return {
        "ok": True,
        "compared_poses": n,
        "max_translation_m": trans[worst_i],
        "max_translation_index": worst_i,
        "end_translation_m": trans[-1],
        "translation_m": series_summary(trans),
        "max_rotation_rad": max(rot),
        "baseline_start_end_gap_m": _start_end_gap(baseline[:n]),
        "loop_start_end_gap_m": _start_end_gap(looped[:n]),
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
    ) -> None:
        self.run_dir = run_dir
        self.baseline_run = baseline_run
        self.thresholds = thresholds or CloudAnalyzerThresholds()

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
                baseline_telemetry = self._telemetry_facts(load_jsonl(baseline_telem_path))
                out["facts"]["baseline_telemetry"] = baseline_telemetry
                odom_telemetry = baseline_telemetry
                odom_telemetry_source = "baseline_run"

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
        lines.extend(
            [
                f"- Baseline gap: {float(cmp.get('baseline_start_end_gap_m', 0.0)):.3f} m",
                f"- Looped gap: {float(cmp.get('loop_start_end_gap_m', 0.0)):.3f} m",
                f"- Max correction: {float(cmp.get('max_translation_m', 0.0)):.3f} m",
            ]
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
