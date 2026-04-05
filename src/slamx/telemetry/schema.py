from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelemetryIssue:
    level: str  # "error" | "warn"
    message: str


REQUIRED_COMMON = {"type", "schema_version"}

REQUIRED_BY_TYPE: dict[str, set[str]] = {
    "keyframe": {"node", "stamp_ns", "pose", "prediction", "scan_match_score", "pose_jump"},
    "scan_match_candidates": {"node", "best_score", "top"},
    "optimization": {"node", "success", "cost"},
    "loop_closure_candidates": {"node", "candidates"},
    "loop_closure_accepted": {"node", "i", "j", "score", "rel_ij"},
    "loop_closure_rejected": {"node", "i", "j", "score"},
}


def validate_event(e: dict[str, Any]) -> list[TelemetryIssue]:
    issues: list[TelemetryIssue] = []
    missing_common = REQUIRED_COMMON - set(e.keys())
    if missing_common:
        issues.append(TelemetryIssue("error", f"missing common keys: {sorted(missing_common)}"))
        return issues

    et = e.get("type")
    if not isinstance(et, str):
        issues.append(TelemetryIssue("error", "type must be string"))
        return issues

    req = REQUIRED_BY_TYPE.get(et)
    if req is None:
        issues.append(TelemetryIssue("warn", f"unknown event type: {et}"))
        return issues

    missing = req - set(e.keys())
    if missing:
        issues.append(TelemetryIssue("error", f"{et}: missing keys: {sorted(missing)}"))
    return issues

