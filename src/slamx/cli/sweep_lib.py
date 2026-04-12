from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from slamx.cli import doctor_lib
from slamx.config import deep_merge
from slamx.core.evaluation.ate import build_ate_report, load_gt, load_slam_trajectory


@dataclass(frozen=True)
class SweepAxis:
    key: str  # dotted key path, e.g. slam.local_matching.correlative_grid.linear_window_m
    values: list[Any]


def load_axes(path: Path) -> list[SweepAxis]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a mapping")

    axes: list[SweepAxis] = []

    if "axes" in data:
        raw_axes = data["axes"]
        if not isinstance(raw_axes, list):
            raise ValueError("'axes' must be a list")
        for a in raw_axes:
            if not isinstance(a, dict) or "key" not in a or "values" not in a:
                raise ValueError("each axis must be {key: ..., values: [...]}")
            vals = a["values"]
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(f"axis values must be a non-empty list: {a}")
            axes.append(SweepAxis(key=str(a["key"]), values=list(vals)))
        return axes

    # shorthand: { "slam....param": [..], "slam....param2": [..] }
    for k, v in data.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"sweep axis must be a non-empty list: {k}")
        axes.append(SweepAxis(key=str(k), values=list(v)))
    return axes


def _nested_override(dotted_key: str, value: Any) -> dict[str, Any]:
    parts = dotted_key.split(".")
    if any(not p for p in parts):
        raise ValueError(f"invalid dotted key: {dotted_key}")
    out: dict[str, Any] = {}
    cur: dict[str, Any] = out
    for p in parts[:-1]:
        nxt: dict[str, Any] = {}
        cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return out


def iter_sweep_configs(base_cfg: dict[str, Any], axes: list[SweepAxis]) -> Iterable[tuple[str, dict[str, Any]]]:
    keys = [a.key for a in axes]
    for combo in itertools.product(*[a.values for a in axes]):
        override: dict[str, Any] = {}
        tag_parts: list[str] = []
        for k, v in zip(keys, combo, strict=False):
            override = deep_merge(override, _nested_override(k, v))
            tag_parts.append(f"{k}={v}")
        tag = ",".join(tag_parts)
        cfg = deep_merge(base_cfg, override)
        yield tag, cfg


def run_id_from(tag: str) -> str:
    h = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:10]
    return h


def compare_to_baseline(baseline_run: Path, run: Path) -> dict[str, Any]:
    ta = baseline_run / "trajectory.json"
    tb = run / "trajectory.json"
    rep = doctor_lib.trajectory_max_se2_delta(ta, tb)
    rep["baseline"] = str(baseline_run)
    rep["run"] = str(run)
    return rep


def compute_ate_for_run(
    run_dir: Path,
    *,
    gt_path: Path,
    max_dt_ns: int,
    align: bool,
    segment_gap_ns: int | None = None,
) -> dict[str, Any]:
    traj = run_dir / "trajectory.json"
    slam = load_slam_trajectory(traj)
    gt = load_gt(gt_path)
    rep = build_ate_report(
        slam,
        gt,
        max_dt_ns=int(max_dt_ns),
        align=align,
        segment_gap_ns=segment_gap_ns,
    )
    rep["gt_path"] = str(gt_path)
    return rep


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
