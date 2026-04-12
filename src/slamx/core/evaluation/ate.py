from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Traj2D:
    stamp_ns: list[int]
    xy: np.ndarray  # (N,2)


@dataclass(frozen=True)
class Assoc2D:
    slam_xy: np.ndarray
    gt_xy: np.ndarray
    slam_indices: np.ndarray
    gt_indices: np.ndarray
    slam_stamp_ns: np.ndarray
    gt_stamp_ns: np.ndarray
    dt_ns: np.ndarray


def _empty_xy() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float64)


def _empty_i64() -> np.ndarray:
    return np.zeros((0,), dtype=np.int64)


def load_xy_csv(path: Path) -> Traj2D:
    """Load trajectory CSV with headers: stamp_ns,x,y"""
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stamp_ns.append(int(r["stamp_ns"]))
            xy.append((float(r["x"]), float(r["y"])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64) if xy else _empty_xy())


def load_xy_json(path: Path) -> Traj2D:
    """Load trajectory JSON list with keys: stamp_ns,x,y"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    for r in rows:
        stamp_ns.append(int(r["stamp_ns"]))
        xy.append((float(r["x"]), float(r["y"])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64) if xy else _empty_xy())


def load_xy_tum(path: Path) -> Traj2D:
    """Load TUM trajectory format: timestamp_sec tx ty tz qx qy qz qw."""
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 8:
                raise ValueError(
                    f"TUM trajectory line {line_no} in {path} must have at least 8 columns"
                )
            stamp_ns.append(int(round(float(cols[0]) * 1_000_000_000.0)))
            xy.append((float(cols[1]), float(cols[2])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64) if xy else _empty_xy())


def load_slam_trajectory(path: Path) -> Traj2D:
    rows = json.loads(path.read_text(encoding="utf-8"))
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    for r in rows:
        s = r.get("stamp_ns")
        if s is None:
            continue
        stamp_ns.append(int(s))
        xy.append((float(r["x"]), float(r["y"])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64) if xy else _empty_xy())


def load_gt(path: Path) -> Traj2D:
    if path.suffix.lower() in {".json"}:
        return load_xy_json(path)

    if path.suffix.lower() in {".csv"}:
        return load_xy_csv(path)

    if path.suffix.lower() in {".tum"}:
        return load_xy_tum(path)

    raise ValueError("GT must be .csv, .json, or .tum")


def load_estimated_trajectory(path: Path) -> Traj2D:
    """Load an estimated trajectory from either slamx trajectory.json or generic csv/json."""
    if path.is_dir():
        p = path / "trajectory.json"
        return load_slam_trajectory(p)
    if path.suffix.lower() == ".json":
        # Heuristic: if list entries have 'i' key, treat as slamx trajectory.json
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "i" in rows[0]:
            return load_slam_trajectory(path)
        return load_xy_json(path)
    if path.suffix.lower() == ".csv":
        return load_xy_csv(path)
    if path.suffix.lower() == ".tum":
        return load_xy_tum(path)
    raise ValueError("traj must be a run_dir, .csv, .json, .tum, or slamx trajectory.json")


def associate_by_time_detailed(slam: Traj2D, gt: Traj2D, *, max_dt_ns: int = 50_000_000) -> Assoc2D:
    """Associate by nearest timestamp within max_dt_ns and retain match metadata."""
    if len(slam.stamp_ns) == 0 or len(gt.stamp_ns) == 0:
        return Assoc2D(
            slam_xy=_empty_xy(),
            gt_xy=_empty_xy(),
            slam_indices=_empty_i64(),
            gt_indices=_empty_i64(),
            slam_stamp_ns=_empty_i64(),
            gt_stamp_ns=_empty_i64(),
            dt_ns=_empty_i64(),
        )

    gt_t = np.asarray(gt.stamp_ns, dtype=np.int64)
    slam_t = np.asarray(slam.stamp_ns, dtype=np.int64)
    gt_xy = gt.xy
    slam_xy = slam.xy

    order = np.argsort(gt_t)
    gt_t = gt_t[order]
    gt_xy = gt_xy[order]

    matched_s: list[np.ndarray] = []
    matched_g: list[np.ndarray] = []
    matched_s_idx: list[int] = []
    matched_g_idx: list[int] = []
    matched_s_stamp: list[int] = []
    matched_g_stamp: list[int] = []
    matched_dt_ns: list[int] = []
    for i, ts in enumerate(slam_t):
        j = int(np.searchsorted(gt_t, ts))
        candidates = []
        if 0 <= j < gt_t.size:
            candidates.append(j)
        if 0 <= j - 1 < gt_t.size:
            candidates.append(j - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(int(gt_t[k] - ts)))
        dt = abs(int(gt_t[best] - ts))
        if dt <= int(max_dt_ns):
            matched_s.append(slam_xy[i])
            matched_g.append(gt_xy[best])
            matched_s_idx.append(i)
            matched_g_idx.append(best)
            matched_s_stamp.append(int(ts))
            matched_g_stamp.append(int(gt_t[best]))
            matched_dt_ns.append(dt)

    if not matched_s:
        return Assoc2D(
            slam_xy=_empty_xy(),
            gt_xy=_empty_xy(),
            slam_indices=_empty_i64(),
            gt_indices=_empty_i64(),
            slam_stamp_ns=_empty_i64(),
            gt_stamp_ns=_empty_i64(),
            dt_ns=_empty_i64(),
        )

    return Assoc2D(
        slam_xy=np.vstack(matched_s),
        gt_xy=np.vstack(matched_g),
        slam_indices=np.asarray(matched_s_idx, dtype=np.int64),
        gt_indices=np.asarray(matched_g_idx, dtype=np.int64),
        slam_stamp_ns=np.asarray(matched_s_stamp, dtype=np.int64),
        gt_stamp_ns=np.asarray(matched_g_stamp, dtype=np.int64),
        dt_ns=np.asarray(matched_dt_ns, dtype=np.int64),
    )


def associate_by_time(slam: Traj2D, gt: Traj2D, *, max_dt_ns: int = 50_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Return matched xy arrays (slam_xy, gt_xy). Uses nearest timestamp within max_dt_ns."""
    assoc = associate_by_time_detailed(slam, gt, max_dt_ns=max_dt_ns)
    return assoc.slam_xy, assoc.gt_xy


def _auto_segment_gap_ns(stamps_sorted: np.ndarray, *, min_gap_ns: int = 0) -> int | None:
    if stamps_sorted.size < 2:
        return int(min_gap_ns) if min_gap_ns > 0 else None
    diffs = np.diff(stamps_sorted)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return int(min_gap_ns) if min_gap_ns > 0 else None
    median_gap_ns = int(np.median(positive_diffs))
    return max(int(min_gap_ns), int(median_gap_ns * 5))


def _time_axis_details(
    stamp_ns: list[int] | np.ndarray,
    *,
    min_gap_ns: int = 0,
    segment_gap_ns: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    stamps_sorted = np.sort(np.asarray(stamp_ns, dtype=np.int64))
    if stamps_sorted.size == 0:
        summary = {
            "points": 0,
            "first_stamp_ns": None,
            "last_stamp_ns": None,
            "span_s": 0.0,
            "median_dt_s": None,
            "p90_dt_s": None,
            "max_dt_s": None,
            "gap_threshold_ns": int(segment_gap_ns) if segment_gap_ns is not None else None,
            "gap_threshold_s": float(int(segment_gap_ns) / 1_000_000_000.0)
            if segment_gap_ns is not None
            else None,
            "segment_count": 0,
            "largest_segment_gap_s": 0.0,
            "gaps": [],
            "segments": [],
        }
        return summary, stamps_sorted, _empty_i64()

    diffs = np.diff(stamps_sorted)
    gap_ns = int(segment_gap_ns) if segment_gap_ns is not None else _auto_segment_gap_ns(
        stamps_sorted, min_gap_ns=min_gap_ns
    )
    gap_indices = (
        np.flatnonzero(diffs > gap_ns)
        if diffs.size > 0 and gap_ns is not None
        else np.zeros((0,), dtype=np.int64)
    )

    point_segment_ids = np.zeros((stamps_sorted.size,), dtype=np.int64)
    segments: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    start = 0
    for seg_idx, gap_idx_val in enumerate(gap_indices.tolist()):
        end = int(gap_idx_val) + 1
        point_segment_ids[start:end] = seg_idx
        seg_start = int(stamps_sorted[start])
        seg_end = int(stamps_sorted[end - 1])
        segments.append(
            {
                "segment_index": int(seg_idx),
                "points": int(end - start),
                "start_stamp_ns": seg_start,
                "end_stamp_ns": seg_end,
                "span_s": float((seg_end - seg_start) / 1_000_000_000.0),
            }
        )
        gaps.append(
            {
                "after_index": int(gap_idx_val),
                "from_stamp_ns": int(stamps_sorted[gap_idx_val]),
                "to_stamp_ns": int(stamps_sorted[gap_idx_val + 1]),
                "gap_s": float(diffs[gap_idx_val] / 1_000_000_000.0),
            }
        )
        start = end

    final_seg_idx = len(segments)
    point_segment_ids[start:] = final_seg_idx
    final_start = int(stamps_sorted[start])
    final_end = int(stamps_sorted[-1])
    segments.append(
        {
            "segment_index": int(final_seg_idx),
            "points": int(stamps_sorted.size - start),
            "start_stamp_ns": final_start,
            "end_stamp_ns": final_end,
            "span_s": float((final_end - final_start) / 1_000_000_000.0),
        }
    )

    summary = {
        "points": int(stamps_sorted.size),
        "first_stamp_ns": int(stamps_sorted[0]),
        "last_stamp_ns": int(stamps_sorted[-1]),
        "span_s": float((stamps_sorted[-1] - stamps_sorted[0]) / 1_000_000_000.0),
        "median_dt_s": float(np.median(diffs) / 1_000_000_000.0) if diffs.size > 0 else None,
        "p90_dt_s": float(np.quantile(diffs, 0.9) / 1_000_000_000.0) if diffs.size > 0 else None,
        "max_dt_s": float(np.max(diffs) / 1_000_000_000.0) if diffs.size > 0 else None,
        "gap_threshold_ns": int(gap_ns) if gap_ns is not None else None,
        "gap_threshold_s": float(gap_ns / 1_000_000_000.0) if gap_ns is not None else None,
        "segment_count": int(len(segments)),
        "largest_segment_gap_s": max((g["gap_s"] for g in gaps), default=0.0),
        "gaps": gaps,
        "segments": segments,
    }
    return summary, stamps_sorted, point_segment_ids


def summarize_timestamp_series(
    stamp_ns: list[int] | np.ndarray,
    *,
    min_gap_ns: int = 0,
    segment_gap_ns: int | None = None,
) -> dict[str, Any]:
    summary, _stamps_sorted, _point_segment_ids = _time_axis_details(
        stamp_ns, min_gap_ns=min_gap_ns, segment_gap_ns=segment_gap_ns
    )
    return summary


def _matched_span_summary(stamps: np.ndarray, total_span_s: float | None) -> dict[str, Any]:
    if stamps.size == 0:
        return {
            "start_stamp_ns": None,
            "end_stamp_ns": None,
            "span_s": 0.0,
            "span_ratio": 0.0 if total_span_s and total_span_s > 0.0 else None,
        }
    start_stamp = int(np.min(stamps))
    end_stamp = int(np.max(stamps))
    span_s = float((end_stamp - start_stamp) / 1_000_000_000.0)
    span_ratio = None
    if total_span_s is not None and total_span_s > 0.0:
        span_ratio = float(span_s / total_span_s)
    return {
        "start_stamp_ns": start_stamp,
        "end_stamp_ns": end_stamp,
        "span_s": span_s,
        "span_ratio": span_ratio,
    }


def build_ate_report(
    slam: Traj2D,
    gt: Traj2D,
    *,
    max_dt_ns: int,
    align: bool,
    segment_gap_ns: int | None = None,
) -> dict[str, Any]:
    assoc = associate_by_time_detailed(slam, gt, max_dt_ns=max_dt_ns)
    rep = compute_ate_rmse(assoc.slam_xy, assoc.gt_xy, align=align)

    traj_axis, _traj_sorted, _traj_segment_ids = _time_axis_details(
        slam.stamp_ns,
        min_gap_ns=int(max_dt_ns) * 2,
        segment_gap_ns=segment_gap_ns,
    )
    gt_axis, _gt_sorted, gt_segment_ids = _time_axis_details(
        gt.stamp_ns,
        min_gap_ns=int(max_dt_ns) * 2,
        segment_gap_ns=segment_gap_ns,
    )

    matched_pairs = int(assoc.slam_indices.size)
    matched_unique_gt_points = int(np.unique(assoc.gt_indices).size) if assoc.gt_indices.size > 0 else 0
    matched_traj = _matched_span_summary(assoc.slam_stamp_ns, float(traj_axis.get("span_s") or 0.0))
    matched_gt = _matched_span_summary(assoc.gt_stamp_ns, float(gt_axis.get("span_s") or 0.0))

    matched_gt_segments: list[dict[str, Any]] = []
    if assoc.gt_indices.size > 0:
        matched_segment_ids = gt_segment_ids[assoc.gt_indices]
        for seg_id in np.unique(matched_segment_ids):
            mask = matched_segment_ids == seg_id
            seg_stamps = assoc.gt_stamp_ns[mask]
            seg_start = int(np.min(seg_stamps))
            seg_end = int(np.max(seg_stamps))
            matched_gt_segments.append(
                {
                    "segment_index": int(seg_id),
                    "matched_pairs": int(np.count_nonzero(mask)),
                    "start_stamp_ns": seg_start,
                    "end_stamp_ns": seg_end,
                    "span_s": float((seg_end - seg_start) / 1_000_000_000.0),
                }
            )

    association = {
        "matched_pairs": matched_pairs,
        "traj_points": int(len(slam.stamp_ns)),
        "gt_points": int(len(gt.stamp_ns)),
        "matched_unique_gt_points": matched_unique_gt_points,
        "matched_traj_point_ratio": (
            float(matched_pairs / len(slam.stamp_ns)) if len(slam.stamp_ns) > 0 else None
        ),
        "matched_gt_point_ratio": (
            float(matched_unique_gt_points / len(gt.stamp_ns)) if len(gt.stamp_ns) > 0 else None
        ),
        "matched_traj_start_stamp_ns": matched_traj["start_stamp_ns"],
        "matched_traj_end_stamp_ns": matched_traj["end_stamp_ns"],
        "matched_traj_span_s": matched_traj["span_s"],
        "matched_traj_span_ratio": matched_traj["span_ratio"],
        "matched_gt_start_stamp_ns": matched_gt["start_stamp_ns"],
        "matched_gt_end_stamp_ns": matched_gt["end_stamp_ns"],
        "matched_gt_span_s": matched_gt["span_s"],
        "matched_gt_span_ratio": matched_gt["span_ratio"],
        "matched_gt_segment_count": int(len(matched_gt_segments)),
        "matched_gt_segments": matched_gt_segments,
        "dt_ms": {
            "p50": float(np.median(assoc.dt_ns) / 1_000_000.0) if assoc.dt_ns.size > 0 else None,
            "p90": float(np.quantile(assoc.dt_ns, 0.9) / 1_000_000.0) if assoc.dt_ns.size > 0 else None,
            "max": float(np.max(assoc.dt_ns) / 1_000_000.0) if assoc.dt_ns.size > 0 else None,
        },
    }

    warnings: list[str] = []
    if int(gt_axis.get("segment_count") or 0) > 1:
        warnings.append(
            "GT contains multiple time segments; unmatched trajectory time ranges may be ignored in ATE."
        )
    matched_traj_point_ratio = association.get("matched_traj_point_ratio")
    if matched_traj_point_ratio is not None and matched_traj_point_ratio < 0.95:
        warnings.append(
            f"Only {matched_pairs}/{len(slam.stamp_ns)} trajectory points matched GT within the association window."
        )
    matched_traj_span_ratio = association.get("matched_traj_span_ratio")
    if matched_traj_span_ratio is not None and matched_traj_span_ratio < 0.95:
        warnings.append(
            f"Matched trajectory span covers only {matched_traj_span_ratio * 100.0:.1f}% of the trajectory time span."
        )
    if int(gt_axis.get("segment_count") or 0) > 1 and int(association["matched_gt_segment_count"]) < int(
        gt_axis.get("segment_count") or 0
    ):
        warnings.append(
            f"Matched pairs cover {association['matched_gt_segment_count']}/{gt_axis['segment_count']} GT segments."
        )

    rep["max_dt_ns"] = int(max_dt_ns)
    rep["trajectory_time_axis"] = traj_axis
    rep["gt_time_axis"] = gt_axis
    rep["association"] = association
    if warnings:
        rep["warnings"] = warnings
    return rep


def umeyama_alignment_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find R,t minimizing ||(R*src+t) - dst|| (no scale). Returns (R(2,2), t(2,))."""
    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = X.T @ Y
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - (R @ mu_s)
    return R, t


def compute_ate_rmse(
    slam_xy: np.ndarray,
    gt_xy: np.ndarray,
    *,
    align: bool = True,
) -> dict[str, Any]:
    if slam_xy.shape != gt_xy.shape or slam_xy.size == 0:
        return {"ok": False, "message": "no matched pairs", "n": int(slam_xy.shape[0])}
    if align:
        R, t = umeyama_alignment_2d(slam_xy, gt_xy)
        slam_xy2 = (R @ slam_xy.T).T + t
    else:
        slam_xy2 = slam_xy
        R = np.eye(2)
        t = np.zeros(2)
    e = gt_xy - slam_xy2
    d = np.sqrt(np.sum(e * e, axis=1))
    rmse = float(np.sqrt(np.mean(d * d)))
    return {
        "ok": True,
        "n": int(d.size),
        "rmse_m": rmse,
        "mean_m": float(np.mean(d)),
        "p50_m": float(np.median(d)),
        "p90_m": float(np.quantile(d, 0.9)),
        "max_m": float(np.max(d)),
        "align": bool(align),
        "R": R.tolist(),
        "t": t.tolist(),
    }
