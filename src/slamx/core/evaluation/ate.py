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


def load_xy_csv(path: Path) -> Traj2D:
    """Load trajectory CSV with headers: stamp_ns,x,y"""
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stamp_ns.append(int(r["stamp_ns"]))
            xy.append((float(r["x"]), float(r["y"])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64))


def load_xy_json(path: Path) -> Traj2D:
    """Load trajectory JSON list with keys: stamp_ns,x,y"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    stamp_ns: list[int] = []
    xy: list[tuple[float, float]] = []
    for r in rows:
        stamp_ns.append(int(r["stamp_ns"]))
        xy.append((float(r["x"]), float(r["y"])))
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64))


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
    return Traj2D(stamp_ns=stamp_ns, xy=np.asarray(xy, dtype=np.float64))


def load_gt(path: Path) -> Traj2D:
    if path.suffix.lower() in {".json"}:
        return load_xy_json(path)

    if path.suffix.lower() in {".csv"}:
        return load_xy_csv(path)

    raise ValueError("GT must be .csv or .json with stamp_ns,x,y")


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
    raise ValueError("traj must be a run_dir, .csv, .json, or slamx trajectory.json")


def associate_by_time(slam: Traj2D, gt: Traj2D, *, max_dt_ns: int = 50_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Return matched xy arrays (slam_xy, gt_xy). Uses nearest timestamp within max_dt_ns."""
    if len(slam.stamp_ns) == 0 or len(gt.stamp_ns) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    gt_t = np.asarray(gt.stamp_ns, dtype=np.int64)
    slam_t = np.asarray(slam.stamp_ns, dtype=np.int64)
    gt_xy = gt.xy
    slam_xy = slam.xy

    # gt timestamps are assumed sorted; if not, sort
    order = np.argsort(gt_t)
    gt_t = gt_t[order]
    gt_xy = gt_xy[order]

    matched_s: list[np.ndarray] = []
    matched_g: list[np.ndarray] = []
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
    if not matched_s:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(matched_s), np.vstack(matched_g)


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

