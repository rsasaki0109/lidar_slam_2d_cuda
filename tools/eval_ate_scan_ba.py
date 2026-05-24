"""Quantitative ATE of the scan-BA engine vs the Cartographer trajectory.

Replays a LiDAR bag through the scan-BA frontend under one or more variants
(pose-only / joint pose+SDF / CUDA), associates each estimated trajectory to the
Cartographer pseudo-GT by timestamp, and reports the Umeyama-aligned ATE
(rmse / mean / p50 / p90 / max). The estimate frame differs from GT, so alignment
is on by default.

The heavy engine runs (~1.3 s/scan pose-only, slower for joint) are cached per
(variant, max_scans) to a pickle so re-reporting is cheap.

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/eval_ate_scan_ba.py \
    --bag data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
    --topic horizontal_laser_2d --config configs/scan_ba_backpack_s300.yaml \
    --gt runs/cartographer_traj.csv --max-scans 300 \
    --variants pose_only joint --cache runs/ate_scan_ba_cache.pkl
"""
from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from slamx.cli.main import _scan_ba_engine_from_config
from slamx.core.evaluation.ate import Traj2D, build_ate_report, load_gt
from slamx.core.io.bag import iter_scans_bag1, iter_scans_db3


def _variant_cfg(base: dict, variant: str) -> dict:
    """Apply a variant to a copy of the base config's scan_ba block."""
    cfg = copy.deepcopy(base)
    sb = cfg["slam"].setdefault("scan_ba", {})
    if variant == "pose_only":
        sb.update(use_joint=False, use_cuda=False)
    elif variant == "joint":
        sb.update(use_joint=True, use_cuda=False)
    elif variant == "joint_smooth":
        sb.update(use_joint=True, use_cuda=False, joint_sdf_smooth_info=1.0)
    elif variant == "cuda":
        sb.update(use_joint=False, use_cuda=True)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return cfg


def _load_scans(bag: Path, topic: str | None, max_scans: int) -> list:
    suffix = bag.suffix.lower()
    raw = iter_scans_db3(bag, topic=topic) if suffix == ".db3" else iter_scans_bag1(bag, topic=topic)
    scans = []
    for k, sc in enumerate(raw):
        if k >= max_scans:
            break
        scans.append(sc)
    return scans


def _run_variant(base: dict, variant: str, scans: list) -> Traj2D:
    eng = _scan_ba_engine_from_config(_variant_cfg(base, variant), None)
    t0 = time.perf_counter()
    for sc in scans:
        eng.handle_scan(sc)
    dt = time.perf_counter() - t0
    poses = eng.poses
    stamps = eng.stamps_ns
    xy = np.array([[p.x, p.y] for p in poses], dtype=np.float64)
    # drop poses without a usable stamp (bag scans all carry one here)
    stamp_ns = [int(s) for s in stamps]
    print(f"  [{variant}] {len(scans)} scans in {dt:.1f}s ({dt / max(1, len(scans)) * 1e3:.0f} ms/scan)")
    return Traj2D(stamp_ns=stamp_ns, xy=xy)


def _fmt(rep: dict) -> str:
    if not rep.get("ok"):
        return f"n={rep.get('n', 0)} (no matched pairs)"
    return (
        f"rmse={rep['rmse_m']:.3f}  mean={rep['mean_m']:.3f}  "
        f"p50={rep['p50_m']:.3f}  p90={rep['p90_m']:.3f}  max={rep['max_m']:.3f}  n={rep['n']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=Path, default=Path("data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag"))
    ap.add_argument("--topic", default="horizontal_laser_2d")
    ap.add_argument("--config", type=Path, default=Path("configs/scan_ba_backpack_s300.yaml"))
    ap.add_argument("--gt", type=Path, default=Path("runs/cartographer_traj.csv"))
    ap.add_argument("--max-scans", type=int, default=300)
    ap.add_argument("--variants", nargs="+", default=["pose_only", "joint"])
    ap.add_argument("--max-dt-ms", type=float, default=50.0)
    ap.add_argument("--cache", type=Path, default=Path("runs/ate_scan_ba_cache.pkl"))
    ap.add_argument("--no-align", action="store_true", help="report ATE without Umeyama alignment")
    args = ap.parse_args()

    cache: dict = {}
    if args.cache.exists():
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gt = load_gt(args.gt)
    print(f"GT: {len(gt.stamp_ns)} poses from {args.gt}")

    scans = None  # lazy: only load the bag if a variant is uncached
    results: dict[str, dict] = {}
    for variant in args.variants:
        key = f"{variant}@{args.max_scans}"
        if key in cache:
            traj = cache[key]
            print(f"  [{variant}] cached")
        else:
            if scans is None:
                scans = _load_scans(args.bag, args.topic, args.max_scans)
                print(f"loaded {len(scans)} scans from {args.bag}")
            traj = _run_variant(base, variant, scans)
            cache[key] = traj
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cache, "wb") as f:
                pickle.dump(cache, f)
        rep = build_ate_report(
            traj, gt, max_dt_ns=int(args.max_dt_ms * 1e6), align=not args.no_align
        )
        results[variant] = rep

    print("\n=== ATE vs Cartographer (Umeyama-aligned)" + ("" if not args.no_align else " [raw]") + " ===")
    width = max(len(v) for v in results)
    for variant, rep in results.items():
        print(f"  {variant:<{width}}  {_fmt(rep)}")


if __name__ == "__main__":
    main()
