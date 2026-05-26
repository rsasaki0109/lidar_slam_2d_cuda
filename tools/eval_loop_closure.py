"""Quantify loop-closure on the real Cartographer backpack_2d bag.

The P-loop robustness work (edge weights + cauchy kernel) was validated only on a
synthetic single-outlier stress test. This measures it end-to-end on the real bag,
where loops only start to close ~scan 1200 (the backpack returns within loop_dist_m
of an earlier node). Three variants:

  noloop      loop closure off            -- the accumulated dead-reckoning drift
  loop_naive  on, plain L2, edges at 1.0  -- the pre-P-loop behaviour
  loop_robust on, cauchy + inlier weights -- the shipped behaviour

Each estimated trajectory is associated to the Cartographer pseudo-GT by timestamp
and reported as Umeyama-aligned ATE. We also count the accepted loop edges so a
"no improvement" result can be read as "no loops fired" vs "loops fired but didn't
help". Runs are cached per (variant, max_scans).

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/eval_loop_closure.py \
    --max-scans 1300 --variants noloop loop_naive loop_robust
"""
from __future__ import annotations

import argparse
import copy
import math
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from slamx.cli.main import _scan_ba_engine_from_config
from slamx.core.evaluation.ate import Traj2D, build_ate_report, load_gt
from slamx.core.io.bag import iter_scans_bag1, iter_scans_db3


def _variant_cfg(base: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base)
    sb = cfg["slam"].setdefault("scan_ba", {})
    if variant == "noloop":
        sb.update(loop_closure_enabled=False)
    elif variant == "loop_naive":
        sb.update(
            loop_closure_enabled=True,
            loop_robust_loss="linear",
            loop_robust_f_scale=1.0,
            loop_edge_weighting=False,
        )
    elif variant == "loop_robust":
        sb.update(
            loop_closure_enabled=True,
            loop_robust_loss="cauchy",
            loop_robust_f_scale=0.5,
            loop_edge_weighting=True,
        )
    elif variant in ("loop_naive_loose", "loop_robust_loose"):
        # Loosen the detection gates so marginal/wrong matches slip through. With tight
        # gates the robust solve is break-even (the gates already reject outliers); this
        # exposes the regime the robust formulation is actually for -- once false edges
        # get in, the plain L2 solve is dragged off while the kernel + weights clip them.
        sb.update(
            loop_closure_enabled=True,
            loop_accept_inlier_ratio=0.2,
            loop_accept_rms_m=0.7,
            loop_max_correction_m=3.0,
        )
        if variant == "loop_naive_loose":
            sb.update(loop_robust_loss="linear", loop_robust_f_scale=1.0, loop_edge_weighting=False)
        else:
            sb.update(loop_robust_loss="cauchy", loop_robust_f_scale=0.5, loop_edge_weighting=True)
    elif variant == "loop_multiinit_noverify":
        # yaw sweep ON, geometric verification OFF -- isolates the sweep's effect from
        # the verification's. If this matches loop_robust (single init) the sweep changes
        # no accepted edge (revisits are at similar headings), so any regression in
        # loop_multiinit is purely the verification's false rejections.
        sb.update(
            loop_closure_enabled=True,
            loop_robust_loss="cauchy",
            loop_robust_f_scale=0.5,
            loop_edge_weighting=True,
            loop_init_yaw_offsets_rad=(0.0, math.pi / 2, -math.pi / 2, math.pi),
            loop_ambiguity_margin=0.0,
        )
    elif variant in ("loop_multiinit", "loop_robust_widecand", "loop_multiinit_widecand"):
        # P-loop2 detection-side robustness. loop_multiinit = shipped robust + a yaw
        # sweep and geometric verification at the tight gates (a no-regression check on
        # the clean bag). The *_widecand pair loosens CANDIDATE generation (loop_dist_m
        # 2.5->5.0) so the detector proposes wrong revisits a single align mis-registers;
        # comparing single-init robust vs multi-init shows whether the yaw sweep + rival
        # verification hold up once detection is stressed (the detection-side analogue of
        # the loose-gate experiment).
        sb.update(
            loop_closure_enabled=True,
            loop_robust_loss="cauchy",
            loop_robust_f_scale=0.5,
            loop_edge_weighting=True,
        )
        if variant != "loop_robust_widecand":
            sb.update(
                loop_init_yaw_offsets_rad=(0.0, math.pi / 2, -math.pi / 2, math.pi),
                loop_ambiguity_margin=0.3,
                loop_solution_sep_m=0.5,
            )
        if variant != "loop_multiinit":
            sb.update(loop_dist_m=5.0)
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


def _run_variant(base: dict, variant: str, scans: list) -> dict:
    eng = _scan_ba_engine_from_config(_variant_cfg(base, variant), None)
    t0 = time.perf_counter()
    for sc in scans:
        eng.handle_scan(sc)
    dt = time.perf_counter() - t0
    poses = eng.poses
    xy = np.array([[p.x, p.y] for p in poses], dtype=np.float64)
    stamp_ns = [int(s) for s in eng.stamps_ns]
    n_loops = len(getattr(eng, "_loop_edges", ()))
    print(
        f"  [{variant}] {len(scans)} scans in {dt:.1f}s "
        f"({dt / max(1, len(scans)) * 1e3:.0f} ms/scan), loop edges={n_loops}"
    )
    return {"traj": Traj2D(stamp_ns=stamp_ns, xy=xy), "n_loops": n_loops}


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
    ap.add_argument("--max-scans", type=int, default=1300)
    ap.add_argument("--variants", nargs="+", default=["noloop", "loop_naive", "loop_robust"])
    ap.add_argument("--max-dt-ms", type=float, default=50.0)
    ap.add_argument("--cache", type=Path, default=Path("runs/ate_loop_closure_cache.pkl"))
    ap.add_argument("--no-align", action="store_true", help="report ATE without Umeyama alignment")
    args = ap.parse_args()

    cache: dict = {}
    if args.cache.exists():
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gt = load_gt(args.gt)
    print(f"GT: {len(gt.stamp_ns)} poses from {args.gt}")

    scans = None
    results: dict[str, dict] = {}
    n_loops: dict[str, int] = {}
    for variant in args.variants:
        key = f"{variant}@{args.max_scans}"
        if key in cache:
            run = cache[key]
            print(f"  [{variant}] cached (loop edges={run['n_loops']})")
        else:
            if scans is None:
                scans = _load_scans(args.bag, args.topic, args.max_scans)
                print(f"loaded {len(scans)} scans from {args.bag}")
            run = _run_variant(base, variant, scans)
            cache[key] = run
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cache, "wb") as f:
                pickle.dump(cache, f)
        rep = build_ate_report(
            run["traj"], gt, max_dt_ns=int(args.max_dt_ms * 1e6), align=not args.no_align
        )
        results[variant] = rep
        n_loops[variant] = run["n_loops"]

    print("\n=== loop-closure ATE vs Cartographer (Umeyama-aligned)" + ("" if not args.no_align else " [raw]") + " ===")
    width = max(len(v) for v in results)
    for variant, rep in results.items():
        print(f"  {variant:<{width}}  loops={n_loops[variant]:<3d}  {_fmt(rep)}")


if __name__ == "__main__":
    main()
