"""Real-data loop-closure GIF on the Google Cartographer backpack_2d bag.

Replays the same scan stream through the scan-BA engine twice -- loop closure OFF
(left) vs ON (right) -- and animates the running map (scan points placed at the
current pose estimates) plus the trajectory. With loop closure ON, revisiting the
start region adds pose-graph constraints that pull accumulated drift back, so the
walls stay crisp; OFF, the map smears as drift grows.

The heavy engine run (~1.3 s/scan) is cached to an .npz so re-rendering is cheap.

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/make_loop_gif_real.py \
    --bag data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
    --topic horizontal_laser_2d --config configs/scan_ba_backpack_s300.yaml \
    --max-scans 300 --cache runs/loop_real_cache.npz --out docs/assets/loop_closure.gif
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from slamx.cli.main import _scan_ba_engine_from_config  # noqa: E402
from slamx.core.io.bag import iter_scans_bag1, iter_scans_db3  # noqa: E402
from slamx.core.types import Pose2, transform_points_xy  # noqa: E402


def _loop_cfg(config_path: Path, *, loop: bool) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sb = cfg["slam"].setdefault("scan_ba", {})
    sb.update(
        dict(
            loop_closure_enabled=loop,
            loop_detect_every_n=5,
            loop_dist_m=2.5,
            loop_min_gap=30,
            loop_max_candidates=3,
            loop_submap_window=10,
            loop_accept_inlier_ratio=0.4,
            loop_accept_rms_m=0.3,
            loop_max_correction_m=1.5,
        )
    )
    return cfg


def _run(cfg: dict, scans: list):
    eng = _scan_ba_engine_from_config(cfg, None)
    snaps: list[list[Pose2]] = []
    for sc in scans:
        eng.handle_scan(sc)
        snaps.append([Pose2(p.x, p.y, p.theta) for p in eng.graph.poses])
    return snaps, eng


def _build_cache(args) -> dict:
    suffix = args.bag.suffix.lower()
    raw = iter_scans_db3(args.bag, topic=args.topic) if suffix == ".db3" else iter_scans_bag1(args.bag, topic=args.topic)
    scans = []
    for k, sc in enumerate(raw):
        if k >= args.max_scans:
            break
        scans.append(sc)
    print(f"loaded {len(scans)} scans")

    off_snaps, eng_off = _run(_loop_cfg(args.config, loop=False), scans)
    print("loop OFF done")
    on_snaps, eng_on = _run(_loop_cfg(args.config, loop=True), scans)
    print(f"loop ON done; loop edges={len(eng_on._loop_edges)}")

    # preprocessed sensor-frame points are identical for both runs
    pts = [s.astype(np.float32) for s in eng_on._scans]

    def packs(snaps):
        return [np.array([[p.x, p.y, p.theta] for p in s], dtype=np.float32) for s in snaps]

    return {
        "pts": pts,
        "off": packs(off_snaps),
        "on": packs(on_snaps),
        "n_loop_edges": len(eng_on._loop_edges),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--max-scans", type=int, default=300)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--point-stride", type=int, default=3)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    if args.cache.exists():
        print(f"loading cache {args.cache}")
        with open(args.cache, "rb") as f:
            data = pickle.load(f)
    else:
        data = _build_cache(args)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump(data, f)
        print(f"wrote cache {args.cache}")

    pts = [p[:: args.point_stride] for p in data["pts"]]
    off, on = data["off"], data["on"]
    n = len(pts)
    print(f"render n={n}, loop edges={data['n_loop_edges']}")

    def world_pts(pack, k):
        snap = pack[k]  # (k+1, 3): all pose estimates as of frame k
        parts = []
        for i in range(min(snap.shape[0], len(pts))):
            if pts[i].size:
                x, y, th = float(snap[i, 0]), float(snap[i, 1]), float(snap[i, 2])
                c, s = np.cos(th), np.sin(th)
                T = np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)
                parts.append(transform_points_xy(T, pts[i].astype(np.float64)))
        return np.concatenate(parts) if parts else np.zeros((0, 2))

    # shared view from the loop-ON final extent
    allp = np.concatenate([world_pts(on, n - 1), world_pts(off, n - 1)])
    cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
    r = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.6 + 2.0

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("#0b1220")

    def style(ax, title, color):
        ax.set_facecolor("#0b1220")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_title(title, color=color, fontsize=13)

    frames = list(range(0, n, args.frame_stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    frames += [n - 1] * args.fps

    def draw(k):
        axl.clear()
        axr.clear()
        style(axl, "loop closure OFF (drifts, walls smear)", "#f87171")
        style(axr, "loop closure ON (drift corrected)", "#60a5fa")
        for ax, pack, pcol, lcol in ((axl, off, "#fca5a5", "#f87171"), (axr, on, "#93c5fd", "#60a5fa")):
            wp = world_pts(pack, k)
            if wp.size:
                ax.scatter(wp[:, 0], wp[:, 1], s=0.5, c=pcol, alpha=0.3, linewidths=0)
            P = pack[k]
            tx, ty = P[: k + 1, 0], P[: k + 1, 1]
            ax.plot(tx, ty, "-", color=lcol, lw=1.3)
            if len(tx):
                ax.plot(tx[-1], ty[-1], "o", color="#fde047", ms=6)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
