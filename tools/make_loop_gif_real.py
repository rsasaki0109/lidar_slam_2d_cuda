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
    edges: list[list[tuple[int, int]]] = []  # loop edges present after each scan
    for sc in scans:
        eng.handle_scan(sc)
        snaps.append([Pose2(p.x, p.y, p.theta) for p in eng.graph.poses])
        edges.append(sorted(eng._loop_edges))
    return snaps, edges, eng


def _build_cache(args) -> dict:
    suffix = args.bag.suffix.lower()
    raw = iter_scans_db3(args.bag, topic=args.topic) if suffix == ".db3" else iter_scans_bag1(args.bag, topic=args.topic)
    scans = []
    for k, sc in enumerate(raw):
        if k >= args.max_scans:
            break
        scans.append(sc)
    print(f"loaded {len(scans)} scans")

    off_snaps, _, eng_off = _run(_loop_cfg(args.config, loop=False), scans)
    print("loop OFF done")
    on_snaps, on_edges, eng_on = _run(_loop_cfg(args.config, loop=True), scans)
    print(f"loop ON done; loop edges={len(eng_on._loop_edges)}")

    # preprocessed sensor-frame points are identical for both runs
    pts = [s.astype(np.float32) for s in eng_on._scans]

    def packs(snaps):
        return [np.array([[p.x, p.y, p.theta] for p in s], dtype=np.float32) for s in snaps]

    return {
        "pts": pts,
        "off": packs(off_snaps),
        "on": packs(on_snaps),
        "on_edges": on_edges,
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
    ap.add_argument("--point-stride", type=int, default=2)
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
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
    on_edges = data.get("on_edges", [[] for _ in off])
    edge_sets = [set(map(tuple, e)) for e in on_edges]
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

    # tight shared view: fit the union of both runs' full extent (OFF drifts wider) with
    # a small margin, so the map fills the panels instead of floating in dark space.
    allp = np.concatenate([world_pts(on, n - 1), world_pts(off, n - 1)])
    margin = 1.2
    x0, x1 = allp[:, 0].min() - margin, allp[:, 0].max() + margin
    y0 = allp[:, 1].min() - margin
    y1 = allp[:, 1].max() + margin * 2.4  # extra headroom so titles clear the map
    W, H = x1 - x0, y1 - y0

    # size the figure to the map aspect so equal-aspect panels fill it (no letterboxing).
    # side-by-side OFF | ON (landscape, consistent with the demo GIF above it).
    panel_h = 4.6
    panel_w = panel_h * (W / H)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(panel_w * 2 + 0.4, panel_h), dpi=125)
    fig.patch.set_facecolor("#0b1220")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.03)
    flash = max(1, int(round(args.fps * 0.5)))  # frames a freshly-fired edge stays bright

    def style(ax, title, color):
        ax.set_facecolor("#0b1220")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        for sp in ax.spines.values():
            sp.set_color("#1e293b")
        ax.text(0.012, 0.96, title, transform=ax.transAxes, color=color,
                fontsize=13, fontweight="bold", va="top", ha="left")

    frames = list(range(0, n, args.frame_stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    frames += [n - 1] * int(args.fps * 1.2)  # short hold on the final corrected map

    def draw(k):
        axl.clear()
        axr.clear()
        style(axl, "loop closure OFF", "#f87171")
        style(axr, "loop closure ON", "#60a5fa")
        for ax, pack, pcol, lcol in ((axl, off, "#fda4af", "#f87171"), (axr, on, "#7dd3fc", "#38bdf8")):
            wp = world_pts(pack, k)
            if wp.size:
                ax.scatter(wp[:, 0], wp[:, 1], s=1.1, c=pcol, alpha=0.45, linewidths=0)
            P = pack[k]
            tx, ty = P[: k + 1, 0], P[: k + 1, 1]
            ax.plot(tx, ty, "-", color=lcol, lw=1.6, alpha=0.95)
            if len(tx):
                ax.plot(tx[-1], ty[-1], "o", color="#fde047", ms=7,
                        markeredgecolor="#fff7cc", markeredgewidth=0.8)

        # ON panel: draw every loop-closure constraint; flash the ones just fired
        snap = on[k]
        cur = edge_sets[k]
        recent = cur - edge_sets[max(0, k - flash)]
        for i, j in cur:
            if i < snap.shape[0] and j < snap.shape[0]:
                xs = [float(snap[i, 0]), float(snap[j, 0])]
                ys = [float(snap[i, 1]), float(snap[j, 1])]
                if (i, j) in recent:
                    axr.plot(xs, ys, "-", color="#fde047", lw=1.8, alpha=0.95)
                else:
                    axr.plot(xs, ys, "-", color="#34d399", lw=0.6, alpha=0.4)
        axr.text(0.988, 0.035, f"loop closures: {len(cur)}", transform=axr.transAxes,
                 color="#34d399", fontsize=12, fontweight="bold", va="bottom", ha="right")
        if recent:  # a constraint just fired this frame -> flash the snap (corner, off the map)
            axr.text(0.988, 0.96, "SNAP", transform=axr.transAxes, color="#fde047",
                     fontsize=18, fontweight="bold", va="top", ha="right", alpha=0.9)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
