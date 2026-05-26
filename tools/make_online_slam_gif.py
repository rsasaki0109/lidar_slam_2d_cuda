"""Single-panel "online SLAM" showcase GIF on the Cartographer backpack_2d bag.

A cinematic, single-panel replay (cuVSLAM / CudaRobotics-style): the scan-BA engine
runs online with loop closure ON, and we animate the live map building up as the
backpack drives. The accumulated scan points are placed at the *current* pose
estimates, so when a revisit fires a pose-graph constraint the whole map visibly
SNAPS back into alignment. The freshest scan is drawn bright (the "live" beam),
older geometry fades to cyan, the trajectory trails behind, and loop edges flash
green as they close.

Unlike the OFF|ON comparison GIF, this is one big panel meant as the hero image:
run it long enough (~1300 scans) that the map becomes a legible floor plan and the
first big loop (~scan 1200) closes on camera.

The heavy engine run (~1.3 s/scan) is cached to a pickle so re-rendering is cheap.

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/make_online_slam_gif.py \
    --bag data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
    --topic horizontal_laser_2d --config configs/scan_ba_backpack_s300.yaml \
    --max-scans 1300 --cache runs/online_slam_cache.pkl --out docs/assets/online_slam.gif
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


def _loop_cfg(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sb = cfg["slam"].setdefault("scan_ba", {})
    sb.update(
        dict(
            loop_closure_enabled=True,
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
    for k, sc in enumerate(scans):
        eng.handle_scan(sc)
        snaps.append([Pose2(p.x, p.y, p.theta) for p in eng.graph.poses])
        edges.append(sorted(eng._loop_edges))
        if (k + 1) % 50 == 0:
            print(f"  scan {k + 1}/{len(scans)}  loop edges={len(eng._loop_edges)}", flush=True)
    return snaps, edges, eng


def _build_cache(args) -> dict:
    suffix = args.bag.suffix.lower()
    raw = iter_scans_db3(args.bag, topic=args.topic) if suffix == ".db3" else iter_scans_bag1(args.bag, topic=args.topic)
    scans = []
    for k, sc in enumerate(raw):
        if k >= args.max_scans:
            break
        scans.append(sc)
    print(f"loaded {len(scans)} scans", flush=True)

    snaps, edges, eng = _run(_loop_cfg(args.config), scans)
    print(f"loop ON done; loop edges={len(eng._loop_edges)}", flush=True)

    pts = [s.astype(np.float32) for s in eng._scans]

    def packs(s):
        return [np.array([[p.x, p.y, p.theta] for p in snap], dtype=np.float32) for snap in s]

    return {
        "pts": pts,
        "snaps": packs(snaps),
        "edges": edges,
        "n_loop_edges": len(eng._loop_edges),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--max-scans", type=int, default=1300)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--point-stride", type=int, default=3)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--snap-thresh-m", type=float, default=0.08,
                    help="min pose shift between scans to count as a map SNAP")
    ap.add_argument("--snap-hold", type=int, default=25, help="scans the SNAP flash lingers")
    args = ap.parse_args()

    if args.cache.exists():
        print(f"loading cache {args.cache}", flush=True)
        with open(args.cache, "rb") as f:
            data = pickle.load(f)
    else:
        data = _build_cache(args)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump(data, f)
        print(f"wrote cache {args.cache}", flush=True)

    pts = [p[:: args.point_stride] for p in data["pts"]]
    snaps = data["snaps"]
    edges = data.get("edges", [[] for _ in snaps])
    edge_sets = [set(map(tuple, e)) for e in edges]
    n = len(pts)

    # A "SNAP" is a frame where the pose-graph solve shifted already-placed poses by
    # more than snap_thresh -- i.e. the map visibly jumped. Loops fire every few scans
    # but most only nudge the window; flagging real corrections keeps the flash meaningful
    # instead of permanently on. corr[k] = max displacement of poses [0,k-1) vs prev frame.
    corr = np.zeros(n)
    for k in range(1, n):
        a, b = snaps[k], snaps[k - 1]
        m = min(a.shape[0], b.shape[0]) - 1
        if m > 0:
            corr[k] = float(np.hypot(a[:m, 0] - b[:m, 0], a[:m, 1] - b[:m, 1]).max())
    snap_on = np.zeros(n, dtype=bool)
    for k in np.nonzero(corr > args.snap_thresh_m)[0]:
        snap_on[k : min(n, k + args.snap_hold)] = True
    print(f"render n={n}, loop edges={data['n_loop_edges']}, "
          f"snap frames={int(snap_on.sum())} (thresh {args.snap_thresh_m} m)", flush=True)

    def world_pts(k, lo=0, hi=None):
        """All scan points i in [lo,hi) placed at frame-k pose estimates."""
        snap = snaps[k]
        hi = min(snap.shape[0], len(pts)) if hi is None else min(hi, snap.shape[0], len(pts))
        parts = []
        for i in range(lo, hi):
            if pts[i].size:
                x, y, th = float(snap[i, 0]), float(snap[i, 1]), float(snap[i, 2])
                c, s = np.cos(th), np.sin(th)
                T = np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)
                parts.append(transform_points_xy(T, pts[i].astype(np.float64)))
        return np.concatenate(parts) if parts else np.zeros((0, 2))

    # fit the view to the fully-built, loop-corrected map (final frame) with a margin,
    # so the camera is steady and the map fills the frame as it grows into it.
    allp = world_pts(n - 1)
    margin = 1.5
    x0, x1 = allp[:, 0].min() - margin, allp[:, 0].max() + margin
    y0, y1 = allp[:, 1].min() - margin, allp[:, 1].max() + margin
    W, H = x1 - x0, y1 - y0

    panel_h = 6.4
    panel_w = panel_h * (W / H)
    fig, ax = plt.subplots(figsize=(panel_w, panel_h), dpi=150)
    fig.patch.set_facecolor("#0b1220")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    flash = 10  # scans a freshly-fired loop edge stays bright

    frames = list(range(0, n, args.frame_stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    frames += [n - 1] * int(args.fps * 1.4)  # hold on the final corrected map

    def draw(k):
        ax.clear()
        ax.set_facecolor("#0b1220")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        for sp in ax.spines.values():
            sp.set_color("#1e293b")

        # accumulated map (all but the freshest few scans) -- cyan, semi-transparent
        live_lo = max(0, k - 2)
        wp = world_pts(k, 0, live_lo)
        if wp.size:
            ax.scatter(wp[:, 0], wp[:, 1], s=1.3, c="#38bdf8", alpha=0.35, linewidths=0)
        # the live beam: the freshest scans, bright
        lp = world_pts(k, live_lo, k + 1)
        if lp.size:
            ax.scatter(lp[:, 0], lp[:, 1], s=2.6, c="#fde047", alpha=0.95, linewidths=0)

        # trajectory + current pose
        P = snaps[k]
        tx, ty = P[: k + 1, 0], P[: k + 1, 1]
        ax.plot(tx, ty, "-", color="#60a5fa", lw=1.4, alpha=0.9)
        if len(tx):
            ax.plot(tx[-1], ty[-1], "o", color="#fde047", ms=8,
                    markeredgecolor="#fff7cc", markeredgewidth=1.0)

        # loop-closure constraints; flash the ones just fired
        cur = edge_sets[k]
        recent = cur - edge_sets[max(0, k - flash)]
        for i, j in cur:
            if i < P.shape[0] and j < P.shape[0]:
                xs = [float(P[i, 0]), float(P[j, 0])]
                ys = [float(P[i, 1]), float(P[j, 1])]
                if (i, j) in recent:
                    ax.plot(xs, ys, "-", color="#fde047", lw=2.0, alpha=0.95)
                else:
                    ax.plot(xs, ys, "-", color="#34d399", lw=0.7, alpha=0.45)

        ax.text(0.014, 0.975, "slamx · scan-BA online SLAM",
                transform=ax.transAxes, color="#e2e8f0", fontsize=15,
                fontweight="bold", va="top", ha="left")
        ax.text(0.014, 0.93, "Cartographer backpack_2d",
                transform=ax.transAxes, color="#64748b", fontsize=10, va="top", ha="left")
        ax.text(0.986, 0.025, f"scan {k + 1}/{n}   loop closures: {len(cur)}",
                transform=ax.transAxes, color="#34d399", fontsize=12,
                fontweight="bold", va="bottom", ha="right")
        if snap_on[k]:  # the solve just shifted the map -> flash
            ax.text(0.986, 0.975, "SNAP ⚡", transform=ax.transAxes, color="#fde047",
                    fontsize=20, fontweight="bold", va="top", ha="right", alpha=0.95)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
