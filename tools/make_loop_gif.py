"""Loop-closure showcase GIF: pose-graph back-end, loop OFF (left) vs ON (right).

A robot drives once around a square ring corridor and returns to the start. Odometry
(here: scan-matching relative poses) carries a small per-step error, so integrating it
open-loop drifts -- the corridor walls smear and the trajectory never closes (left).
When the robot recognises the start, a loop-closure constraint is added and the global
pose-graph solve redistributes the drift, snapping the whole map shut (right). This is
the CudaRobotics gpu_online_slam effect, driven by slamx's real `PoseGraph` back-end --
the same solver the scan-BA engine calls on a detected loop. Only the odometry drift is
synthetic (modelling accumulated front-end error).

Usage:
  env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/python tools/make_loop_gif.py \
    --out docs/assets/loop_closure.gif
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from slamx.core.backend.pose_graph import Edge, PoseGraph, PoseGraphConfig  # noqa: E402
from slamx.core.types import Pose2, transform_points_xy  # noqa: E402


# ring corridor: outer square 0..12, inner square 3..9 -> a 3 m wide loop corridor
def _ring_segments() -> np.ndarray:
    o0, o1, i0, i1 = 0.0, 12.0, 3.0, 9.0
    outer = [(o0, o0, o1, o0), (o1, o0, o1, o1), (o1, o1, o0, o1), (o0, o1, o0, o0)]
    inner = [(i0, i0, i1, i0), (i1, i0, i1, i1), (i1, i1, i0, i1), (i0, i1, i0, i0)]
    return np.array(outer + inner, dtype=np.float64)


def _raycast(px, py, theta, segs, n_beams, max_range, ang_min) -> np.ndarray:
    inc = 2.0 * math.pi / n_beams
    ang = ang_min + np.arange(n_beams) * inc + theta
    dx, dy = np.cos(ang), np.sin(ang)
    v3x, v3y = -dy, dx
    best = np.full(n_beams, np.inf)
    for ax, ay, bx, by in segs:
        ex, ey = bx - ax, by - ay
        v1x, v1y = px - ax, py - ay
        denom = ex * v3x + ey * v3y
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (ex * v1y - ey * v1x) / denom
            t2 = (v1x * v3x + v1y * v3y) / denom
        hit = (np.abs(denom) > 1e-12) & (t1 > 0.05) & (t2 >= 0.0) & (t2 <= 1.0) & (t1 < best)
        best = np.where(hit, t1, best)
    rng = np.where(best <= max_range, best, np.inf)
    ok = np.isfinite(rng)
    a = ang[ok]
    return np.column_stack((rng[ok] * np.cos(a - theta), rng[ok] * np.sin(a - theta)))


def _loop_path(ds: float = 0.22, rc: float = 1.4) -> list[Pose2]:
    """Closed rounded-square centerline of the ring corridor, ending back at the start.
    Corners are quarter-circle arcs so the heading rotates smoothly. Heading = tangent."""
    bl, br = 1.5, 10.5
    pts: list[tuple[float, float]] = []

    def straight(x0, y0, x1, y1):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(round(seg / ds)))
        for k in range(n):
            t = k / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    def arc(cx, cy, a0, a1):
        n = max(1, int(round(rc * abs(a1 - a0) / ds)))
        for k in range(n):
            a = a0 + (a1 - a0) * k / n
            pts.append((cx + rc * math.cos(a), cy + rc * math.sin(a)))

    h = math.pi / 2.0
    straight(bl + rc, bl, br - rc, bl)
    arc(br - rc, bl + rc, -h, 0.0)
    straight(br, bl + rc, br, br - rc)
    arc(br - rc, br - rc, 0.0, h)
    straight(br - rc, br, bl + rc, br)
    arc(bl + rc, br - rc, h, 2.0 * h)
    straight(bl, br - rc, bl, bl + rc)
    arc(bl + rc, bl + rc, 2.0 * h, 3.0 * h)
    pts.append((bl + rc, bl))  # close back to the start point

    poses: list[Pose2] = []
    for i, (x, y) in enumerate(pts):
        j = min(i + 1, len(pts) - 1)
        th = math.atan2(pts[j][1] - y, pts[j][0] - x) if j != i else poses[-1].theta
        poses.append(Pose2(x, y, th))
    return poses


def _rel(a: Pose2, b: Pose2) -> Pose2:
    return a.inverse().compose(b)


def _drift_odometry(
    gt: list[Pose2], *, bias_deg: float, sigma_xy: float, sigma_deg: float, seed: int
) -> tuple[list[Pose2], list[Pose2]]:
    """Per-step relative poses with a small systematic heading bias + noise (models
    accumulated scan-matching error). Returns (drifted_rels, drifted_integrated_poses)."""
    rng = np.random.default_rng(seed)
    bias = math.radians(bias_deg)
    sth = math.radians(sigma_deg)
    rels: list[Pose2] = []
    drifted = [gt[0]]
    for i in range(len(gt) - 1):
        r = _rel(gt[i], gt[i + 1])
        dr = Pose2(
            r.x + rng.normal(0.0, sigma_xy),
            r.y + rng.normal(0.0, sigma_xy),
            r.theta + bias + rng.normal(0.0, sth),
        )
        rels.append(dr)
        drifted.append(drifted[-1].compose(dr))
    return rels, drifted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-beams", type=int, default=360)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--ds", type=float, default=0.22)
    ap.add_argument("--bias-deg", type=float, default=0.22)
    ap.add_argument("--sigma-xy", type=float, default=0.004)
    ap.add_argument("--sigma-deg", type=float, default=0.12)
    ap.add_argument("--point-stride", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    segs = _ring_segments()
    gt = _loop_path(ds=args.ds)
    n = len(gt)
    # sensor-frame scan at each node (raycast the true ring), downsampled for rendering
    scans = [
        _raycast(p.x, p.y, p.theta, segs, args.n_beams, args.max_range, -math.pi)[:: args.point_stride]
        for p in gt
    ]

    rels, drifted = _drift_odometry(
        gt, bias_deg=args.bias_deg, sigma_xy=args.sigma_xy, sigma_deg=args.sigma_deg, seed=args.seed
    )
    open_gap = math.hypot(drifted[-1].x - gt[0].x, drifted[-1].y - gt[0].y)
    print(f"nodes={n}  open-loop drift gap={open_gap:.2f} m")

    # ON: pose graph with the drifted odometry chain + one loop-closure edge at the revisit.
    graph = PoseGraph(cfg=PoseGraphConfig(max_iterations=80))
    for p in drifted:
        graph.add_pose(p)
    for i, dr in enumerate(rels):
        graph.add_edge(Edge(i=i, j=i + 1, rel=dr))
    # loop constraint: last node coincides with the start (true relative, ~identity)
    graph.add_edge(Edge(i=0, j=n - 1, rel=_rel(gt[0], gt[-1])))
    graph.optimize()
    closed = list(graph.poses)
    closed_gap = math.hypot(closed[-1].x - gt[0].x, closed[-1].y - gt[0].y)
    print(f"after loop closure gap={closed_gap:.2f} m")

    # per-frame estimated trajectories.
    # left: open-loop drift grows; right: same until the last node, then the snap.
    off_seq = [drifted[: k + 1] for k in range(n)]
    on_seq = [drifted[: k + 1] for k in range(n - 1)] + [closed]

    def world_pts(poses, k):
        parts = [transform_points_xy(poses[i].as_se2(), scans[i]) for i in range(min(k + 1, len(poses))) if scans[i].size]
        return np.concatenate(parts) if parts else np.zeros((0, 2))

    cx, cy, r = 6.0, 6.0, 8.0
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
    frames += [n - 1] * args.fps  # hold on the snap

    def draw(k):
        axl.clear()
        axr.clear()
        style(axl, "loop closure OFF (drifts open)", "#f87171")
        style(axr, "loop closure ON (snaps shut)", "#60a5fa")
        for ax, seq, pcol, lcol in (
            (axl, off_seq, "#fca5a5", "#f87171"),
            (axr, on_seq, "#93c5fd", "#60a5fa"),
        ):
            poses = seq[k]
            wp = world_pts(poses, k)
            if wp.size:
                ax.scatter(wp[:, 0], wp[:, 1], s=0.5, c=pcol, alpha=0.35, linewidths=0)
            tx = [p.x for p in poses[: k + 1]]
            ty = [p.y for p in poses[: k + 1]]
            ax.plot(tx, ty, "-", color=lcol, lw=1.5)
            if tx:
                ax.plot(tx[-1], ty[-1], "o", color="#fde047", ms=6)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
