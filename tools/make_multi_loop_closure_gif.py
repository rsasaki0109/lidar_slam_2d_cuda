"""Render a one-panel GIF that makes loop closure visibly close a failed odometry loop.

The script uses already-computed replay outputs:

* `--baseline-run`: no-loop trajectory, used as the failed LiDAR odometry trace.
* `--run`: loop-closure trajectory plus telemetry.jsonl with actual
  `loop_closure_accepted` events.

It first draws the no-loop path in red, then snaps/morphs it to the optimized
pose-graph result while revealing the real accepted loop-closure edges. The morph
is a visual before/after transition between two runs; the edges and counts come
directly from telemetry.

Example:
  env -u PYTHONPATH .venv/bin/python tools/make_multi_loop_closure_gif.py \
    --run runs/iilabs_elevator_full_vscan_bb_loop_elevator_sparse_final_cap32_yawtop3_20260528 \
    --baseline-run runs/iilabs_elevator_full_vscan_bb_yawtop3_20260528 \
    --title "iilabs elevator: improved odometry -> multi-loop closure" \
    --out docs/assets/multi_loop_closure.gif
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402


@dataclass(frozen=True)
class LoopEdge:
    node: int
    i: int
    j: int
    score: float


def _load_trajectory(path: Path) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)


def _load_telemetry(path: Path) -> tuple[list[LoopEdge], dict | None]:
    edges: list[LoopEdge] = []
    final_opt: dict | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") == "loop_closure_accepted":
                edges.append(
                    LoopEdge(
                        node=int(ev["node"]),
                        i=int(ev["i"]),
                        j=int(ev["j"]),
                        score=float(ev.get("score", 0.0)),
                    )
                )
            elif ev.get("type") == "optimization" and ev.get("final"):
                final_opt = ev
    return sorted(edges, key=lambda e: (e.node, e.i, e.j)), final_opt


def _fit_bounds(xy: np.ndarray, *, aspect: float, pad_frac: float) -> tuple[float, float, float, float]:
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    span_x = max(1e-6, x1 - x0)
    span_y = max(1e-6, y1 - y0)
    pad = pad_frac * max(span_x, span_y)
    x0 -= pad
    x1 += pad
    y0 -= pad
    y1 += pad

    span_x = x1 - x0
    span_y = y1 - y0
    cur = span_x / span_y
    if cur < aspect:
        extra = (aspect * span_y - span_x) * 0.5
        x0 -= extra
        x1 += extra
    elif cur > aspect:
        extra = (span_x / aspect - span_y) * 0.5
        y0 -= extra
        y1 += extra
    return x0, x1, y0, y1


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _gap_m(xy: np.ndarray) -> float:
    return float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))


def _edge_segments(xy: np.ndarray, edges: list[LoopEdge]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    n = len(xy)
    for e in edges:
        if 0 <= e.i < n and 0 <= e.j < n:
            out.append(np.array([xy[e.i], xy[e.j]], dtype=np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--baseline-run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="failed odometry -> multi-loop closure")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--build-frames", type=int, default=70)
    ap.add_argument("--fail-hold-frames", type=int, default=18)
    ap.add_argument("--snap-frames", type=int, default=48)
    ap.add_argument("--edge-frames", type=int, default=72)
    ap.add_argument("--hold-seconds", type=float, default=1.4)
    ap.add_argument("--dpi", type=int, default=115)
    ap.add_argument("--aspect", type=float, default=1.55)
    args = ap.parse_args()

    loop_traj = _load_trajectory(args.run / "trajectory.json")
    baseline = _load_trajectory(args.baseline_run / "trajectory.json")
    n = min(len(loop_traj), len(baseline))
    if n < 2:
        raise RuntimeError("need at least two poses")
    loop_traj = loop_traj[:n]
    baseline = baseline[:n]

    edges, final_opt = _load_telemetry(args.run / "telemetry.jsonl")
    if not edges:
        raise RuntimeError(f"no loop_closure_accepted events in {args.run / 'telemetry.jsonl'}")

    raw_gap = _gap_m(baseline)
    closed_gap = _gap_m(loop_traj)
    all_xy = np.vstack([baseline, loop_traj])
    x0, x1, y0, y1 = _fit_bounds(all_xy, aspect=args.aspect, pad_frac=0.08)

    fig_h = 7.2
    fig_w = fig_h * args.aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor("#07111f")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    hold_frames = max(1, int(round(args.fps * args.hold_seconds)))
    total = (
        args.build_frames
        + args.fail_hold_frames
        + args.snap_frames
        + args.edge_frames
        + hold_frames
    )

    snap0 = args.build_frames + args.fail_hold_frames
    edge0 = snap0 + args.snap_frames
    hold0 = edge0 + args.edge_frames
    recent_batch = max(5, len(edges) // 14)

    def style_axes() -> None:
        ax.set_facecolor("#07111f")
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#1e293b")

    def draw_title(stage: str) -> None:
        ax.text(
            0.022,
            0.966,
            args.title,
            transform=ax.transAxes,
            color="#e2e8f0",
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.text(
            0.022,
            0.924,
            "red: no-loop LiDAR odometry    cyan: loop-closed pose graph    green/yellow: accepted loop edges",
            transform=ax.transAxes,
            color="#94a3b8",
            fontsize=9.5,
            ha="left",
            va="top",
        )
        ax.text(
            0.978,
            0.966,
            stage,
            transform=ax.transAxes,
            color="#fde047",
            fontsize=16,
            fontweight="bold",
            ha="right",
            va="top",
        )

    def draw_start_end(xy: np.ndarray, color: str, *, gap_label: str) -> None:
        ax.plot(xy[0, 0], xy[0, 1], "o", ms=7, color="#e2e8f0", markeredgewidth=0)
        ax.plot(
            xy[-1, 0],
            xy[-1, 1],
            "o",
            ms=8,
            color=color,
            markeredgecolor="#fff7cc",
            markeredgewidth=0.9,
        )
        ax.plot(
            [xy[0, 0], xy[-1, 0]],
            [xy[0, 1], xy[-1, 1]],
            color=color,
            lw=1.2,
            ls=(0, (3, 3)),
            alpha=0.82,
        )
        ax.text(
            0.978,
            0.076,
            gap_label,
            transform=ax.transAxes,
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    def draw_edges(active_count: int, *, final: bool = False) -> None:
        active_count = max(0, min(len(edges), active_count))
        active = edges[:active_count]
        if not active:
            return
        if final:
            recent = []
        else:
            recent = active[max(0, active_count - recent_batch) :]
        older = active[: max(0, active_count - len(recent))]
        older_segments = _edge_segments(loop_traj, older)
        if older_segments:
            ax.add_collection(
                LineCollection(older_segments, colors="#22c55e", linewidths=0.70, alpha=0.38)
            )
        recent_segments = _edge_segments(loop_traj, recent)
        if recent_segments:
            ax.add_collection(
                LineCollection(recent_segments, colors="#fde047", linewidths=2.0, alpha=0.92)
            )

    def draw(f: int):
        ax.clear()
        style_axes()

        if f < args.build_frames:
            u = f / max(1, args.build_frames - 1)
            upto = max(2, min(n, int(round(1 + u * (n - 1)))))
            ax.plot(baseline[:upto, 0], baseline[:upto, 1], color="#fb7185", lw=2.0, alpha=0.95)
            ax.plot(
                baseline[upto - 1, 0],
                baseline[upto - 1, 1],
                "o",
                ms=6.5,
                color="#fde047",
                markeredgecolor="#fff7cc",
                markeredgewidth=0.9,
            )
            if upto == n:
                draw_start_end(baseline, "#fb7185", gap_label=f"odometry gap {raw_gap:.2f} m")
            draw_title("ODOMETRY DRIFT")
            ax.text(
                0.978,
                0.035,
                f"node {upto}/{n}   loop closures 0/{len(edges)}",
                transform=ax.transAxes,
                color="#fecdd3",
                fontsize=11,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
            return []

        ax.plot(baseline[:, 0], baseline[:, 1], color="#fb7185", lw=1.1, ls=(0, (3, 4)), alpha=0.55)

        if f < snap0:
            shown = baseline
            alpha = 0.0
            active_count = 0
            stage = "ODOMETRY FAILS"
        elif f < edge0:
            alpha = _ease((f - snap0) / max(1, args.snap_frames - 1))
            shown = (1.0 - alpha) * baseline + alpha * loop_traj
            active_count = int(round(alpha * 0.35 * len(edges)))
            stage = "POSE GRAPH SNAP"
        elif f < hold0:
            alpha = 1.0
            shown = loop_traj
            eu = _ease((f - edge0) / max(1, args.edge_frames - 1))
            active_count = int(round((0.35 + 0.65 * eu) * len(edges)))
            stage = "MULTI LOOP CLOSE"
        else:
            alpha = 1.0
            shown = loop_traj
            active_count = len(edges)
            stage = "LOOP CLOSED"

        draw_edges(active_count, final=f >= hold0)
        ax.plot(shown[:, 0], shown[:, 1], color="#38bdf8", lw=2.6, alpha=0.98)
        ax.plot(loop_traj[:, 0], loop_traj[:, 1], color="#7dd3fc", lw=0.9, alpha=0.30)

        gap = _gap_m(shown)
        gap_color = "#fb7185" if alpha < 0.45 else "#fde047" if alpha < 0.95 else "#86efac"
        draw_start_end(shown, gap_color, gap_label=f"loop gap {gap:.3f} m")
        draw_title(stage)

        ax.text(
            0.978,
            0.035,
            f"accepted loop closures {active_count}/{len(edges)}",
            transform=ax.transAxes,
            color="#bbf7d0",
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        ax.text(
            0.022,
            0.045,
            f"raw gap {raw_gap:.2f} m -> closed gap {closed_gap:.3f} m",
            transform=ax.transAxes,
            color="#e2e8f0",
            fontsize=10,
            ha="left",
            va="bottom",
        )
        if final_opt:
            before = final_opt.get("residual_rms_before")
            after = final_opt.get("residual_rms_after")
            if before is not None and after is not None:
                ax.text(
                    0.022,
                    0.078,
                    f"final pose-graph RMS {float(before):.5f} -> {float(after):.5f}",
                    transform=ax.transAxes,
                    color="#7dd3fc",
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )
        return []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=range(total), interval=1000 / args.fps, blit=False)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(
        f"wrote {args.out} frames={total} loop_edges={len(edges)} "
        f"raw_gap={raw_gap:.6f} closed_gap={closed_gap:.6f}"
    )


if __name__ == "__main__":
    main()
