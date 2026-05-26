"""Render one GIF that shows many accepted loop-closure constraints.

The input is an already-computed replay run. This script does not synthesize loop
events: it reads `loop_closure_accepted` events from telemetry.jsonl and draws the
actual accepted pose-graph edges over the optimized trajectory.

Example:
  env -u PYTHONPATH .venv/bin/python tools/make_multi_loop_closure_gif.py \
    --run runs/iilabs_elevator_full_vscan_bb_loop_elevator_sparse_accept2_final_cap32_20260527 \
    --baseline-run runs/iilabs_elevator_full_vscan_bb \
    --title "iilabs elevator multi-loop closure" \
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


def _segments(xy: np.ndarray, edges: list[LoopEdge]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    n = len(xy)
    for e in edges:
        if 0 <= e.i < n and 0 <= e.j < n:
            out.append(np.array([xy[e.i], xy[e.j]], dtype=np.float64))
    return out


def _frame_nodes(n: int, event_nodes: list[int], max_frames: int) -> list[int]:
    base_n = max(24, min(max_frames // 3, 70))
    base = set(np.linspace(0, n - 1, base_n, dtype=int).tolist())
    if len(event_nodes) > max_frames:
        idx = np.linspace(0, len(event_nodes) - 1, max_frames, dtype=int)
        events = {event_nodes[int(i)] for i in idx}
    else:
        events = set(event_nodes)
    frames = sorted(base | events | {n - 1})
    return [int(max(0, min(n - 1, k))) for k in frames]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--baseline-run", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="multi-loop closure")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-event-frames", type=int, default=150)
    ap.add_argument("--hold-seconds", type=float, default=1.8)
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    traj = _load_trajectory(args.run / "trajectory.json")
    edges, final_opt = _load_telemetry(args.run / "telemetry.jsonl")
    if not edges:
        raise RuntimeError(f"no loop_closure_accepted events in {args.run / 'telemetry.jsonl'}")

    baseline = None
    if args.baseline_run is not None:
        baseline_path = args.baseline_run / "trajectory.json"
        if baseline_path.exists():
            baseline = _load_trajectory(baseline_path)

    unique_nodes = sorted({e.node for e in edges})
    frames = _frame_nodes(len(traj), unique_nodes, args.max_event_frames)
    hold = max(1, int(round(args.fps * args.hold_seconds)))
    render_frames = frames + [len(traj) - 1] * hold

    all_xy = [traj]
    if baseline is not None:
        all_xy.append(baseline)
    xy = np.vstack(all_xy)
    pad = max(0.8, 0.07 * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))))
    x0, x1 = float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad)
    y0, y1 = float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad)
    span_x = max(1e-6, x1 - x0)
    span_y = max(1e-6, y1 - y0)

    panel_h = 6.2
    panel_w = panel_h * span_x / span_y
    fig, ax = plt.subplots(figsize=(panel_w, panel_h), dpi=args.dpi)
    fig.patch.set_facecolor("#07111f")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    edge_nodes = np.array([e.node for e in edges], dtype=np.int64)
    recent_window = max(20, len(traj) // 75)

    def draw(k: int):
        ax.clear()
        ax.set_facecolor("#07111f")
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#1e293b")

        if baseline is not None:
            ax.plot(
                baseline[:, 0],
                baseline[:, 1],
                color="#fb7185",
                lw=1.0,
                ls=(0, (3, 4)),
                alpha=0.50,
            )
            ax.text(
                0.018,
                0.055,
                "faint red: no-loop baseline",
                transform=ax.transAxes,
                color="#fecdd3",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        ax.plot(traj[:, 0], traj[:, 1], color="#164e63", lw=0.9, alpha=0.55)
        upto = min(k + 1, len(traj))
        ax.plot(traj[:upto, 0], traj[:upto, 1], color="#38bdf8", lw=2.0, alpha=0.96)
        ax.plot(
            traj[upto - 1, 0],
            traj[upto - 1, 1],
            "o",
            ms=6.5,
            color="#fde047",
            markeredgecolor="#fff7cc",
            markeredgewidth=0.9,
        )

        active = [e for e in edges if e.node <= k]
        recent = [] if k == len(traj) - 1 else [e for e in active if k - recent_window <= e.node <= k]
        active_segments = _segments(traj, active)
        if active_segments:
            ax.add_collection(
                LineCollection(active_segments, colors="#22c55e", linewidths=0.65, alpha=0.35)
            )
        recent_segments = _segments(traj, recent)
        if recent_segments:
            ax.add_collection(
                LineCollection(recent_segments, colors="#fde047", linewidths=1.9, alpha=0.92)
            )

        accepted = int(np.count_nonzero(edge_nodes <= k))
        ax.text(
            0.018,
            0.965,
            args.title,
            transform=ax.transAxes,
            color="#e2e8f0",
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.text(
            0.018,
            0.925,
            "actual accepted loop-closure edges from telemetry",
            transform=ax.transAxes,
            color="#94a3b8",
            fontsize=9,
            ha="left",
            va="top",
        )
        ax.text(
            0.982,
            0.035,
            f"node {k + 1}/{len(traj)}  accepted {accepted}/{len(edges)}",
            transform=ax.transAxes,
            color="#bbf7d0",
            fontsize=10.5,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        if k == len(traj) - 1:
            ax.text(
                0.982,
                0.965,
                "final pose graph",
                transform=ax.transAxes,
                color="#fde047",
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="top",
            )
        elif recent:
            ax.text(
                0.982,
                0.965,
                f"+{len(recent)} loop edges",
                transform=ax.transAxes,
                color="#fde047",
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="top",
            )

        if final_opt:
            before = final_opt.get("residual_rms_before")
            after = final_opt.get("residual_rms_after")
            if before is not None and after is not None:
                ax.text(
                    0.982,
                    0.082,
                    f"final RMS {float(before):.5f} -> {float(after):.5f}",
                    transform=ax.transAxes,
                    color="#7dd3fc",
                    fontsize=9.5,
                    ha="right",
                    va="bottom",
                )
        return []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=render_frames, interval=1000 / args.fps, blit=False)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(
        f"wrote {args.out} "
        f"frames={len(render_frames)} loop_edges={len(edges)} unique_nodes={len(unique_nodes)}"
    )


if __name__ == "__main__":
    main()
