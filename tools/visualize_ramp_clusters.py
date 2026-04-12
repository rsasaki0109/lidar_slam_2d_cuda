#!/usr/bin/env python3
"""Visualize ramp bad clusters from telemetry.jsonl.

Usage:
    python tools/visualize_ramp_clusters.py runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid

Generates:
    - <run_dir>/diagnostics_score_jump.png  (score & jump over nodes)
    - <run_dir>/diagnostics_trajectory.png  (trajectory with bad nodes highlighted)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_telemetry(run_dir: Path) -> tuple[list[dict], list[dict]]:
    telem = run_dir / "telemetry.jsonl"
    keyframes: list[dict] = []
    candidates: list[dict] = []
    with telem.open("r") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "keyframe":
                keyframes.append(rec)
            elif rec.get("type") == "scan_match_candidates":
                candidates.append(rec)
    return keyframes, candidates


def main(run_dir: str, bad_nodes: list[int] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    rd = Path(run_dir)
    keyframes, candidates = load_telemetry(rd)

    nodes = np.array([kf["node"] for kf in keyframes])
    scores = np.array([kf.get("scan_match_score", 0.0) for kf in keyframes])
    jumps = np.array([kf.get("pose_jump", 0.0) for kf in keyframes])
    xs = np.array([kf["pose"]["x"] for kf in keyframes])
    ys = np.array([kf["pose"]["y"] for kf in keyframes])

    if bad_nodes is None:
        # Auto-detect: nodes with worst 5% scores
        threshold = np.percentile(scores[1:], 5)  # skip node 0
        bad_nodes = [int(n) for n, s in zip(nodes[1:], scores[1:]) if s <= threshold]

    bad_set = set(bad_nodes)
    is_bad = np.array([n in bad_set for n in nodes])

    # --- Plot 1: Score & Jump ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(nodes, scores, "b-", linewidth=0.5, alpha=0.7, label="score")
    if np.any(is_bad):
        ax1.scatter(nodes[is_bad], scores[is_bad], c="red", s=15, zorder=5, label="bad cluster")
    ax1.set_ylabel("scan_match_score")
    ax1.legend(loc="lower left")
    ax1.set_title(f"Score & Jump — {rd.name}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(nodes, jumps, "g-", linewidth=0.5, alpha=0.7, label="jump")
    if np.any(is_bad):
        ax2.scatter(nodes[is_bad], jumps[is_bad], c="red", s=15, zorder=5, label="bad cluster")
    ax2.set_ylabel("pose_jump")
    ax2.set_xlabel("node")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out1 = rd / "diagnostics_score_jump.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved: {out1}")
    plt.close(fig)

    # --- Plot 2: Trajectory ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(xs, ys, "b-", linewidth=0.5, alpha=0.5)
    ax.scatter(xs[~is_bad], ys[~is_bad], c="blue", s=2, alpha=0.3)
    if np.any(is_bad):
        ax.scatter(xs[is_bad], ys[is_bad], c="red", s=20, zorder=5, label="bad cluster")
        for n in bad_nodes:
            idx = np.searchsorted(nodes, n)
            if 0 <= idx < len(nodes):
                ax.annotate(str(n), (xs[idx], ys[idx]), fontsize=6, color="red")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory — {rd.name}")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out2 = rd / "diagnostics_trajectory.png"
    fig.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig)

    # --- Summary stats for bad clusters ---
    print(f"\n=== Bad cluster summary ({len(bad_nodes)} nodes) ===")
    for n in sorted(bad_nodes)[:20]:
        idx = int(np.searchsorted(nodes, n))
        if idx < len(nodes) and nodes[idx] == n:
            print(f"  node {n:5d}: score={scores[idx]:.6f}  jump={jumps[idx]:.4f}  "
                  f"pos=({xs[idx]:.3f}, {ys[idx]:.3f})")
    if len(bad_nodes) > 20:
        print(f"  ... and {len(bad_nodes) - 20} more")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_dir> [bad_node1,bad_node2,...]")
        sys.exit(1)
    run_dir = sys.argv[1]
    bad = None
    if len(sys.argv) >= 3:
        bad = [int(x) for x in sys.argv[2].split(",")]
    main(run_dir, bad)
