"""Side-by-side demo GIF: dead reckoning (left) vs scan-BA core (right).

Both panels replay the same scan stream. The left integrates scan-to-scan ICP
with no global map (front-end odometry only) and drifts; the right uses the
scan-BA core trajectory (scan-to-local-map BA) and stays consistent.

Usage:
  python tools/make_demo_gif.py \
    --bag data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
    --topic horizontal_laser_2d \
    --core-traj runs/scan_ba_backpack_s300/trajectory.json \
    --config configs/scan_ba_backpack_s300.yaml \
    --out docs/assets/demo.gif --max-scans 300 --frame-stride 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from slamx.core.io.bag import iter_scans_bag1, iter_scans_db3  # noqa: E402
from slamx.core.local_matching.icp import IcpConfig, IcpScanMatcher  # noqa: E402
from slamx.core.preprocess.pipeline import PreprocessConfig, preprocess_scan  # noqa: E402
from slamx.core.types import Pose2, transform_points_xy  # noqa: E402


def _load_preprocess(config_path: Path | None) -> PreprocessConfig:
    if config_path is None:
        return PreprocessConfig()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pp = (cfg.get("slam", {}) or {}).get("preprocess", {}) or {}
    return PreprocessConfig(
        min_range=pp.get("min_range"),
        max_range=pp.get("max_range"),
        stride=int(pp.get("stride", 1)),
        min_angle_deg=pp.get("min_angle_deg"),
        max_angle_deg=pp.get("max_angle_deg"),
    )


def _dead_reckoning(scans: list, pre: PreprocessConfig) -> list[Pose2]:
    """Scan-to-scan ICP integrated with no global map (drifts)."""
    icp = IcpScanMatcher(
        IcpConfig(max_iterations=25, max_correspondence_dist_m=1.0, min_correspondences=20, trim_fraction=0.2)
    )
    poses = [Pose2(0.0, 0.0, 0.0)]
    last_rel = Pose2(0.0, 0.0, 0.0)
    prev_scan = scans[0]
    for k in range(1, len(scans)):
        prev_pose = poses[-1]
        ref_map = transform_points_xy(prev_pose.as_se2(), prev_scan.points_xy())
        prediction = prev_pose.compose(last_rel)  # constant-velocity guess
        mr = icp.match(scan=scans[k], prediction_map=prediction, ref_points_xy_map=ref_map)
        pose = mr.pose_map
        last_rel = prev_pose.inverse().compose(pose)
        poses.append(pose)
        prev_scan = scans[k]
    return poses


def _accumulate_world_points(scans: list, poses: list[Pose2], stride: int) -> list[np.ndarray]:
    out = []
    for sc, p in zip(scans, poses):
        pts = sc.points_xy()
        if pts.size:
            out.append(transform_points_xy(p.as_se2(), pts[::stride]))
        else:
            out.append(np.zeros((0, 2), dtype=np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=Path)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--core-traj", required=True, type=Path)
    ap.add_argument("--config", default=None, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-scans", type=int, default=300)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--point-stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    pre = _load_preprocess(args.config)

    suffix = args.bag.suffix.lower()
    if suffix == ".db3":
        raw = iter_scans_db3(args.bag, topic=args.topic)
    else:
        raw = iter_scans_bag1(args.bag, topic=args.topic)

    scans = []
    for k, sc in enumerate(raw):
        if k >= args.max_scans:
            break
        scans.append(preprocess_scan(sc, pre))
    n = len(scans)
    print(f"loaded {n} scans")

    core_raw = json.loads(args.core_traj.read_text(encoding="utf-8"))
    core_poses = [Pose2(d["x"], d["y"], d["theta"]) for d in core_raw][:n]
    if len(core_poses) < n:
        n = len(core_poses)
        scans = scans[:n]
    print(f"core poses: {len(core_poses)}")

    dr_poses = _dead_reckoning(scans, pre)[:n]
    print("dead reckoning computed")

    dr_pts = _accumulate_world_points(scans, dr_poses, args.point_stride)
    core_pts = _accumulate_world_points(scans, core_poses, args.point_stride)

    def bounds(pts_list, traj):
        allp = np.concatenate([p for p in pts_list if p.size] + [
            np.array([[q.x, q.y] for q in traj])
        ])
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
        r = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.6 + 2.0
        return cx, cy, r

    drcx, drcy, drr = bounds(dr_pts, dr_poses)
    cocx, cocy, cor = bounds(core_pts, core_poses)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("#0b1220")
    for ax in (axl, axr):
        ax.set_facecolor("#0b1220")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axl.set_title("dead reckoning (scan-to-scan odometry)", color="#f87171", fontsize=13)
    axr.set_title("scan-BA core", color="#60a5fa", fontsize=13)

    frames = list(range(0, n, args.frame_stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    def draw(frame_idx):
        axl.clear()
        axr.clear()
        for ax in (axl, axr):
            ax.set_facecolor("#0b1220")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        axl.set_title("dead reckoning (scan-to-scan odometry)", color="#f87171", fontsize=13)
        axr.set_title("scan-BA core", color="#60a5fa", fontsize=13)

        k = frame_idx
        dp = np.concatenate(dr_pts[: k + 1]) if k >= 0 else np.zeros((0, 2))
        cp = np.concatenate(core_pts[: k + 1]) if k >= 0 else np.zeros((0, 2))
        if dp.size:
            axl.scatter(dp[:, 0], dp[:, 1], s=0.6, c="#fca5a5", alpha=0.35, linewidths=0)
        if cp.size:
            axr.scatter(cp[:, 0], cp[:, 1], s=0.6, c="#93c5fd", alpha=0.35, linewidths=0)

        dx = [p.x for p in dr_poses[: k + 1]]
        dy = [p.y for p in dr_poses[: k + 1]]
        cxs = [p.x for p in core_poses[: k + 1]]
        cys = [p.y for p in core_poses[: k + 1]]
        axl.plot(dx, dy, "-", color="#f87171", lw=1.5)
        axr.plot(cxs, cys, "-", color="#60a5fa", lw=1.5)
        if dx:
            axl.plot(dx[-1], dy[-1], "o", color="#fde047", ms=6)
        if cxs:
            axr.plot(cxs[-1], cys[-1], "o", color="#fde047", ms=6)

        axl.set_xlim(drcx - drr, drcx + drr)
        axl.set_ylim(drcy - drr, drcy + drr)
        axr.set_xlim(cocx - cor, cocx + cor)
        axr.set_ylim(cocy - cor, cocy + cor)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
