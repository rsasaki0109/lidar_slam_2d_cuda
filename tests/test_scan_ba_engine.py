from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from slamx.cli.main import _engine_from_config, app
from slamx.core.scan_ba import ScanBaEngine, ScanBaEngineConfig
from slamx.core.scan_ba.tsdf import Tsdf2DConfig
from slamx.core.types import LaserScan, Pose2


def _l_room_laserscan(pose: Pose2, n_beams: int = 360, range_max: float = 8.0) -> LaserScan:
    """Build a LaserScan by ray-casting the L-room (walls x=5, y=5) from `pose`."""
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / n_beams
    angles = angle_min + np.arange(n_beams) * angle_inc
    cos_a = np.cos(angles + pose.theta)
    sin_a = np.sin(angles + pose.theta)
    t_x = np.where(cos_a > 1e-9, (5.0 - pose.x) / np.where(cos_a != 0, cos_a, 1.0), np.inf)
    t_y = np.where(sin_a > 1e-9, (5.0 - pose.y) / np.where(sin_a != 0, sin_a, 1.0), np.inf)
    hit_x_y = pose.y + t_x * sin_a
    hit_y_x = pose.x + t_y * cos_a
    t_x = np.where((t_x > 0) & (hit_x_y <= 5.0), t_x, np.inf)
    t_y = np.where((t_y > 0) & (hit_y_x <= 5.0), t_y, np.inf)
    t = np.minimum(t_x, t_y)
    ranges = np.where(np.isfinite(t) & (t <= range_max) & (t > 0.05), t, float("inf"))
    return LaserScan(
        stamp_ns=None,
        frame_id="laser",
        angle_min=angle_min,
        angle_max=angle_min + (n_beams - 1) * angle_inc,
        angle_increment=angle_inc,
        ranges=ranges,
        range_min=0.05,
        range_max=range_max,
    )


def _engine() -> ScanBaEngine:
    cfg = ScanBaEngineConfig(
        tsdf=Tsdf2DConfig(
            resolution_m=0.05,
            origin_x_m=-5.0,
            origin_y_m=-5.0,
            size_x_m=14.0,
            size_y_m=14.0,
            truncation_m=0.5,
        ),
        window_size=6,
        seed_scans=3,
        prediction_mode="constant_velocity",
    )
    return ScanBaEngine(cfg=cfg)


def test_engine_tracks_moving_trajectory():
    gt = [
        Pose2(0.0, 0.0, 0.0),
        Pose2(0.2, 0.1, 0.02),
        Pose2(0.4, 0.2, 0.04),
        Pose2(0.6, 0.25, 0.06),
        Pose2(0.8, 0.3, 0.08),
        Pose2(1.0, 0.35, 0.10),
    ]
    eng = _engine()
    out = [eng.handle_scan(_l_room_laserscan(p)) for p in gt]

    # frame is anchored at gt[0] = origin, so recovered poses should track gt
    assert len(out) == len(gt)
    final_err = math.hypot(out[-1].x - gt[-1].x, out[-1].y - gt[-1].y)
    assert final_err < 0.12, f"final drift {final_err:.4f}m, poses={out}"
    assert abs(out[-1].theta - gt[-1].theta) < 0.05
    # trajectory must be monotonically advancing (no blow-up / collapse)
    assert out[-1].x > out[0].x + 0.5


def test_replay_cli_with_scan_ba_frontend(tmp_path: Path):
    # write a synthetic moving trajectory to JSONL
    gt = [Pose2(0.0 + 0.15 * i, 0.05 * i, 0.01 * i) for i in range(8)]
    lines = []
    for i, p in enumerate(gt):
        scan = _l_room_laserscan(p)
        lines.append(
            json.dumps(
                {
                    "stamp_ns": i * 100_000_000,
                    "angle_min": scan.angle_min,
                    "angle_max": scan.angle_max,
                    "angle_increment": scan.angle_increment,
                    "ranges": [None if not math.isfinite(r) else float(r) for r in scan.ranges],
                }
            )
        )
    jsonl = tmp_path / "scans.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cfg = tmp_path / "scan_ba.yaml"
    cfg.write_text(
        "\n".join(
            [
                "slam:",
                "  frontend: scan_ba",
                "  scan_ba:",
                "    window_size: 6",
                "    seed_scans: 3",
                "    tsdf:",
                "      resolution_m: 0.05",
                "      origin_x_m: -5.0",
                "      origin_y_m: -5.0",
                "      size_x_m: 14.0",
                "      size_y_m: 14.0",
                "      truncation_m: 0.5",
                "  prediction:",
                "    mode: constant_velocity",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"

    runner = CliRunner()
    r = runner.invoke(
        app,
        ["replay", str(jsonl), "--config", str(cfg), "--out", str(out), "--no-write-map"],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    assert len(traj) == len(gt)
    # last pose should have advanced in +x
    assert traj[-1]["x"] > traj[0]["x"] + 0.5


def test_engine_dispatch_from_config():
    cfg = {"slam": {"frontend": "scan_ba", "scan_ba": {"window_size": 8}}}
    eng = _engine_from_config(cfg, None)
    assert isinstance(eng, ScanBaEngine)
    assert eng.cfg.window_size == 8


def _joint_engine() -> ScanBaEngine:
    cfg = ScanBaEngineConfig(
        tsdf=Tsdf2DConfig(
            resolution_m=0.05,
            origin_x_m=-5.0,
            origin_y_m=-5.0,
            size_x_m=14.0,
            size_y_m=14.0,
            truncation_m=0.5,
        ),
        window_size=6,
        seed_scans=3,
        prediction_mode="constant_velocity",
        use_joint=True,
        joint_sdf_prior_info=10.0,
    )
    return ScanBaEngine(cfg=cfg)


def test_engine_joint_mode_tracks_trajectory():
    """The joint pose+SDF window solver is a drop-in window backend: it must track
    the same moving trajectory the pose-only engine tracks, without blowing up."""
    gt = [
        Pose2(0.0, 0.0, 0.0),
        Pose2(0.2, 0.1, 0.02),
        Pose2(0.4, 0.2, 0.04),
        Pose2(0.6, 0.25, 0.06),
        Pose2(0.8, 0.3, 0.08),
        Pose2(1.0, 0.35, 0.10),
    ]
    eng = _joint_engine()
    assert eng._joint_active and not eng._cuda_active
    out = [eng.handle_scan(_l_room_laserscan(p)) for p in gt]

    assert len(out) == len(gt)
    final_err = math.hypot(out[-1].x - gt[-1].x, out[-1].y - gt[-1].y)
    assert final_err < 0.15, f"final drift {final_err:.4f}m, poses={out}"
    assert out[-1].x > out[0].x + 0.5


def test_engine_joint_dispatch_from_config():
    cfg = {
        "slam": {
            "frontend": "scan_ba",
            "scan_ba": {"window_size": 8, "use_joint": True, "joint_sdf_smooth_info": 1.5},
        }
    }
    eng = _engine_from_config(cfg, None)
    assert isinstance(eng, ScanBaEngine)
    assert eng.cfg.use_joint is True
    assert eng.cfg.joint_sdf_smooth_info == 1.5
    assert eng._joint_active


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
