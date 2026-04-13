from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app
from slamx.cli.main import _engine_from_config
from slamx.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    return load_config(ROOT / "configs" / name)


def test_locked_cartographer_parity_configs_capture_expected_benchmark_values() -> None:
    s300 = _load_config("cartographer_parity_medium_s300_locked.yaml")
    s2k = _load_config("cartographer_parity_noloop_s2k.yaml")

    loop_300 = s300["slam"]["loop_detection"]
    assert loop_300["enabled"] is True
    assert loop_300["search_radius_m"] == 2.5
    assert loop_300["min_separation_nodes"] == 45
    assert loop_300["accept_score"] == -0.28

    loop_2k = s2k["slam"]["loop_detection"]
    assert loop_2k["enabled"] is False
    assert loop_2k["search_radius_m"] == 4.0
    assert loop_2k["min_separation_nodes"] == 45
    assert loop_2k["accept_score"] == -2.0


def test_working_cartographer_parity_configs_expose_loop_icp_overrides() -> None:
    medium = _load_config("cartographer_parity_medium.yaml")
    full = _load_config("cartographer_parity_full.yaml")

    medium_loop_icp = medium["slam"]["loop_detection"]["icp"]
    assert medium_loop_icp["max_iterations"] == 15
    assert medium_loop_icp["max_correspondence_dist_m"] == 1.0
    assert medium_loop_icp["min_correspondences"] == 30
    assert medium_loop_icp["trim_fraction"] == 0.15

    full_loop = full["slam"]["loop_detection"]
    assert full_loop["icp_accept_rms"] == 0.10
    assert full_loop["icp"]["max_iterations"] == 15
    assert full_loop["icp"]["max_correspondence_dist_m"] == 1.0


def test_cartographer_parity_full_config_drops_removed_optimization_window() -> None:
    cfg = _load_config("cartographer_parity_full.yaml")
    assert "optimization_window" not in cfg["slam"]["pose_graph"]


def test_cartographer_parity_configs_still_build_local_slam_engine() -> None:
    for name in (
        "cartographer_parity_medium.yaml",
        "cartographer_parity_medium_s300_locked.yaml",
        "cartographer_parity_noloop_s2k.yaml",
        "cartographer_parity_full.yaml",
    ):
        cfg = _load_config(name)
        engine = _engine_from_config(cfg, telemetry=None)
        assert engine.cfg.pose_graph.max_iterations > 0
        if name in {"cartographer_parity_medium.yaml", "cartographer_parity_full.yaml"}:
            assert engine.cfg.loop_icp.max_iterations == 15
            assert engine.cfg.loop_icp.max_correspondence_dist_m == 1.0


def test_cartographer_parity_configs_replay_fixture_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    fixture = ROOT / "examples" / "fixture_scans.jsonl"

    for name in (
        "cartographer_parity_medium.yaml",
        "cartographer_parity_medium_s300_locked.yaml",
        "cartographer_parity_noloop_s2k.yaml",
        "cartographer_parity_full.yaml",
    ):
        out = tmp_path / name.replace(".yaml", "")
        result = runner.invoke(
            app,
            [
                "replay",
                str(fixture),
                "--config",
                str(ROOT / "configs" / name),
                "--out",
                str(out),
                "--deterministic",
                "--seed",
                "0",
                "--no-write-map",
            ],
        )
        assert result.exit_code == 0, result.stdout + result.stderr
        assert (out / "trajectory.json").exists()


def test_cartographer_benchmark_note_points_at_locked_configs() -> None:
    rep = json.loads(
        (ROOT / "notes" / "benchmark_cartographer_agreement_b0_2014-07-11.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        rep["runs_300_parity_medium"]["config"]
        == "configs/cartographer_parity_medium_s300_locked.yaml"
    )
    assert rep["runs_2000_parity_medium"]["config"] == "configs/cartographer_parity_noloop_s2k.yaml"
