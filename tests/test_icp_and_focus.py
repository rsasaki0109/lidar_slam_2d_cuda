from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from slamx.cli.main import app


def test_replay_with_icp_config(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"

    cfg = tmp_path / "icp.yaml"
    cfg.write_text(
        "\n".join(
            [
                "slam:",
                "  preprocess:",
                "    stride: 1",
                "  local_matching:",
                "    type: icp",
                "    icp:",
                "      max_iterations: 5",
                "      max_correspondence_dist_m: 1.0",
                "      min_correspondences: 5",
                "      trim_fraction: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "run"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(out),
            "--deterministic",
            "--seed",
            "0",
            "--no-write-map",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    telem = (out / "telemetry.jsonl").read_text(encoding="utf-8")
    assert "\"scan_match_candidates\"" in telem


def test_diff_focus_worst(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"

    runner = CliRunner()
    for name in ("a", "b"):
        r = runner.invoke(
            app,
            [
                "replay",
                str(fixture),
                "--config",
                str(cfg),
                "--out",
                str(tmp_path / name),
                "--deterministic",
                "--seed",
                "1",
            ],
        )
        assert r.exit_code == 0

    d = runner.invoke(app, ["diff", str(tmp_path / "a"), str(tmp_path / "b"), "--focus", "worst"])
    assert d.exit_code == 0, d.stdout + d.stderr
    rep = json.loads(d.stdout)
    assert rep["focus_detail"] is not None


def test_doctor_includes_suggestions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "fixture_scans.jsonl"
    cfg = root / "configs" / "default.yaml"

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "replay",
            str(fixture),
            "--config",
            str(cfg),
            "--out",
            str(tmp_path / "r"),
            "--deterministic",
            "--seed",
            "0",
        ],
    )
    assert r.exit_code == 0
    d = runner.invoke(app, ["doctor", str(tmp_path / "r")])
    assert d.exit_code == 0
    rep = json.loads(d.stdout)
    assert "suggestions" in rep

