from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from slamx.cli import doctor_lib
from slamx.cli import bench_lib
from slamx.cli import datasets
from slamx.cli import report_lib
from slamx.cli import sweep_lib
from slamx.config import deep_merge, load_config
from slamx.core.backend.pose_graph import PoseGraphConfig
from slamx.core.frontend.local_slam import LocalSlamConfig, LocalSlamEngine
from slamx.core.io import iter_scans_jsonl, list_laserscan_topics, list_topics_by_suffix
from slamx.core.io.bag import iter_imu_bag, iter_pointcloud2_bag, iter_scans_bag1, iter_scans_db3
from slamx.core.preprocess.virtual_scan import VirtualScanConfig, pointcloud_to_virtual_scan
from slamx.core.map_io import export_trajectory_csv, save_occupancy_yaml, save_pgm, save_trajectory_json
from slamx.core.observability import JsonlTelemetry
from slamx.core.map_representation import OccupancyGridConfig, OccupancyGridMap
from slamx.determinism import DeterminismContext, apply_determinism

app = typer.Typer(no_args_is_help=True, add_completion=False)

eval_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(eval_app, name="eval")

datasets_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(datasets_app, name="datasets")

bench_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(bench_app, name="bench")


def _engine_from_config(cfg: dict[str, Any], telemetry: JsonlTelemetry | None) -> LocalSlamEngine:
    slam = cfg.get("slam", {})
    preprocess = slam.get("preprocess", {})
    lm = slam.get("local_matching", {}) or {}
    matcher_type = str(lm.get("type", "correlative"))
    matcher = lm.get("correlative_grid", {}) or {}
    icp = lm.get("icp", {}) or {}
    hybrid_refinement = lm.get("hybrid_refinement", {}) or {}
    hybrid_fallback = lm.get("hybrid_fallback", {}) or {}
    hybrid_fallback_grid = hybrid_fallback.get("correlative_grid", {}) or {}
    pred = slam.get("prediction", {}) or {}
    submap = slam.get("submap", {})
    opt_every = int(slam.get("optimize_every_n_keyframes", 10))
    loop = slam.get("loop_detection", {}) or {}
    loop_corr = loop.get("correlative_grid", {}) or {}
    loop_icp_cfg = loop.get("icp", {}) or {}
    pg = slam.get("pose_graph", {}) or {}
    cap_raw = pg.get("max_nfev_cap")
    adapt_from = slam.get("optimize_adaptive_from_node")
    adapt_min = slam.get("optimize_min_interval_for_long_runs", 200)
    skip_opt_from = slam.get("pose_graph_skip_optimization_from_node")
    pitch_comp = slam.get("pitch_compensation", {}) or {}
    scan_context_cfg = slam.get("scan_context", {}) or {}

    from slamx.core.preprocess.pipeline import PreprocessConfig
    from slamx.core.local_matching.correlative import CorrelativeGridConfig
    from slamx.core.local_matching.hybrid import HybridFallbackConfig, HybridRefinementConfig
    from slamx.core.local_matching.icp import IcpConfig
    from slamx.core.local_matching.scan_context import ScanContextConfig
    from slamx.core.submap.builder import SubmapBuilderConfig
    from slamx.core.loop_detection.heuristic import HeuristicLoopConfig

    engine_cfg = LocalSlamConfig(
        preprocess=PreprocessConfig(
            min_range=preprocess.get("min_range"),
            max_range=preprocess.get("max_range"),
            stride=int(preprocess.get("stride", 1)),
            min_angle_deg=preprocess.get("min_angle_deg"),
            max_angle_deg=preprocess.get("max_angle_deg"),
            gradient_mask_diff_m=preprocess.get("gradient_mask_diff_m"),
            gradient_mask_max_range=preprocess.get("gradient_mask_max_range"),
            gradient_mask_window=int(preprocess.get("gradient_mask_window", 0)),
            gradient_mask_ratio=preprocess.get("gradient_mask_ratio"),
            pitch_compensation_enabled=bool(pitch_comp.get("enabled", False)),
            pitch_sensor_height_m=float(pitch_comp.get("sensor_height_m", 0.5)),
            pitch_floor_margin=float(pitch_comp.get("floor_margin", 1.5)),
        ),
        matcher_type=matcher_type,
        correlative=CorrelativeGridConfig(
            linear_step_m=float(matcher.get("linear_step_m", 0.05)),
            angular_step_deg=float(matcher.get("angular_step_deg", 2.0)),
            linear_window_m=float(matcher.get("linear_window_m", 0.5)),
            angular_window_deg=float(matcher.get("angular_window_deg", 30.0)),
            sigma_hit_m=float(matcher.get("sigma_hit_m", 0.15)),
            range_weight_mode=str(matcher.get("range_weight_mode", "none")),
            range_weight_min_m=float(matcher.get("range_weight_min_m", 1.0)),
        ),
        icp=IcpConfig(
            max_iterations=int(icp.get("max_iterations", 20)),
            max_correspondence_dist_m=float(icp.get("max_correspondence_dist_m", 0.5)),
            min_correspondences=int(icp.get("min_correspondences", 30)),
            trim_fraction=float(icp.get("trim_fraction", 0.2)),
            range_weight_mode=str(icp.get("range_weight_mode", "none")),
            range_weight_min_m=float(icp.get("range_weight_min_m", 1.0)),
            icp_mode=str(icp.get("icp_mode", "point")),
            normal_k=int(icp.get("normal_k", 5)),
        ),
        hybrid_refinement=HybridRefinementConfig(
            top_k=int(hybrid_refinement.get("top_k", 1)),
            min_linear_dist_m=float(hybrid_refinement.get("min_linear_dist_m", 0.0)),
            min_angular_dist_deg=float(hybrid_refinement.get("min_angular_dist_deg", 0.0)),
        ),
        hybrid_fallback=HybridFallbackConfig(
            enabled=bool(hybrid_fallback.get("enabled", False)),
            trigger_score=float(hybrid_fallback.get("trigger_score", -0.01)),
            min_score_gain=float(hybrid_fallback.get("min_score_gain", 0.0)),
            correlative=CorrelativeGridConfig(
                linear_step_m=float(hybrid_fallback_grid.get("linear_step_m", 0.05)),
                angular_step_deg=float(hybrid_fallback_grid.get("angular_step_deg", 2.0)),
                linear_window_m=float(hybrid_fallback_grid.get("linear_window_m", 0.20)),
                angular_window_deg=float(hybrid_fallback_grid.get("angular_window_deg", 15.0)),
                sigma_hit_m=float(hybrid_fallback_grid.get("sigma_hit_m", 0.10)),
            ),
        ),
        prediction_mode=str(pred.get("mode", "hold")),
        prediction_gain=float(pred.get("gain", 1.0)),
        submap=SubmapBuilderConfig(
            max_submap_scans=int(submap.get("max_submap_scans", 30)),
            downsample_stride=int(submap.get("downsample_stride", 2)),
        ),
        optimize_every_n_keyframes=opt_every,
        pose_graph=PoseGraphConfig(
            max_iterations=int(pg.get("max_iterations", 50)),
            max_nfev_cap=int(cap_raw) if cap_raw is not None else None,
        ),
        optimize_adaptive_from_node=int(adapt_from) if adapt_from is not None else None,
        optimize_min_interval_for_long_runs=int(adapt_min),
        pose_graph_skip_optimization_from_node=(
            int(skip_opt_from) if skip_opt_from is not None else None
        ),
        loop=HeuristicLoopConfig(
            enabled=bool(loop.get("enabled", False)),
            search_radius_m=float(loop.get("search_radius_m", 1.5)),
            min_separation_nodes=int(loop.get("min_separation_nodes", 30)),
            max_candidates=int(loop.get("max_candidates", 3)),
            accept_score=float(loop.get("accept_score", -0.25)),
            icp_accept_rms=float(loop.get("icp_accept_rms", 0.15)),
        ),
        loop_icp=IcpConfig(
            max_iterations=int(loop_icp_cfg.get("max_iterations", 30)),
            max_correspondence_dist_m=float(loop_icp_cfg.get("max_correspondence_dist_m", 2.0)),
            min_correspondences=int(loop_icp_cfg.get("min_correspondences", 20)),
            trim_fraction=float(loop_icp_cfg.get("trim_fraction", 0.3)),
            icp_mode=str(loop_icp_cfg.get("icp_mode", "point")),
            normal_k=int(loop_icp_cfg.get("normal_k", 5)),
        ),
        loop_correlative=CorrelativeGridConfig(
            linear_step_m=float(loop_corr.get("linear_step_m", 0.05)),
            angular_step_deg=float(loop_corr.get("angular_step_deg", 2.0)),
            linear_window_m=float(loop_corr.get("linear_window_m", 0.5)),
            angular_window_deg=float(loop_corr.get("angular_window_deg", 30.0)),
            sigma_hit_m=float(loop_corr.get("sigma_hit_m", 0.25)),
        ),
        scan_context=ScanContextConfig(
            n_sectors=int(scan_context_cfg.get("n_sectors", 60)),
            max_range=float(scan_context_cfg.get("max_range", 10.0)),
        ),
        scan_context_enabled=bool(scan_context_cfg.get("enabled", False)),
        scan_context_trigger_score=float(scan_context_cfg.get("trigger_score", -0.03)),
        scan_context_top_k=int(scan_context_cfg.get("top_k", 3)),
        scan_context_min_separation_nodes=int(scan_context_cfg.get("min_separation_nodes", 50)),
    )
    return LocalSlamEngine(cfg=engine_cfg, telemetry=telemetry)


@app.command()
def replay(
    input_path: Annotated[Path, typer.Argument(help="JSONL scans or .db3/.bag bag")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="YAML config"),
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Run output directory")] = Path(
        "runs/latest"
    ),
    deterministic: Annotated[bool, typer.Option(help="Fixed seeds & single-thread hints")] = False,
    seed: Annotated[int, typer.Option(help="RNG / numpy seed when deterministic")] = 0,
    topic: Annotated[
        str | None, typer.Option("--topic", help="LaserScan topic name for .db3 input")
    ] = None,
    write_map: Annotated[bool, typer.Option(help="Export occupancy grid map at end")] = True,
    max_scans: Annotated[
        int | None, typer.Option("--max-scans", help="Stop after N scans (for quick bench)")
    ] = None,
    pointcloud_topic: Annotated[
        str | None,
        typer.Option("--pointcloud-topic", help="PointCloud2 topic; enables virtual 2D scan mode"),
    ] = None,
    imu_topic: Annotated[
        str | None,
        typer.Option("--imu-topic", help="IMU topic for pitch compensation"),
    ] = None,
) -> None:
    """Re-run SLAM offline (JSONL or ROS bags)."""
    apply_determinism(DeterminismContext(enabled=deterministic, seed=seed))
    base_cfg = {
        "slam": {
            "preprocess": {},
            "local_matching": {"correlative_grid": {}},
            "submap": {},
            "optimize_every_n_keyframes": 10,
        }
    }
    user_cfg = load_config(config)
    cfg = deep_merge(base_cfg, user_cfg)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    telem = JsonlTelemetry(out / "telemetry.jsonl")
    eng = _engine_from_config(cfg, telem)

    # Pre-read IMU samples if --imu-topic is specified
    if imu_topic is not None:
        if input_path.suffix.lower() not in {".db3", ".bag"}:
            raise typer.BadParameter("--imu-topic requires a .db3 or .bag input")
        imu_samples = list(iter_imu_bag(input_path, topic=imu_topic))
        eng.set_imu_buffer(imu_samples)
        typer.echo(f"Loaded {len(imu_samples)} IMU samples from {imu_topic}")

    final_optimize_at_end = bool(cfg.get("slam", {}).get("final_optimize_at_end", False))
    ogm: OccupancyGridMap | None = None
    if write_map:
        mc = cfg.get("map", {}).get("occupancy_grid", {})
        ogm = OccupancyGridMap(
            OccupancyGridConfig(
                resolution_m=float(mc.get("resolution_m", 0.05)),
                size_x_m=float(mc.get("size_x_m", 40.0)),
                size_y_m=float(mc.get("size_y_m", 40.0)),
                origin_x_m=float(mc.get("origin_x_m", -20.0)),
                origin_y_m=float(mc.get("origin_y_m", -20.0)),
            )
        )

    # Build virtual scan config from YAML if present
    vs_cfg_dict = cfg.get("slam", {}).get("virtual_scan", {}) or {}
    vs_cfg = VirtualScanConfig(
        z_min=float(vs_cfg_dict.get("z_min", -0.1)),
        z_max=float(vs_cfg_dict.get("z_max", 0.3)),
        angle_min_deg=float(vs_cfg_dict.get("angle_min_deg", -180.0)),
        angle_max_deg=float(vs_cfg_dict.get("angle_max_deg", 180.0)),
        angular_resolution_deg=float(vs_cfg_dict.get("angular_resolution_deg", 0.5)),
        max_range=float(vs_cfg_dict.get("max_range", 30.0)),
    )

    use_pointcloud = pointcloud_topic is not None
    if use_pointcloud:
        if input_path.suffix.lower() not in {".db3", ".bag"}:
            raise typer.BadParameter("--pointcloud-topic requires a .db3 or .bag input")
        clouds = iter_pointcloud2_bag(input_path, topic=pointcloud_topic)
        scans = (pointcloud_to_virtual_scan(c, vs_cfg) for c in clouds)
    elif input_path.suffix.lower() in {".jsonl"}:
        scans = iter_scans_jsonl(input_path)
    elif input_path.suffix.lower() in {".db3"}:
        scans = iter_scans_db3(input_path, topic=topic)
    elif input_path.suffix.lower() in {".bag"}:
        scans = iter_scans_bag1(input_path, topic=topic)
    else:
        raise typer.BadParameter(f"Unsupported input: {input_path}")

    for k, scan in enumerate(scans):
        if max_scans is not None and k >= int(max_scans):
            break
        pose = eng.handle_scan(scan)
        if ogm is not None:
            # map update uses the same preprocessed scan used by frontend
            # (Phase 1: reuse raw scan; acceptable for now)
            ogm.update(pose_map=pose, scan=scan)

    if final_optimize_at_end and len(eng.graph.poses) > 1:
        opt = eng.graph.optimize()
        telem.emit("optimization", {"node": len(eng.graph.poses) - 1, **opt})

    save_trajectory_json(out / "trajectory.json", eng.graph.poses, eng.stamps_ns)
    if ogm is not None:
        img = ogm.to_8bit_occupancy()
        pgm = out / "map.pgm"
        save_pgm(pgm, img)
        save_occupancy_yaml(
            out / "map.yaml",
            image=str(pgm.name),
            resolution=ogm.res,
            origin=[ogm.origin_x, ogm.origin_y, 0.0],
            extra={"slamx": {"map_type": "occupancy_grid", "schema_version": 1}},
        )
    typer.echo(f"Wrote {out / 'trajectory.json'} and telemetry.")


@app.command()
def diff(
    run_a: Annotated[Path, typer.Argument(help="First run directory")],
    run_b: Annotated[Path, typer.Argument(help="Second run directory")],
    focus: Annotated[str, typer.Option(help="none|worst")] = "none",
    window: Annotated[int, typer.Option(help="Event window around focus node")] = 10,
) -> None:
    """Compare two runs (trajectory + key telemetry series)."""
    ta = run_a / "trajectory.json"
    tb = run_b / "trajectory.json"
    if not ta.exists() or not tb.exists():
        typer.echo("Both runs must contain trajectory.json", err=True)
        raise typer.Exit(1)
    traj = doctor_lib.trajectory_max_se2_delta(ta, tb)

    telem_a = run_a / "telemetry.jsonl"
    telem_b = run_b / "telemetry.jsonl"
    telem: dict[str, Any] | None = None
    focus_detail: dict[str, Any] | None = None
    if telem_a.exists() and telem_b.exists():
        sa = doctor_lib.telemetry_keyframe_series(telem_a)
        sb = doctor_lib.telemetry_keyframe_series(telem_b)
        telem = {
            "run_a": {k: doctor_lib.series_summary(v) for k, v in sa.items()},
            "run_b": {k: doctor_lib.series_summary(v) for k, v in sb.items()},
            "delta": {
                "scan_match_score": doctor_lib.series_diff(
                    sa.get("scan_match_score", []), sb.get("scan_match_score", [])
                ),
                "pose_jump": doctor_lib.series_diff(
                    sa.get("pose_jump", []), sb.get("pose_jump", [])
                ),
            },
        }
        if focus.lower() == "worst":
            # pick worst index based on scan_match_score by default
            wi = int(telem["delta"]["scan_match_score"].get("worst_index", 0))
            nodes = sa.get("node", [])
            if 0 <= wi < len(nodes):
                node_id = int(nodes[wi])
                n0 = node_id - int(window)
                n1 = node_id + int(window)
                focus_detail = {
                    "focus": {"metric": "scan_match_score", "worst_index": wi, "node": node_id},
                    "run_a_events": doctor_lib.extract_events_by_node(telem_a, n0, n1),
                    "run_b_events": doctor_lib.extract_events_by_node(telem_b, n0, n1),
                }

    rep = {"trajectory": traj, "telemetry": telem, "focus_detail": focus_detail}
    typer.echo(json.dumps(rep, indent=2, ensure_ascii=False))
    if not traj.get("ok"):
        raise typer.Exit(1)


@app.command("view-map")
def view_map(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing map.pgm/map.yaml")],
    width: Annotated[int, typer.Option(help="ASCII preview width")] = 80,
) -> None:
    """Quickly inspect exported occupancy map (no GUI)."""
    pgm = run_dir / "map.pgm"
    if not pgm.exists():
        raise typer.BadParameter(f"missing {pgm}")

    data = pgm.read_bytes()
    if not data.startswith(b"P5"):
        raise typer.BadParameter("only binary PGM (P5) supported")
    header, rest = data.split(b"\n", 1)
    # Parse: P5\nW H\n255\n...
    parts = data.split(b"\n", 3)
    if len(parts) < 4:
        raise typer.BadParameter("invalid PGM header")
    wh = parts[1].strip().split()
    if len(wh) != 2:
        raise typer.BadParameter("invalid PGM width/height")
    w, h = int(wh[0]), int(wh[1])
    maxv = int(parts[2].strip())
    if maxv != 255:
        raise typer.BadParameter("unsupported PGM maxval")
    pix = parts[3]
    if len(pix) < w * h:
        raise typer.BadParameter("truncated PGM payload")

    # ASCII preview (downsample)
    step = max(1, w // max(1, width))
    out_lines: list[str] = []
    ramp = " .:-=+*#%@"
    for y in range(0, h, step):
        row = pix[y * w : (y + 1) * w : step]
        line = []
        for b in row:
            # 0=occupied(dark) .. 254=free(light)
            idx = int((b / 255.0) * (len(ramp) - 1))
            line.append(ramp[idx])
        out_lines.append("".join(line))

    typer.echo(json.dumps({"pgm": str(pgm), "width": w, "height": h, "step": step}, ensure_ascii=False))
    typer.echo("\n".join(out_lines))


@app.command()
def sweep(
    axes_yaml: Annotated[Path, typer.Argument(help="YAML defining sweep axes")],
    input_path: Annotated[Path, typer.Argument(help="JSONL scans or .db3 bag")],
    base_config: Annotated[
        Path | None,
        typer.Option("--base-config", "-c", help="Base YAML config"),
    ] = None,
    out_root: Annotated[
        Path, typer.Option("--out-root", "-o", help="Output root directory")
    ] = Path("runs/sweep"),
    topic: Annotated[
        str | None, typer.Option("--topic", help="LaserScan topic name for .db3 input")
    ] = None,
    deterministic: Annotated[bool, typer.Option(help="Force deterministic mode")] = True,
    seed: Annotated[int, typer.Option(help="Seed used for all runs (deterministic)")] = 0,
    gt: Annotated[
        Path | None,
        typer.Option("--gt", help="Optional GT for ATE (csv/json/tum)"),
    ] = None,
    max_dt_ms: Annotated[int, typer.Option(help="Max timestamp association gap (ms)")] = 50,
    segment_gap_ms: Annotated[
        int | None,
        typer.Option(help="Optional time gap threshold used to split GT/traj into segments (ms)"),
    ] = None,
    no_align: Annotated[bool, typer.Option(help="Disable SE2 alignment in ATE")] = False,
    write_reports: Annotated[
        bool, typer.Option("--write-reports", help="Write notes-style markdown per run under out_root/notes/")
    ] = False,
) -> None:
    """Run a parameter sweep and compare each run to baseline."""
    apply_determinism(DeterminismContext(enabled=deterministic, seed=seed))

    base_cfg = {
        "slam": {
            "preprocess": {},
            "local_matching": {"correlative_grid": {}},
            "submap": {},
            "optimize_every_n_keyframes": 10,
        }
    }
    user_base = load_config(base_config)
    base_cfg = deep_merge(base_cfg, user_base)

    axes = sweep_lib.load_axes(axes_yaml)

    out_root.mkdir(parents=True, exist_ok=True)
    sweep_meta = {
        "axes_yaml": str(axes_yaml),
        "input_path": str(input_path),
        "base_config": str(base_config) if base_config else None,
        "axes": [{"key": a.key, "values": a.values} for a in axes],
        "deterministic": deterministic,
        "seed": seed,
        "gt": str(gt) if gt else None,
        "max_dt_ms": int(max_dt_ms),
        "segment_gap_ms": int(segment_gap_ms) if segment_gap_ms is not None else None,
        "align": bool(not no_align),
        "write_reports": bool(write_reports),
    }
    sweep_lib.save_json(out_root / "sweep_meta.json", sweep_meta)

    def iter_scans():
        if input_path.suffix.lower() in {".jsonl"}:
            return iter_scans_jsonl(input_path)
        if input_path.suffix.lower() in {".db3"}:
            return iter_scans_db3(input_path, topic=topic)
        if input_path.suffix.lower() in {".bag"}:
            return iter_scans_bag1(input_path, topic=topic)
        raise typer.BadParameter(f"Unsupported input: {input_path}")

    # Baseline
    baseline_dir = out_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with (baseline_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)
    telem = JsonlTelemetry(baseline_dir / "telemetry.jsonl")
    eng = _engine_from_config(base_cfg, telem)
    for scan in iter_scans():
        eng.handle_scan(scan)
    save_trajectory_json(baseline_dir / "trajectory.json", eng.graph.poses, eng.stamps_ns)
    baseline_ate: dict[str, Any] | None = None
    if gt is not None:
        baseline_ate = sweep_lib.compute_ate_for_run(
            baseline_dir,
            gt_path=gt,
            max_dt_ns=int(max_dt_ms) * 1_000_000,
            align=not no_align,
            segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
        )
    if write_reports:
        notes_dir = out_root / "notes"
        rep = report_lib.build_report(
            run_dir=baseline_dir,
            gt_path=gt,
            max_dt_ns=int(max_dt_ms) * 1_000_000,
            align=not no_align,
            segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
        )
        md = report_lib.render_markdown(rep)
        report_lib.write_notes_markdown(notes_dir / "baseline.md", md)

    results: list[dict[str, Any]] = []
    for tag, cfg in sweep_lib.iter_sweep_configs(base_cfg, axes):
        rid = sweep_lib.run_id_from(tag)
        run_dir = out_root / f"run_{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "tag.txt").open("w", encoding="utf-8") as f:
            f.write(tag + "\n")
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        telem = JsonlTelemetry(run_dir / "telemetry.jsonl")
        eng = _engine_from_config(cfg, telem)
        for scan in iter_scans():
            eng.handle_scan(scan)
        save_trajectory_json(run_dir / "trajectory.json", eng.graph.poses, eng.stamps_ns)

        rep = sweep_lib.compare_to_baseline(baseline_dir, run_dir)
        if gt is not None:
            rep["ate"] = sweep_lib.compute_ate_for_run(
                run_dir,
                gt_path=gt,
                max_dt_ns=int(max_dt_ms) * 1_000_000,
                align=not no_align,
                segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
            )
        if write_reports:
            notes_dir = out_root / "notes"
            md = report_lib.render_markdown(
                report_lib.build_report(
                    run_dir=run_dir,
                    gt_path=gt,
                    max_dt_ns=int(max_dt_ms) * 1_000_000,
                    align=not no_align,
                    segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
                )
            )
            report_lib.write_notes_markdown(notes_dir / f"run_{rid}.md", md)
        rep["tag"] = tag
        rep["run_id"] = rid
        results.append(rep)

    if gt is not None:
        results.sort(key=lambda r: float((r.get("ate") or {}).get("rmse_m", 1e9)))
    else:
        results.sort(
            key=lambda r: (
                float(r.get("max_translation_m", 1e9)),
                float(r.get("max_rotation_rad", 1e9)),
            )
        )
    sweep_lib.save_json(out_root / "results.json", results)
    typer.echo(
        json.dumps(
            {"baseline": str(baseline_dir), "baseline_ate": baseline_ate, "results": results},
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("loop-tune")
def loop_tune(
    input_path: Annotated[Path, typer.Argument(help="JSONL scans or .db3 bag")],
    gt: Annotated[Path, typer.Option("--gt", help="GT CSV/JSON/TUM trajectory")],
    base_config: Annotated[
        Path | None,
        typer.Option("--base-config", "-c", help="Base YAML config"),
    ] = None,
    out_root: Annotated[
        Path, typer.Option("--out-root", "-o", help="Output root directory")
    ] = Path("runs/loop_tune"),
    topic: Annotated[
        str | None, typer.Option("--topic", help="LaserScan topic name for .db3 input")
    ] = None,
    matcher: Annotated[str, typer.Option(help="correlative|icp")] = "icp",
    search_radius: Annotated[
        str, typer.Option(help="Comma-separated radii meters, e.g. 1.0,1.5,2.0")
    ] = "1.0,1.5,2.0",
    accept_score: Annotated[
        str, typer.Option(help="Comma-separated thresholds, e.g. -0.6,-0.4,-0.25")
    ] = "-0.6,-0.4,-0.25",
    min_separation_nodes: Annotated[int, typer.Option(help="Min node separation")] = 30,
    max_dt_ms: Annotated[int, typer.Option(help="Max timestamp association gap (ms)")] = 50,
    segment_gap_ms: Annotated[
        int | None,
        typer.Option(help="Optional time gap threshold used to split GT/traj into segments (ms)"),
    ] = None,
    no_align: Annotated[bool, typer.Option(help="Disable SE2 alignment in ATE")] = False,
    write_reports: Annotated[bool, typer.Option(help="Write notes-style markdown per run")] = True,
    deterministic: Annotated[bool, typer.Option(help="Force deterministic mode")] = True,
    seed: Annotated[int, typer.Option(help="Seed used for all runs")] = 0,
) -> None:
    """Targeted tuning for loop closure knobs, ranked by ATE."""
    axes = [
        sweep_lib.SweepAxis(key="slam.loop_detection.enabled", values=[True]),
        sweep_lib.SweepAxis(key="slam.local_matching.type", values=[matcher]),
        sweep_lib.SweepAxis(
            key="slam.loop_detection.search_radius_m",
            values=[float(x) for x in search_radius.split(",") if x.strip()],
        ),
        sweep_lib.SweepAxis(
            key="slam.loop_detection.accept_score",
            values=[float(x) for x in accept_score.split(",") if x.strip()],
        ),
        sweep_lib.SweepAxis(key="slam.loop_detection.min_separation_nodes", values=[int(min_separation_nodes)]),
    ]

    # Reuse sweep() logic by writing a temporary axes file would be noisy; do minimal inline sweep.
    apply_determinism(DeterminismContext(enabled=deterministic, seed=seed))
    base_cfg = {
        "slam": {
            "preprocess": {},
            "local_matching": {"correlative_grid": {}},
            "submap": {},
            "optimize_every_n_keyframes": 10,
        }
    }
    user_base = load_config(base_config)
    base_cfg = deep_merge(base_cfg, user_base)

    out_root.mkdir(parents=True, exist_ok=True)
    sweep_lib.save_json(
        out_root / "tune_meta.json",
        {
            "input_path": str(input_path),
            "gt": str(gt),
            "axes": [{"key": a.key, "values": a.values} for a in axes],
            "segment_gap_ms": int(segment_gap_ms) if segment_gap_ms is not None else None,
            "deterministic": deterministic,
            "seed": seed,
        },
    )

    def iter_scans():
        if input_path.suffix.lower() in {".jsonl"}:
            return iter_scans_jsonl(input_path)
        if input_path.suffix.lower() in {".db3"}:
            return iter_scans_db3(input_path, topic=topic)
        if input_path.suffix.lower() in {".bag"}:
            return iter_scans_bag1(input_path, topic=topic)
        raise typer.BadParameter(f"Unsupported input: {input_path}")

    baseline_dir = out_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with (baseline_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False, allow_unicode=True)
    telem = JsonlTelemetry(baseline_dir / "telemetry.jsonl")
    eng = _engine_from_config(base_cfg, telem)
    for scan in iter_scans():
        eng.handle_scan(scan)
    save_trajectory_json(baseline_dir / "trajectory.json", eng.graph.poses, eng.stamps_ns)
    baseline_ate = sweep_lib.compute_ate_for_run(
        baseline_dir,
        gt_path=gt,
        max_dt_ns=int(max_dt_ms) * 1_000_000,
        align=not no_align,
        segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
    )
    if write_reports:
        notes_dir = out_root / "notes"
        report_lib.write_notes_markdown(
            notes_dir / "baseline.md",
            report_lib.render_markdown(
                report_lib.build_report(
                    run_dir=baseline_dir,
                    gt_path=gt,
                    max_dt_ns=int(max_dt_ms) * 1_000_000,
                    align=not no_align,
                    segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
                )
            ),
        )

    results: list[dict[str, Any]] = []
    for tag, cfg in sweep_lib.iter_sweep_configs(base_cfg, axes):
        rid = sweep_lib.run_id_from(tag)
        run_dir = out_root / f"run_{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tag.txt").write_text(tag + "\n", encoding="utf-8")
        with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        telem = JsonlTelemetry(run_dir / "telemetry.jsonl")
        eng = _engine_from_config(cfg, telem)
        for scan in iter_scans():
            eng.handle_scan(scan)
        save_trajectory_json(run_dir / "trajectory.json", eng.graph.poses, eng.stamps_ns)

        rep = sweep_lib.compare_to_baseline(baseline_dir, run_dir)
        rep["ate"] = sweep_lib.compute_ate_for_run(
            run_dir,
            gt_path=gt,
            max_dt_ns=int(max_dt_ms) * 1_000_000,
            align=not no_align,
            segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
        )
        rep["tag"] = tag
        rep["run_id"] = rid
        results.append(rep)

        if write_reports:
            notes_dir = out_root / "notes"
            report_lib.write_notes_markdown(
                notes_dir / f"run_{rid}.md",
                report_lib.render_markdown(
                    report_lib.build_report(
                        run_dir=run_dir,
                        gt_path=gt,
                        max_dt_ns=int(max_dt_ms) * 1_000_000,
                        align=not no_align,
                        segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
                    )
                ),
            )

    results.sort(key=lambda r: float((r.get("ate") or {}).get("rmse_m", 1e9)))
    sweep_lib.save_json(out_root / "results.json", results)

    leaderboard = {
        "baseline": str(baseline_dir),
        "baseline_ate": baseline_ate,
        "top": results[:10],
        "count": len(results),
    }
    sweep_lib.save_json(out_root / "leaderboard.json", leaderboard)
    typer.echo(json.dumps(leaderboard, indent=2, ensure_ascii=False))


@app.command("bag-info")
def bag_info(
    bag_path: Annotated[Path, typer.Argument(help="ROS bag2 .db3 or ROS1 .bag")],
) -> None:
    """List LaserScan topics inside a bag (for selecting --topic)."""
    try:
        topics = list_laserscan_topics(bag_path)
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    typer.echo(json.dumps({"bag": str(bag_path), "laserscan_topics": topics}, indent=2, ensure_ascii=False))


@app.command("export-tf-trajectory")
def export_tf_trajectory(
    bag_path: Annotated[Path, typer.Argument(help="ROS bag2 .db3 or ROS1 .bag")],
    parent: Annotated[str, typer.Option(help="Parent frame (e.g. map)")] = "map",
    child: Annotated[str, typer.Option(help="Child frame (e.g. base_link)")] = "base_link",
    tf_topic: Annotated[str, typer.Option(help="TF topic")] = "/tf",
    tf_static_topic: Annotated[str, typer.Option(help="Static TF topic")] = "/tf_static",
    static_from_bag: Annotated[
        Path | None, typer.Option(help="Optional bag to source /tf_static from")
    ] = None,
    out_csv: Annotated[Path, typer.Option("--out", "-o", help="Output CSV stamp_ns,x,y")] = Path(
        "runs/cartographer_traj.csv"
    ),
) -> None:
    """Extract 2D (x,y) trajectory from TF inside a bag (for Cartographer comparison)."""
    try:
        from slamx.core.tf import TFBuffer2D, quat_to_yaw
        from slamx.core.types import Pose2
    except Exception as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    try:
        # ensure rosbags installed
        import rosbags  # noqa: F401
    except ImportError as e:
        typer.echo("TF export requires: pip install 'slamx[rosbag]'", err=True)
        raise typer.Exit(2) from e

    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        typer.echo("TF export requires: pip install 'slamx[rosbag]'", err=True)
        raise typer.Exit(2) from e

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    buf = TFBuffer2D()
    rows = []

    def _header_stamp_ns(hdr: object) -> int | None:
        if hdr is None:
            return None
        st = getattr(hdr, "stamp", None)
        if st is None:
            return None
        sec = int(getattr(st, "sec", 0))
        nsec = int(getattr(st, "nanosec", getattr(st, "nsec", 0)))
        return sec * 1_000_000_000 + nsec

    with AnyReader([bag_path]) as reader:
        conns = list(getattr(reader, "connections"))
        tf_conns = [c for c in conns if c.topic == tf_topic and str(c.msgtype).endswith("TFMessage")]
        if not tf_conns:
            avail = [t for t, _mt in list_topics_by_suffix(bag_path, msgtype_suffix="TFMessage")]
            raise typer.BadParameter(f"TF topic not found: {tf_topic}. Available: {avail}")
        static_bag_path = static_from_bag or bag_path
        with AnyReader([static_bag_path]) as static_reader:
            static_conns = [
                c
                for c in list(getattr(static_reader, "connections"))
                if c.topic == tf_static_topic and str(c.msgtype).endswith("TFMessage")
            ]

            # Preload static transforms so parent->child chains that depend on /tf_static work
            # from the first dynamic /tf message onward.
            for use_static in static_conns:
                for _conn, _ts, raw in static_reader.messages(connections=[use_static]):
                    msg = static_reader.deserialize(raw, use_static.msgtype)
                    for tr in getattr(msg, "transforms", []):
                        hdr = getattr(tr, "header", None)
                        parent_frame = str(getattr(hdr, "frame_id", "")).lstrip("/") if hdr else ""
                        child_frame = str(getattr(tr, "child_frame_id", "")).lstrip("/")
                        t = getattr(tr, "transform", None)
                        if t is None:
                            continue
                        trans = getattr(t, "translation", None)
                        rot = getattr(t, "rotation", None)
                        if trans is None or rot is None:
                            continue
                        yaw = quat_to_yaw(float(rot.x), float(rot.y), float(rot.z), float(rot.w))
                        buf.set_transform(
                            parent_frame,
                            child_frame,
                            Pose2(float(trans.x), float(trans.y), float(yaw)),
                        )

        use = tf_conns[0]
        for _conn, _ts, raw in reader.messages(connections=[use]):
            msg = reader.deserialize(raw, use.msgtype)
            # tf2_msgs/TFMessage has .transforms list
            batch_stamp: int | None = None
            for tr in getattr(msg, "transforms", []):
                hdr = getattr(tr, "header", None)
                sns = _header_stamp_ns(hdr)
                if sns is not None and (batch_stamp is None or sns > batch_stamp):
                    batch_stamp = sns
                parent_frame = str(getattr(hdr, "frame_id", "")).lstrip("/") if hdr else ""
                child_frame = str(getattr(tr, "child_frame_id", "")).lstrip("/")
                t = getattr(tr, "transform", None)
                if t is None:
                    continue
                trans = getattr(t, "translation", None)
                rot = getattr(t, "rotation", None)
                if trans is None or rot is None:
                    continue
                yaw = quat_to_yaw(float(rot.x), float(rot.y), float(rot.z), float(rot.w))
                buf.set_transform(
                    parent_frame,
                    child_frame,
                    Pose2(float(trans.x), float(trans.y), float(yaw)),
                )

            T = buf.lookup(parent.lstrip("/"), child.lstrip("/"))
            if T is not None and batch_stamp is not None:
                rows.append((batch_stamp, float(T.x), float(T.y)))

    # write CSV
    import csv

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp_ns", "x", "y"])
        w.writerows(rows)

    typer.echo(json.dumps({"out": str(out_csv), "rows": len(rows)}, ensure_ascii=False))


@datasets_app.command("download-cartographer-backpack2d")
def download_cartographer_backpack2d(
    bag_name: Annotated[str, typer.Argument(help="One of Cartographer public backpack_2d .bag names")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Download directory")] = Path("data/cartographer_backpack2d"),
) -> None:
    """Download a Cartographer public 2D backpack ROS1 bag."""
    spec = datasets.cartographer_backpack2d_bag(bag_name)
    path = datasets.download(spec, out_dir)
    typer.echo(json.dumps({"downloaded": str(path), "url": spec.url}, ensure_ascii=False))


@bench_app.command("cartographer-backpack2d")
def bench_cartographer_backpack2d(
    bag_name: Annotated[str, typer.Argument(help="Cartographer public backpack_2d .bag name")],
    scan_topic: Annotated[str | None, typer.Option("--topic", help="LaserScan topic inside bag")] = None,
    out_root: Annotated[Path, typer.Option("--out-root", "-o", help="Bench output root")] = Path("runs/bench"),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="slamx YAML config")] = Path("configs/default.yaml"),
    deterministic: Annotated[bool, typer.Option(help="Deterministic run")] = True,
    seed: Annotated[int, typer.Option(help="Seed")] = 0,
    carto_recorded_bag: Annotated[
        Path | None,
        typer.Option(help="Bag recorded while running Cartographer (must include /tf)"),
    ] = None,
    tf_parent: Annotated[str, typer.Option(help="TF parent frame")] = "map",
    tf_child: Annotated[str, typer.Option(help="TF child frame")] = "base_link",
    tf_topic: Annotated[str, typer.Option(help="TF topic")] = "/tf",
    max_dt_ms: Annotated[int, typer.Option(help="Association window (ms)")] = 50,
    no_align: Annotated[bool, typer.Option(help="Disable alignment in agreement metric")] = False,
    max_scans: Annotated[int, typer.Option(help="Limit scans for quick bench")] = 800,
) -> None:
    """Benchmark slamx on Cartographer public backpack_2d bag and optionally compare against Cartographer TF."""
    data_dir = Path("data/cartographer_backpack2d")
    bag_path = bench_lib.ensure_cartographer_bag_downloaded(bag_name, data_dir)

    if scan_topic is None:
        # best-effort auto-pick first LaserScan topic
        try:
            topics = list_laserscan_topics(bag_path)
            scan_topic = topics[0][0] if topics else None
        except Exception:
            scan_topic = None

    run_dir = out_root / f"slamx_{bag_name.replace('.bag','')}"
    # run slamx (call command function directly; avoids click/typer invocation quirks)
    replay(
        input_path=bag_path,
        config=config,
        out=run_dir,
        deterministic=deterministic,
        seed=seed,
        topic=scan_topic,
        write_map=False,
        max_scans=max_scans,
    )

    carto_csv: Path | None = None
    agreement: dict[str, Any] | None = None
    if carto_recorded_bag is not None:
        carto_csv = out_root / f"cartographer_{bag_name.replace('.bag','')}.csv"
        export_tf_trajectory(
            bag_path=carto_recorded_bag,
            parent=tf_parent,
            child=tf_child,
            tf_topic=tf_topic,
            out_csv=carto_csv,
        )
        agreement = bench_lib.compute_agreement(
            slamx_traj=run_dir,
            carto_csv=carto_csv,
            max_dt_ns=int(max_dt_ms) * 1_000_000,
            align=not no_align,
        )

    rep = {
        "bag_path": str(bag_path),
        "slamx_run": str(run_dir),
        "scan_topic": scan_topic,
        "cartographer_traj_csv": str(carto_csv) if carto_csv else None,
        "agreement": agreement,
    }
    bench_lib.write_json(out_root / "bench_result.json", rep)
    notes_path = out_root / "notes" / f"bench_{bag_name.replace('.bag','')}.md"
    report_lib.write_notes_markdown(notes_path, bench_lib.render_bench_markdown(rep))
    typer.echo(json.dumps({"bench": str(out_root), "notes": str(notes_path)}, ensure_ascii=False))


@bench_app.command("compare")
def bench_compare(
    slamx_run_dir: Annotated[Path, typer.Argument(help="slamx run dir (contains trajectory.json)")],
    carto_recorded_bag: Annotated[Path, typer.Argument(help="Bag recorded while running Cartographer (must include /tf)")],
    out_root: Annotated[Path, typer.Option("--out-root", "-o", help="Output root")] = Path("runs/bench_compare"),
    tf_parent: Annotated[str, typer.Option(help="TF parent frame")] = "map",
    tf_child: Annotated[str, typer.Option(help="TF child frame")] = "base_link",
    tf_topic: Annotated[str, typer.Option(help="TF topic")] = "/tf",
    max_dt_ms: Annotated[int, typer.Option(help="Association window (ms)")] = 50,
    no_align: Annotated[bool, typer.Option(help="Disable alignment in agreement metric")] = False,
) -> None:
    """Compare an existing slamx run to Cartographer TF trajectory (agreement RMSE)."""
    out_root.mkdir(parents=True, exist_ok=True)
    carto_csv = out_root / "cartographer_traj.csv"
    export_tf_trajectory(
        bag_path=carto_recorded_bag,
        parent=tf_parent,
        child=tf_child,
        tf_topic=tf_topic,
        out_csv=carto_csv,
    )
    agreement = bench_lib.compute_agreement(
        slamx_traj=slamx_run_dir,
        carto_csv=carto_csv,
        max_dt_ns=int(max_dt_ms) * 1_000_000,
        align=not no_align,
    )
    rep = {
        "slamx_run": str(slamx_run_dir),
        "carto_recorded_bag": str(carto_recorded_bag),
        "cartographer_traj_csv": str(carto_csv),
        "agreement": agreement,
    }
    bench_lib.write_json(out_root / "bench_compare.json", rep)
    notes_path = out_root / "notes" / "bench_compare.md"
    report_lib.write_notes_markdown(notes_path, bench_lib.render_bench_markdown(rep))
    typer.echo(json.dumps({"out": str(out_root), "notes": str(notes_path)}, ensure_ascii=False))


@app.command("export-slamx-trajectory")
def export_slamx_traj(
    run_dir: Annotated[Path, typer.Argument(help="slamx run directory")],
    out_csv: Annotated[Path, typer.Option("--out", "-o", help="Output CSV stamp_ns,x,y")] = Path(
        "runs/slamx_traj.csv"
    ),
) -> None:
    """Export slamx trajectory.json to CSV (stamp_ns,x,y)."""
    p = export_trajectory_csv(run_dir, out_csv)
    typer.echo(json.dumps({"out": str(p)}, ensure_ascii=False))

@eval_app.command("ate")
def eval_ate(
    traj: Annotated[Path, typer.Argument(help="run_dir or estimated traj (.csv/.json/.tum/trajectory.json)")],
    gt: Annotated[Path, typer.Option("--gt", help="Ground-truth CSV/JSON/TUM trajectory")],
    max_dt_ms: Annotated[int, typer.Option(help="Max timestamp association gap (ms)")] = 50,
    segment_gap_ms: Annotated[
        int | None,
        typer.Option(help="Optional time gap threshold used to split GT/traj into segments (ms)"),
    ] = None,
    no_align: Annotated[bool, typer.Option(help="Disable SE2 alignment (use raw)")] = False,
) -> None:
    """Compute ATE RMSE against ground truth (2D translation)."""
    from slamx.core.evaluation.ate import build_ate_report, load_estimated_trajectory, load_gt

    est = load_estimated_trajectory(traj)
    gt_traj = load_gt(gt)
    rep = build_ate_report(
        est,
        gt_traj,
        max_dt_ns=int(max_dt_ms) * 1_000_000,
        align=not no_align,
        segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
    )
    rep["traj"] = str(traj)
    rep["gt"] = str(gt)
    typer.echo(json.dumps(rep, indent=2, ensure_ascii=False))
    if not rep.get("ok"):
        raise typer.Exit(1)


@app.command()
def doctor(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to diagnose")],
) -> None:
    """Inspect telemetry for common failure signatures."""
    rep = doctor_lib.diagnose_run(run_dir)
    typer.echo(json.dumps(rep, indent=2, ensure_ascii=False))
    if any(f.get("level") == "error" for f in rep.get("findings", [])):
        raise typer.Exit(1)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory")],
    gt: Annotated[
        Path | None, typer.Option("--gt", help="Optional GT for ATE (csv/json/tum)")
    ] = None,
    max_dt_ms: Annotated[int, typer.Option(help="Max timestamp association gap (ms)")] = 50,
    segment_gap_ms: Annotated[
        int | None,
        typer.Option(help="Optional time gap threshold used to split GT/traj into segments (ms)"),
    ] = None,
    no_align: Annotated[bool, typer.Option(help="Disable SE2 alignment (use raw)")] = False,
    notes_path: Annotated[
        Path, typer.Option("--out", help="Output markdown under notes/")
    ] = Path("notes/report_latest.md"),
) -> None:
    """Generate a markdown report under notes/ (facts vs inference separated)."""
    rep = report_lib.build_report(
        run_dir=run_dir,
        gt_path=gt,
        max_dt_ns=int(max_dt_ms) * 1_000_000,
        align=not no_align,
        segment_gap_ns=(int(segment_gap_ms) * 1_000_000 if segment_gap_ms is not None else None),
    )
    md = report_lib.render_markdown(rep)
    report_lib.write_notes_markdown(notes_path, md)
    typer.echo(json.dumps({"notes": str(notes_path), "run_dir": str(run_dir)}, ensure_ascii=False))


@app.command("list-runs")
def list_runs(
    root: Annotated[Path, typer.Option(help="Runs root")] = Path("runs"),
) -> None:
    if not root.exists():
        typer.echo("(no runs dir)")
        raise typer.Exit(0)
    for p in sorted(root.iterdir()):
        if p.is_dir():
            typer.echo(str(p))


@app.command("clean")
def clean(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to delete")],
) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
        typer.echo(f"Removed {run_dir}")
    else:
        typer.echo(f"Missing {run_dir}", err=True)
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
