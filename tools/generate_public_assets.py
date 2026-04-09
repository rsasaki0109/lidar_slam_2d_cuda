from __future__ import annotations

import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slamx.core.evaluation.ate import (  # noqa: E402
    associate_by_time,
    compute_ate_rmse,
    load_estimated_trajectory,
    umeyama_alignment_2d,
)


@dataclass(frozen=True)
class BenchmarkCard:
    label: str
    reference_label: str
    reference_path: Path
    slamx_label: str
    slamx_path: Path
    gt_path: Path


@dataclass(frozen=True)
class OverlayPanel:
    label: str
    slamx_path: Path
    reference_path: Path
    reference_label: str


BENCHMARK_CARDS = [
    BenchmarkCard(
        label="Backpack 300 scans",
        reference_label="bench_fast",
        reference_path=ROOT / "runs" / "slamx_backpack2d_short",
        slamx_label="lidar_slam_2d parity",
        slamx_path=ROOT / "runs" / "slamx_parity_medium_s300",
        gt_path=ROOT / "runs" / "cartographer_traj_s300_window.csv",
    ),
    BenchmarkCard(
        label="Backpack 2k scans",
        reference_label="bench_fast",
        reference_path=ROOT / "runs" / "slamx_backpack2d_2k",
        slamx_label="lidar_slam_2d parity",
        slamx_path=ROOT / "runs" / "slamx_parity_noloop_s2k",
        gt_path=ROOT / "runs" / "cartographer_traj_s2k_window.csv",
    ),
    BenchmarkCard(
        label="IILABS loop 2k",
        reference_label="Cartographer sampled",
        reference_path=ROOT / "runs" / "iilabs_loop_carto_at_slamx_s2k.csv",
        slamx_label="lidar_slam_2d best",
        slamx_path=ROOT / "runs" / "iilabs_loop_s2k_refinegrid_top3_fallback_mid",
        gt_path=ROOT
        / "data"
        / "iilabs3d"
        / "iilabs3d_dataset"
        / "benchmark"
        / "velodyne_vlp-16"
        / "loop"
        / "ground_truth.tum",
    ),
    BenchmarkCard(
        label="IILABS slippage full",
        reference_label="Cartographer sampled",
        reference_path=ROOT / "runs" / "iilabs_slippage_carto_at_slamx.csv",
        slamx_label="lidar_slam_2d best",
        slamx_path=ROOT / "runs" / "iilabs_slippage_hybrid_refdense_cv_s2_refinegrid",
        gt_path=ROOT
        / "data"
        / "iilabs3d"
        / "iilabs3d_dataset"
        / "benchmark"
        / "velodyne_vlp-16"
        / "slippage"
        / "ground_truth.tum",
    ),
]

OVERLAY_PANELS = [
    OverlayPanel(
        label="Backpack 300 scans",
        slamx_path=ROOT / "runs" / "slamx_parity_medium_s300",
        reference_path=ROOT / "runs" / "cartographer_traj_s300_window.csv",
        reference_label="Cartographer pseudo GT",
    ),
    OverlayPanel(
        label="Backpack 2k scans",
        slamx_path=ROOT / "runs" / "slamx_parity_noloop_s2k",
        reference_path=ROOT / "runs" / "cartographer_traj_s2k_window.csv",
        reference_label="Cartographer pseudo GT",
    ),
    OverlayPanel(
        label="IILABS loop 2k",
        slamx_path=ROOT / "runs" / "iilabs_loop_s2k_refinegrid_top3_fallback_mid",
        reference_path=ROOT
        / "data"
        / "iilabs3d"
        / "iilabs3d_dataset"
        / "benchmark"
        / "velodyne_vlp-16"
        / "loop"
        / "ground_truth.tum",
        reference_label="Ground truth",
    ),
    OverlayPanel(
        label="IILABS slippage full",
        slamx_path=ROOT / "runs" / "iilabs_slippage_hybrid_refdense_cv_s2_refinegrid",
        reference_path=ROOT
        / "data"
        / "iilabs3d"
        / "iilabs3d_dataset"
        / "benchmark"
        / "velodyne_vlp-16"
        / "slippage"
        / "ground_truth.tum",
        reference_label="Ground truth",
    ),
]


def _metric(path: Path, gt_path: Path) -> float:
    est = load_estimated_trajectory(path)
    gt = load_estimated_trajectory(gt_path)
    slam_xy, gt_xy = associate_by_time(est, gt, max_dt_ns=50_000_000)
    rep = compute_ate_rmse(slam_xy, gt_xy, align=True)
    if not rep.get("ok"):
        raise RuntimeError(f"Unable to compute ATE for {path} vs {gt_path}")
    return float(rep["rmse_m"])


def _fmt_m(value: float) -> str:
    return f"{value:.3f} m"


def _fmt_gain(reference: float, slamx: float) -> str:
    if slamx <= 0.0:
        return "best"
    gain = reference / slamx
    if gain >= 1.05:
        return f"{gain:.1f}x lower"
    delta = slamx - reference
    if abs(delta) < 1e-6:
        return "parity"
    if delta < 0.0:
        return f"{abs(delta):.3f} m lower"
    return f"{delta:.3f} m higher"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _benchmark_summary_svg(cards: list[BenchmarkCard]) -> str:
    width = 1200
    height = 760
    card_w = 520
    card_h = 240
    gap_x = 40
    gap_y = 32
    origin_x = 60
    origin_y = 170

    results: list[tuple[BenchmarkCard, float, float]] = []
    for card in cards:
        ref_rmse = _metric(card.reference_path, card.gt_path)
        slamx_rmse = _metric(card.slamx_path, card.gt_path)
        results.append((card, ref_rmse, slamx_rmse))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">lidar_slam_2d benchmark summary</title>',
        '<desc id="desc">A public benchmark summary comparing reference trajectories against lidar_slam_2d results across backpack and IILABS datasets.</desc>',
        '<defs><linearGradient id="bg" x1="0" x2="1" y1="0" y2="1"><stop offset="0%" stop-color="#0b1220"/><stop offset="100%" stop-color="#111827"/></linearGradient></defs>',
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        '<text x="60" y="78" font-size="42" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#f8fafc">lidar_slam_2d benchmark summary</text>',
        '<text x="60" y="116" font-size="20" font-family="Inter,Arial,sans-serif" fill="#cbd5e1">Lower align RMSE is better. Backpack numbers use Cartographer pseudo GT; IILABS uses provided ground truth.</text>',
    ]

    for idx, (card, ref_rmse, slamx_rmse) in enumerate(results):
        col = idx % 2
        row = idx // 2
        x = origin_x + col * (card_w + gap_x)
        y = origin_y + row * (card_h + gap_y)
        gain = _fmt_gain(ref_rmse, slamx_rmse)
        improvement = slamx_rmse < ref_rmse
        pill_fill = "#0f766e" if improvement else "#334155"
        slamx_fill = "#60a5fa" if improvement else "#f59e0b"

        parts.extend(
            [
                f'<rect x="{x}" y="{y}" rx="28" ry="28" width="{card_w}" height="{card_h}" fill="#0f172a" stroke="#1e293b" stroke-width="2"/>',
                f'<text x="{x + 28}" y="{y + 44}" font-size="26" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#f8fafc">{escape(card.label)}</text>',
                f'<rect x="{x + 28}" y="{y + 60}" rx="16" ry="16" width="140" height="32" fill="{pill_fill}"/>',
                f'<text x="{x + 98}" y="{y + 82}" text-anchor="middle" font-size="16" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#ecfeff">{escape(gain)}</text>',
                f'<text x="{x + 28}" y="{y + 130}" font-size="18" font-family="Inter,Arial,sans-serif" fill="#94a3b8">{escape(card.reference_label)}</text>',
                f'<text x="{x + 28}" y="{y + 175}" font-size="38" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#e2e8f0">{_fmt_m(ref_rmse)}</text>',
                f'<text x="{x + 280}" y="{y + 130}" font-size="18" font-family="Inter,Arial,sans-serif" fill="#7dd3fc">{escape(card.slamx_label)}</text>',
                f'<text x="{x + 280}" y="{y + 175}" font-size="38" font-family="Inter,Arial,sans-serif" font-weight="700" fill="{slamx_fill}">{_fmt_m(slamx_rmse)}</text>',
                f'<line x1="{x + 242}" y1="{y + 118}" x2="{x + 242}" y2="{y + 192}" stroke="#1e293b" stroke-width="2"/>',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)


def _downsample(points: list[tuple[float, float]], *, max_points: int = 700) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _normalize(
    points: list[tuple[float, float]],
    *,
    bounds: list[tuple[float, float]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> list[tuple[float, float]]:
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((w - 36.0) / span_x, (h - 48.0) / span_y)
    offset_x = x + (w - span_x * scale) / 2.0
    offset_y = y + (h + span_y * scale) / 2.0
    return [
        (offset_x + (px - min_x) * scale, offset_y - (py - min_y) * scale)
        for px, py in points
    ]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _trajectory_overlay_svg(panels: list[OverlayPanel]) -> str:
    width = 1400
    height = 980
    panel_w = 610
    panel_h = 350
    gap_x = 40
    gap_y = 40
    origin_x = 60
    origin_y = 170

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">lidar_slam_2d trajectory gallery</title>',
        '<desc id="desc">Overlay panels showing aligned lidar_slam_2d trajectories against reference trajectories across backpack and IILABS datasets.</desc>',
        '<defs><linearGradient id="bg2" x1="0" x2="1" y1="0" y2="1"><stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#eef2ff"/></linearGradient></defs>',
        '<rect width="100%" height="100%" fill="url(#bg2)"/>',
        '<text x="60" y="78" font-size="42" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#0f172a">Trajectory overlays</text>',
        '<text x="60" y="116" font-size="20" font-family="Inter,Arial,sans-serif" fill="#475569">lidar_slam_2d trajectories are aligned in SE(2) to the reference path for visual comparison.</text>',
        '<rect x="60" y="130" width="14" height="14" rx="7" fill="#2563eb"/>',
        '<text x="84" y="142" font-size="16" font-family="Inter,Arial,sans-serif" fill="#0f172a">lidar_slam_2d</text>',
        '<rect x="150" y="130" width="14" height="14" rx="7" fill="#94a3b8"/>',
        '<text x="174" y="142" font-size="16" font-family="Inter,Arial,sans-serif" fill="#0f172a">reference</text>',
    ]

    for idx, panel in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x = origin_x + col * (panel_w + gap_x)
        y = origin_y + row * (panel_h + gap_y)

        slamx = load_estimated_trajectory(panel.slamx_path)
        ref = load_estimated_trajectory(panel.reference_path)
        slam_xy, ref_xy = associate_by_time(slamx, ref, max_dt_ns=50_000_000)
        rep = compute_ate_rmse(slam_xy, ref_xy, align=True)
        if not rep.get("ok"):
            raise RuntimeError(f"Unable to render overlay for {panel.label}")

        R, t = umeyama_alignment_2d(slam_xy, ref_xy)
        slamx_aligned = (R @ slamx.xy.T).T + t
        ref_raw = [tuple(v) for v in ref.xy.tolist()]
        slamx_raw = [tuple(v) for v in slamx_aligned.tolist()]
        combined = ref_raw + slamx_raw
        ref_points = _normalize(
            _downsample(ref_raw),
            bounds=combined,
            x=x,
            y=y + 44,
            w=panel_w,
            h=panel_h - 70,
        )
        slamx_points = _normalize(
            _downsample(slamx_raw),
            bounds=combined,
            x=x,
            y=y + 44,
            w=panel_w,
            h=panel_h - 70,
        )

        if combined:
            bounds = combined
            xs = [p[0] for p in bounds]
            ys = [p[1] for p in bounds]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
        else:
            span = 0.0

        parts.extend(
            [
                f'<rect x="{x}" y="{y}" rx="28" ry="28" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
                f'<text x="{x + 24}" y="{y + 38}" font-size="24" font-family="Inter,Arial,sans-serif" font-weight="700" fill="#0f172a">{escape(panel.label)}</text>',
                f'<text x="{x + panel_w - 24}" y="{y + 38}" text-anchor="end" font-size="16" font-family="Inter,Arial,sans-serif" fill="#475569">{escape(panel.reference_label)} | align RMSE {_fmt_m(float(rep["rmse_m"]))}</text>',
                f'<polyline points="{_polyline(ref_points)}" fill="none" stroke="#94a3b8" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>',
                f'<polyline points="{_polyline(slamx_points)}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
                f'<text x="{x + 24}" y="{y + panel_h - 18}" font-size="14" font-family="Inter,Arial,sans-serif" fill="#64748b">matched span: {int(rep["n"])} points | footprint: {span:.2f} m</text>',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    out_dir = ROOT / "docs" / "assets"
    _write(out_dir / "benchmark-summary.svg", _benchmark_summary_svg(BENCHMARK_CARDS))
    _write(out_dir / "trajectory-gallery.svg", _trajectory_overlay_svg(OVERLAY_PANELS))
    print(f"Wrote SVG assets to {out_dir}")


if __name__ == "__main__":
    main()
