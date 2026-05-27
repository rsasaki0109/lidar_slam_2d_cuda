# lidar_slam_2d_cuda

![slamx multi-loop closure](docs/assets/multi_loop_closure.gif)

*Multi-loop closure on the full iilabs elevator run. The GIF first draws the failed no-loop LiDAR odometry in red, then snaps it to the loop-closed pose graph in cyan while revealing the real accepted loop edges from `telemetry.jsonl`. The start/end loop gap shrinks from 1.27 m to 0.007 m with 195 accepted loop closures; ATE on the same timestamps improves from 0.599 m to 0.030 m after alignment.*

2D LiDAR SLAM with a fixed-lag, **scan-level bundle-adjustment** core (CUDA-bound). ROS-free, CLI-first.

## Quick start

```bash
pip install -e .

# JSONL demo
slamx replay examples/fixture_scans.jsonl --config configs/scan_ba_backpack_s300.yaml --out runs/demo

# ROS bag (LaserScan topic)
pip install -e .[rosbag]
slamx replay <bag> --topic <scan_topic> --config configs/scan_ba_backpack_s300.yaml --out runs/demo
```

## How it works

Each scan is aligned against a TSDF rebuilt from a sliding window of recent scans (scan-to-local-map), then a fixed-lag window of poses is jointly optimized (Gauss-Newton / LM) with motion and anchor priors. Revisited places are detected against past nodes, verified by TSDF alignment, and closed with a global pose-graph solve. Code: `src/slamx/core/scan_ba/`. Design and roadmap: `notes/design_cuda_scan_ba.md`.

## GPU acceleration

Set `slam.scan_ba.use_cuda: true` (needs `pip install -e .[cuda]`) to run the per-scan TSDF fold and the fixed-lag window solve on the GPU, with the local map kept resident on the device. Output is numerically identical to the CPU path (poses match to ~1e-15 m) and falls back to CPU automatically when CUDA is absent.

End-to-end on Google Cartographer `backpack_2d` (RTX 4070 Ti SUPER, 80 scans):

| | per scan | total | speedup |
|--|----------|-------|---------|
| CPU (numpy) | 851 ms | 68.1 s | 1.0x |
| **CUDA (cupy)** | **126 ms** | **10.1 s** | **6.7x** |

The window solve alone is 1.1–14x faster depending on point count (`tools/bench_scan_ba_cuda.py`); folding the TSDF rebuild onto the device too is what lifts the full pipeline to 6.7x.

## Status

CPU reference + GPU path (fixed-lag scan-BA, loop closure, device-resident TSDF) landing through P2.9. Next: joint pose+SDF bundle adjustment (P3).
