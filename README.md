# lidar_slam_2d

Modern 2D LiDAR SLAM experiment platform with a ROS-free core and a CLI-first workflow for replay, evaluation, and benchmark iteration.

![lidar_slam_2d benchmark summary](docs/assets/benchmark-summary.svg)

![lidar_slam_2d trajectory gallery](docs/assets/trajectory-gallery.svg)

## Why this repo is worth publishing now

- **Backpack parity is already strong.** The current parity line takes the Cartographer backpack_2d comparison from `1.593 -> 0.081 m` on the 300-scan slice and from `7.693 -> 0.475 m` on the 2k-scan slice.
- **Generalization is real.** The current front-end already beats the sampled Cartographer reference on multiple IILABS sequences, including `loop` and `slippage`.
- **The workflow is inspectable.** Configs, notes, telemetry, and replay outputs are all tracked in a way that makes the iteration path understandable instead of hiding the tuning history.

## Quick start

```bash
pip install -e .
slamx replay examples/fixture_scans.jsonl --out runs/demo --no-write-map
slamx eval ate runs/demo --gt runs/cartographer_traj_s300_window.csv
```

The published project name is **`lidar_slam_2d`**. The CLI command remains **`slamx`** for now.

Optional ROS bag support:

```bash
pip install -e .[rosbag]
```

## Public site

A lightweight GitHub Pages site lives in `docs/` and can be deployed with `.github/workflows/pages.yml`.
The page is intentionally small and reuses the same SVG assets shown above.

## Current status

The public-facing story is ready, but the research line is still active. The main open benchmark gap is the `ramp` sequence: a pitch-aware local-gradient mask helps a little on 1750-scan screening, but it does not yet close the full 2k gap.

## Benchmark reproduction configs

For the Cartographer `backpack_2d` parity numbers, the exact recorded best runs now have fixed configs:

- `configs/cartographer_parity_medium_s300_locked.yaml` reproduces the best 300-scan run.
- `configs/cartographer_parity_noloop_s2k.yaml` reproduces the best 2k-scan run.

`configs/cartographer_parity_medium.yaml` remains the actively edited parity working config, and `configs/cartographer_parity_full.yaml` is retained only as a deprecated exploratory full-bag config.

## Refreshing the public assets

```bash
python tools/generate_public_assets.py
```
