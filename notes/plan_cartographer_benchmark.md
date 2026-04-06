## 結論

- **目的**: 公開データで `slamx` と Google Cartographer を **同一入力**で走らせ、**軌跡一致度（擬似 GT としての agreement / ATE）**を数値化する。
- **現状（2026-04 確認）**:
  - `slamx` は Cartographer 公式 backpack_2d の ROS1 `.bag` を直接 `replay` できる。
  - Cartographer は **ROS Noetic（focal）の apt にバイナリが無い**一方、**ROS Melodic（bionic）では apt 導入可能**のため、`tools/cartographer_noetic/` の Docker イメージを **Melodic ベース**にしている。
  - **オフライン処理**は `offline_backpack_2d.launch` + `no_rviz:=true`（`demo_backpack_2d.launch` は bag 終了後も戻らず固まりやすい）。
  - `slamx export-tf-trajectory` は **TF の `header.stamp`（sim 時刻）**で CSV を書く（従来の bag ログ時刻だと `replay` 軌跡と合わない）。
- **測定例（b0-2014-07-11-10-58-16.bag）**（擬似 GT = Cartographer TF、`eval_max_dt_ms=50`、いずれも **slamx の時間窓に `cartographer_traj.csv` を合わせた**）:
  - **300 スキャン**: align **RMSE 約 1.59 m**（n=258）、`--no-align` **約 3.09 m**。
  - **2000 スキャン**（実用的な延長区間）: align **RMSE 約 7.69 m**（n=1953）、`--no_align` **約 16.98 m**（長区間では slamx と Cartographer の形状差が支配的になりやすい）。
  - 集計 JSON: `lidar_slam_2d/notes/benchmark_cartographer_agreement_b0_2014-07-11.json`
- **全区間 slamx（実装対応後）**: `scipy.least_squares` の **変数数・残差数が袋長に伴って爆発**するのが主因だった。対策として YAML で **`slam.pose_graph.max_nfev_cap`**、`optimize_adaptive_from_node` / `optimize_min_interval_for_long_runs`、および大規模時 **`pose_graph_skip_optimization_from_node`**（指定ノード以降は **グローバル BA を省略**しスキャンマッチのオドメのみ）を追加。**`configs/bench_backpack_full.yaml`** で約 **5522 姿勢・~24 分**程度まで `replay` 完了を確認（Tail は BA 無しのため Cartographer 擬似 GT との RMSE は悪化しうる）。計測値は `notes/benchmark_cartographer_agreement_b0_2014-07-11.json` の `runs_full_bag_slamx`。

## 確認済み事実

- **リポジトリ/ワークスペース**: `lidar_slam_2d`
- **slamx CLI**: `replay`, `export-tf-trajectory`, `eval ate`, `bench compare`, `datasets download-cartographer-backpack2d` など（実装済み）
- **公開データ**: `horizontal_laser_2d` は `sensor_msgs/msg/MultiEchoLaserScan`
- **Docker 記録**: `./tools/cartographer_noetic/run_backpack2d.sh <bag> horizontal_laser_2d <out.bag>` → `runs/carto_recorded.bag`（`/tf`, `/tf_static`）。`rosbag record` の終了時は **`.bag.active` 残り**があり得るため、スクリプト側で `reindex` + `cp` フォールバックあり。

## 未確認/要確認項目

- **全区間を BA 省略なしで**現実時間内に回す（スパースヤコビアン・ウィンドウ化・階層最適化など）
- `map` / `base_link` 以外のフレーム名になるデータセットでは `--parent` / `--child` の再確認
- **真の GT** が無い限り、数値は **「Cartographer にどれだけ寄せたか」**の読みに留まる

## 次アクション（コマンドテンプレ）

### A. slamx（短区間）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config lidar_slam_2d/configs/bench_fast.yaml \
  --out lidar_slam_2d/runs/slamx_backpack2d_short \
  --max-scans 300 \
  --deterministic --seed 0 \
  --no-write-map
```

### A'. slamx（延長区間の例: 2000 スキャン）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config lidar_slam_2d/configs/bench_fast.yaml \
  --out lidar_slam_2d/runs/slamx_backpack2d_2k \
  --max-scans 2000 \
  --deterministic --seed 0 \
  --no-write-map
```

`trajectory.json` の先頭・末尾 `stamp_ns` で `runs/cartographer_traj.csv` を切り出し（例: `runs/cartographer_traj_s2k_window.csv`）、`eval ate` に渡す。

### A''. slamx（全区間・`bench_backpack_full.yaml`）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config lidar_slam_2d/configs/bench_backpack_full.yaml \
  --out lidar_slam_2d/runs/slamx_backpack2d_full_capped \
  --deterministic --seed 0 \
  --no-write-map
```

- `slam.pose_graph` … `max_nfev_cap` で各 BA の scipy 予算を上限化
- `optimize_adaptive_*` … 長系列で最適化間隔を自動延長
- `pose_graph_skip_optimization_from_node` … 指定ノード以降は **グローバル BA スキップ**（速度優先・精度は特に後半で劣化しうる）

短区間の回帰には引き続き **`configs/bench_fast.yaml` + `--max-scans`** が無難。

### B. Cartographer（Docker / Melodic）

```bash
cd lidar_slam_2d
./tools/cartographer_noetic/run_backpack2d.sh \
  lidar_slam_2d/data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  horizontal_laser_2d \
  lidar_slam_2d/runs/carto_recorded.bag
```

### C. 比較

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx export-tf-trajectory \
  lidar_slam_2d/runs/carto_recorded.bag \
  --parent map --child base_link --tf-topic /tf \
  --out lidar_slam_2d/runs/cartographer_traj.csv

env -u PYTHONPATH .venv/bin/slamx eval ate \
  lidar_slam_2d/runs/slamx_backpack2d_short \
  --gt lidar_slam_2d/runs/cartographer_traj.csv \
  --max-dt-ms 50

env -u PYTHONPATH .venv/bin/slamx eval ate \
  lidar_slam_2d/runs/slamx_backpack2d_short \
  --gt lidar_slam_2d/runs/cartographer_traj.csv \
  --max-dt-ms 50 --no-align

# または
env -u PYTHONPATH .venv/bin/slamx bench compare \
  lidar_slam_2d/runs/slamx_backpack2d_short \
  lidar_slam_2d/runs/carto_recorded.bag \
  --tf-parent map --tf-child base_link --tf-topic /tf
```

### C'. slamx 時間窓に GT を合わせる（300 スキャン例）

`runs/slamx_backpack2d_short/trajectory.json` の先頭・末尾 `stamp_ns` で `runs/cartographer_traj.csv` をフィルタし、`runs/cartographer_traj_s300_window.csv` を作って `--gt` に指定する（リポジトリ内の `notes/benchmark_cartographer_agreement_b0_2014-07-11.json` と同条件）。
