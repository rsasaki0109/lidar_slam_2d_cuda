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
- **次の一手**: この bag は水平約 **5522** フレーム。`configs/bench_fast.yaml` で **`--max-scans` 無し／5000 付近**の `replay` は **壁時計で極端に重く**、同一セッション内で `trajectory.json` まで到達しなかった例がある（恐らく pose graph の最適化コスト）。**全区間の apple-to-apple** は、`optimize_every_n_keyframes` の更なる間引き・CLI での途中保存・別バックエンド等の **実装側対応**が現実的。当面は **`--max-scans 2000` 前後**で延長評価するか、**300 で回帰確認**と割り切る。

## 確認済み事実

- **リポジトリ/ワークスペース**: `lidar_slam_2d`
- **slamx CLI**: `replay`, `export-tf-trajectory`, `eval ate`, `bench compare`, `datasets download-cartographer-backpack2d` など（実装済み）
- **公開データ**: `horizontal_laser_2d` は `sensor_msgs/msg/MultiEchoLaserScan`
- **Docker 記録**: `./tools/cartographer_noetic/run_backpack2d.sh <bag> horizontal_laser_2d <out.bag>` → `runs/carto_recorded.bag`（`/tf`, `/tf_static`）。`rosbag record` の終了時は **`.bag.active` 残り**があり得るため、スクリプト側で `reindex` + `cp` フォールバックあり。

## 未確認/要確認項目

- **bench_fast** のまま全区間 `replay` を安定完了させる（または代替 config / 実装で同一 bag 全区間を現実時間で回せるようにする）
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

### A''. slamx（全区間・現状は非推奨 / 未完成になりやすい）

`configs/bench_backpack_full.yaml`（`optimize_every_n_keyframes: 200`）や `bench_fast` の **無制限** `replay` は長時間ブロックしうる。上記 A' のような **明示的 `--max-scans`** を推奨。

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
