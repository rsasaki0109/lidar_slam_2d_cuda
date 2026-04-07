# Cartographer ベンチ & slamx 同等性能ライン — 引き継ぎメモ（Claude / 次担当向け）

## 0. このノートの目的

- **ワークスペース**: `lidar_slam_2d`
- **やっていること**: Google Cartographer 公式 **backpack_2d** の ROS1 bag を入力に、`slamx` と Cartographer を **同一データ**で回し、**軌跡の比較**（主に ATE 風 RMSE）を再現可能にする。
- **話の置き直し（方針）**: 「Cartographer に勝った」ではなく、**同じ入力で Cartographer に出来るだけ寄せる**（Lua 数値のポート、相関スキャンマッチ、ループ、BA まわり）→ 必要なら **真の GT データセット**へ移行。

---

## 1. 「擬似 GT」とは何か（必読）

| 用語 | 意味 |
|------|------|
| **擬似 GT（pseudo GT）** | **Cartographer 自身**が `map`→`base_link` の TF で出した軌跡を CSV 化し、「正解」として `slamx eval ate` に渡すこと。**真の世界座標での位置真値ではない**。 |
| **解釈** | RMSE は *「slamx が Cartographer のその実行結果とどれだけ重なるか」*。**真の誤差（Leica / MoCap 等）ではない**。 |
| **整合のために必要な修正** | `export-tf-trajectory` は **TF メッセージの `header.stamp`（シミュ時間）**で行を書く。bag の記録時刻だけだと `replay` 軌跡と **時刻合わせできない**（実装済み。下記「確認済み事実」参照）。 |

**真の GT** へ進める候補: IILABS（OptiTrack + 2D LRF + rosbag）など。`plan` 段階では未導入。

---

## 2. データ & 固定パス

| 種別 | 絶対パス |
|------|-----------|
| 例 bag | `lidar_slam_2d/data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag` |
| 水平レーザ topic | `horizontal_laser_2d`（`sensor_msgs/MultiEchoLaserScan`） |
| Cart 記録 bag（TF 含む例） | `lidar_slam_2d/runs/carto_recorded.bag` |
| Cart 全軌跡 CSV（擬似 GT） | `lidar_slam_2d/runs/cartographer_traj.csv` |
| 300 窓 GT | `lidar_slam_2d/runs/cartographer_traj_s300_window.csv` |
| 2k 窓 GT | `lidar_slam_2d/runs/cartographer_traj_s2k_window.csv` |
| ベンチ結果 JSON（要更新） | `lidar_slam_2d/notes/benchmark_cartographer_agreement_b0_2014-07-11.json` |
| Lua ポート対応表 | `lidar_slam_2d/notes/cartographer_backpack2d_port.md` |

---

## 3. 設定ファイルの役割（どれをいつ使うか）

| ファイル | 用途 |
|----------|------|
| `configs/bench_fast.yaml` | **高速回帰**: ICP、粗め preprocess。**Cart 一致目的ではない**。短時間ベンチ用。 |
| `configs/bench_backpack_full.yaml` | **全区間完走優先**: adaptive 間引き、`pose_graph_skip_optimization_from_node` で **後半 BA カット**。RMSE は擬似 GT と **悪化しやすい**。 |
| `configs/cartographer_backpack2d_port.yaml` | Cartographer Lua **数値の出典付きポート**（相関、格子 0.05 等）。ループ off のまま、など。 |
| `configs/cartographer_backpack2d_port_full.yaml` | 上記 + `bench_backpack_full` 系の長系列ガード（**後半 BA カットあり**）。 |
| **`configs/cartographer_parity_medium.yaml`** | **短〜中距離で Cart 寄せ**: 相関 + フルグラフ BA（スキップなし）+ **ヒューリスティック ループ有効**。**壁時計は重い**（相関スキャンマッチ）。 |
| **`configs/cartographer_parity_full.yaml`** | **全区間用（試験的）**: `slam.pose_graph.optimization_window` で **スライディングウィンドウ BA**。プレフィックス姿勢は固定され **大域再配置は Cart 未満**だが、全グラフ `least_squares` より現実的。 |

### 実装メモ: `optimization_window`

- **YAML**: `slam.pose_graph.optimization_window`（整数 N 本の末尾ノードだけ変数、それ以前は当該ステップでは固定）。
- **コード**: `src/slamx/core/backend/pose_graph.py` の `PoseGraphConfig.optimization_window`。
- **CLI 読込**: `src/slamx/cli/main.py` の `_engine_from_config`。

---

## 4. 測定結果（数値の系譜）

**集計の正本は `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`。** 以下は README 用の要約（**2026-04-08 時点**。2k parity は進行中の可能性あり → 完了後 JSON を更新すること）。

### 4.1 旧ライン: `bench_fast`（ICP・速度優先）

| 区間 | slamx ラン例 | align RMSE (m) | no_align (m) | n |
|------|----------------|----------------|--------------|---|
| 300 | `runs/slamx_backpack2d_short` | ~1.59 | ~3.09 | 258 |
| 2000 | `runs/slamx_backpack2d_2k` | ~7.69 | ~16.98 | 1953 |
| 全区間 | `runs/slamx_backpack2d_full_capped` + `bench_backpack_full` | ~17.14 | ~42.14 | 5438 |

（上記は JSON に同期済みの古いイベント。）

### 4.2 **Cart 同等寄せライン**: `cartographer_parity_medium`（相関 + ループ + BA 省略なし）

| 区間 | slamx ラン | align RMSE (m) | no_align (m) | n | メモ |
|------|------------|----------------|--------------|---|------|
| 300 | `runs/slamx_parity_medium_s300` | **~0.081** | **~0.429** | 258 | 壁時計 **~45 分規模**で確認（環境依存）。**bench_fast より桁違いに改善**。 |
| 2000 | `runs/slamx_parity_medium_s2k` | （未完了なら TBD） | （TBD） | — | バックグラウンド実行中だった。**完走後** `eval ate` → JSON へ追記。 |

**Windows GT**: 300 / 2k はそれぞれ `cartographer_traj_s300_window.csv` / `s2k_window.csv`（slamx の `trajectory.json` の stamp 範囲に合わせた Cart CSV）。

### 4.3 Cartographer 側

- Pipeline 例: **ROS Melodic** + `offline_backpack_2d.launch` + `no_rviz:=true`（`tools/cartographer_noetic/` Docker）。
- Noetic apt には Cartographer バイナリが無い想定で **Melodic ベース**。

---

## 5. コマンドチートシート（コピペ用）

前置き: `env -u PYTHONPATH` で venv の `slamx` を使う（PYTHONPATH 汚染回避）。

### 5.1 slamx — parity medium（300 スキャン）

```bash
cd lidar_slam_2d
rm -rf runs/slamx_parity_medium_s300
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config configs/cartographer_parity_medium.yaml \
  --out runs/slamx_parity_medium_s300 \
  --max-scans 300 \
  --deterministic --seed 0 \
  --no-write-map
```

### 5.2 slamx — parity medium（2000 スキャン・長時間）

```bash
cd lidar_slam_2d
rm -rf runs/slamx_parity_medium_s2k
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config configs/cartographer_parity_medium.yaml \
  --out runs/slamx_parity_medium_s2k \
  --max-scans 2000 \
  --deterministic --seed 0 \
  --no-write-map
```

（バックグラウンドなら `nohup ... > runs/slamx_parity_medium_s2k.log 2>&1 &`）

### 5.3 slamx — 全区間 parity full（ウィンドウ BA）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx replay \
  data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  --topic horizontal_laser_2d \
  --config configs/cartographer_parity_full.yaml \
  --out runs/slamx_parity_full \
  --deterministic --seed 0 \
  --no-write-map
```

### 5.4 旧: bench_fast / full capped（比較用）

`plan` 本文中の以前のセクションと同様。パスは必ず `lidar_slam_2d/` 基準で統一。

### 5.5 Cartographer（Docker）

```bash
cd lidar_slam_2d
./tools/cartographer_noetic/run_backpack2d.sh \
  lidar_slam_2d/data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag \
  horizontal_laser_2d \
  lidar_slam_2d/runs/carto_recorded.bag
```

### 5.6 TF → CSV（擬似 GT）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx export-tf-trajectory \
  runs/carto_recorded.bag \
  --parent map --child base_link --tf-topic /tf \
  --out runs/cartographer_traj.csv
```

### 5.7 `eval ate`（時間窓 GT）

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_medium_s300 \
  --gt runs/cartographer_traj_s300_window.csv --max-dt-ms 50

env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_medium_s300 \
  --gt runs/cartographer_traj_s300_window.csv --max-dt-ms 50 --no-align
```

2k 完走後は `runs/slamx_parity_medium_s2k` + `runs/cartographer_traj_s2k_window.csv`。

### 5.8 窓 CSV の作り方（再生成）

`runs/<slamx_run>/trajectory.json` の **先頭・末尾の `stamp_ns`** で `cartographer_traj.csv` をフィルタ（既存の `*_window.csv` と同じ論理）。スクリプトが無ければ `awk`/小さな Python で可。

---

## 6. ループ検出（telemetry）

- `cartographer_parity_*` で **ヒューリスティック NN + スキャンマッチ**（`src/slamx/core/loop_detection/heuristic.py`）。
- `telemetry.jsonl` に `loop_closure_accepted` / `loop_closure_rejected` / `loop_closure_candidates` が出る。**accept がほぼ無い**なら `search_radius_m`, `min_separation_nodes`, `accept_score` のスイープ候補。

---

## 7. 未完了タスク（次担当向けチェックリスト）

1. **`runs/slamx_parity_medium_s2k` の完走確認**  
   - `trajectory.json` の有無、所要時間の記録。  
   - `eval ate`（align / no_align）→ **`benchmark_cartographer_agreement_b0_2014-07-11.json` に `runs_2000_parity_medium` 等のキーで追記**（旧 `bench_fast` 2k は温存で比較列として残すとよい）。
2. **`cartographer_parity_full` 全区間**の壁時計 & RMSE（擬似 GT 全区間 CSV）— ウィンドウ BA のドリフト限界を評価。
3. **真の GT データ 1 本**を同じ `eval ate` に載せる（IILABS 等）。擬似 GT からの卒業条件を明文化。
4. **コード**: ウィンドウ BA のプレフィックス固定による累積誤差 → 間欠フル BA / マージン化など（要設計）。
5. **外部**: `edge-scene-db` 向け Issue（backpack_2d 公開 bag の文書化）— `https://github.com/rsasaki0109/edge-scene-db/issues/1`（状況はリポ側で確認）。

---

## 8. テスト

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/pytest -q
```

ウィンドウ BA: `tests/test_pose_graph_window.py`。

---

## 9. Git / 主要コミットの目安（調査用）

- ポーズグラフ `optimization_window` + parity 設定 YAML 追加: `Add sliding-window pose-graph optimize + Cartographer parity configs`（ハッシュは `git log --oneline -5` で確認）。

---

## 10. 変更不要だった領域

- このファイル以外を勝手に大掃除しないこと。**ユーザー指示が無い限り**ベンチ JSON の数値は手で整合させる（スクリプト化は任意）。
