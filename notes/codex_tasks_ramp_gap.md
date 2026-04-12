# Codex Tasks: Ramp 2k Gap 解消

## 背景

2D SLAM (slamx) で IILABS ベンチマーク 6 データセットのうち 5 つで Cartographer を超えたが、
**ramp 2k だけ残っている** (slamx 0.45m vs Carto 0.10m)。

原因: ramp（傾斜路）で pitch 変化が起き、2D LaserScan の観測が歪む。
近距離rayが床/斜面を拾い、scan matching の cost surface を壊す。

既に試してダメだったもの:
- range-weighted ICP/correlative (linear/sigmoid) → ほぼ横ばい
- gradient mask (absolute diff / ratio) → 微改善のみ
- wider search / top-k refinement / adaptive fallback → 効果なし
- min_range / angle crop / trim_fraction tuning → 効果なし

ramp bag には以下のセンサデータがある:
- `/eve/scan` (sensor_msgs/LaserScan) — 現在使用中
- `/eve/imu/data` (sensor_msgs/Imu) — **未使用、利用可能**
- `/eve/lidar3d` (sensor_msgs/PointCloud2) — **未使用、利用可能**
- `/tf`, `/tf_static` (tf2_msgs/TFMessage)

GT: `data/iilabs3d/.../ramp/ground_truth.tum`
Carto sampled: `runs/iilabs_ramp_carto_at_slamx_s2k.csv`
Baseline run: `runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid` (align 0.4530m)

---

## Task 1: IMU-based pitch compensation

### 目的
IMU の orientation (quaternion) から pitch 角を読み、scan の有効range を pitch に応じて動的にクリップする。
pitch が大きいフレームでは近距離ray を aggressive に除去し、matching の入力品質を上げる。

### 実装方針
1. `src/slamx/core/io/bag.py` に `iter_imu_bag()` を追加
   - `/eve/imu/data` から stamp_ns, orientation (qx,qy,qz,qw), angular_velocity を yield
   - 型は新規 `ImuData` dataclass (stamp_ns, orientation_quat, angular_velocity)
2. `src/slamx/core/preprocess/pipeline.py` に `pitch_compensate_ranges()` を追加
   - 現在の pitch 角 (rad) を受け取り、bearing ごとに床/天井を拾う ray を推定して nan 化
   - 簡易モデル: `floor_range(bearing) = sensor_height / tan(pitch + bearing_elevation)`
   - `sensor_height` と `pitch` は config から渡す
3. `LocalSlamEngine.handle_scan()` で IMU を time-sync して pitch を取得、preprocess に渡す
4. CLI: `--imu-topic /eve/imu/data` オプション追加、YAML に `slam.imu.enabled`, `slam.imu.sensor_height_m` を追加

### テスト
- `tests/test_imu_preprocess.py`: pitch=0 でクリップなし、pitch=15deg で近距離除去を確認
- ramp 2k replay して ATE 評価

### 評価コマンド
```bash
env -u PYTHONPATH .venv/bin/slamx replay \
  data/iilabs3d/.../ramp/velodyne_ramp_2025-02-05-14-40-54.bag \
  --topic /eve/scan --imu-topic /eve/imu/data \
  --config configs/iilabs_ramp_imu_pitch.yaml \
  --out runs/iilabs_ramp_s2k_imu_pitch \
  --max-scans 2000 --deterministic --seed 0 --no-write-map

env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_ramp_s2k_imu_pitch \
  --gt data/iilabs3d/.../ramp/ground_truth.tum --max-dt-ms 50
```

---

## Task 2: Point-to-line ICP

### 目的
現在の point-to-point ICP を point-to-line (point-to-plane の 2D 版) に拡張。
壁面のような linear structure に対してよりrobustなアライメントを実現する。

### 実装方針
1. `src/slamx/core/local_matching/icp.py` に `IcpPointToLineConfig` と `IcpPointToLineMatcher` を追加
   - reference points から局所法線を推定 (近傍 k 点の PCA)
   - 残差を法線方向の射影に変更: `r = n · (p_src - p_dst)`
   - 法線推定が不安定な点は point-to-point にフォールバック
2. `IcpConfig` に `mode: str = "point"` を追加 ("point" or "line")
3. hybrid matcher の ICP refiner としても使えるようにする

### テスト
- `tests/test_icp_point_to_line.py`:
  - 壁面 (collinear points) に対して point-to-line が point-to-point より少ない反復で収束することを確認
  - 空の reference / scan に対してクラッシュしないことを確認

### 注意
- 法線推定は O(N log N) で KDTree + k-NN
- `min_correspondences` は既存と同じ制約を維持
- `trim_fraction` も同様に適用

---

## Task 3: Scan context / descriptor-based relocalization

### 目的
bad cluster (node 1541, 1722-1729) で score が低い場合に、scan descriptor ベースの候補生成で
correlative search よりも広い探索を可能にする。

### 実装方針
1. `src/slamx/core/local_matching/scan_context.py` に ScanContext descriptor を実装
   - ring × sector の 2D histogram (各セルに平均 range)
   - distance metric: column-shift して cosine similarity の最大を取る
2. `LocalSlamEngine` に descriptor キャッシュを持たせ、各 keyframe で descriptor を保存
3. bad score 時に descriptor 類似度でサブマップを検索し、追加の ICP refinement 候補を生成
4. 既存の hybrid fallback 機構と統合

### テスト
- `tests/test_scan_context.py`: 同一 scan の descriptor distance = 0、回転 scan は column shift で一致

### 評価
- ramp 2k の bad cluster で relocalization が入るか telemetry で確認

---

## Task 4: 3D LiDAR → 仮想 2D scan 再投影

### 目的
`/eve/lidar3d` (PointCloud2) から pitch 補正済みの仮想 2D scan を生成し、
既存の 2D SLAM パイプラインに投入する。

### 実装方針
1. `src/slamx/core/io/bag.py` に `iter_pointcloud2_bag()` を追加
   - PointCloud2 から xyz 点群を decode (rosbags の deserializer を使用)
2. `src/slamx/core/preprocess/virtual_scan.py` を新規作成
   - 3D 点群を水平面に投影して仮想 2D scan を生成
   - `z_min`, `z_max` で slice する高さ帯を config 化
   - bearing ごとに最近距離の点を取って range を生成
   - pitch / roll に依存しない安定した 2D 観測が得られる
3. CLI: `--pointcloud-topic /eve/lidar3d` オプション追加
   - 指定時は LaserScan の代わりに PointCloud2 から仮想 scan を生成

### テスト
- `tests/test_virtual_scan.py`:
  - 既知の 3D 点群から正しい 2D scan が生成されることを確認
  - z_min/z_max フィルタが効くことを確認

### 評価コマンド
```bash
env -u PYTHONPATH .venv/bin/slamx replay \
  data/iilabs3d/.../ramp/velodyne_ramp_2025-02-05-14-40-54.bag \
  --pointcloud-topic /eve/lidar3d \
  --config configs/iilabs_ramp_virtual_scan.yaml \
  --out runs/iilabs_ramp_s2k_virtual_scan \
  --max-scans 2000 --deterministic --seed 0 --no-write-map
```

---

## 優先順位

1. **Task 4 (3D→仮想2D scan)** — 最もインパクトが大きい。pitch 問題を根本から解消。
2. **Task 1 (IMU pitch compensation)** — Task 4 より軽量で、既存パイプラインへの変更が小さい。
3. **Task 2 (Point-to-line ICP)** — 壁面マッチングの質を上げる。他データセットにも効く可能性。
4. **Task 3 (Scan context)** — relocalization 改善。ramp gap の主因ではないが、robustness 向上。

## 共通の注意事項

- テストは `env -u PYTHONPATH .venv/bin/pytest tests/ -q` で全パスを確認
- 既存の 5/6 勝利データセットで regression しないこと
- コミットは機能ごとに分割
- YAML config は `configs/iilabs_ramp_*.yaml` に置く
