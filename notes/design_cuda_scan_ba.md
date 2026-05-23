# Fixed-lag scan-level BA on CUDA — 設計メモ

2026-05-23 起稿。`lidar_slam_2d_cuda` の中核となる新フロントエンドの設計骨子。

## 1. 動機と立ち位置

- 現行は pose graph + scan matching の構成。pose だけが変数で、スキャン点群はマッチング段で消費されて以降は再利用されない。
- 新方式は **fixed-lag ウィンドウ内のスキャン点群を直接コストに残し**、ウィンドウ全体で同時に最適化する。視覚 SLAM の bundle adjustment と同種だが、構造（map）側は離散 voxel の SDF / occupancy として表現する。
- 期待される利点:
  - **点単位の残差を並列化**できるため CUDA との相性が良い (各 scan 1000〜2000 点 × ウィンドウ K=30〜100 → 残差数 30k〜200k)。
  - スキャン間の弱い整合性を引きずらず、ウィンドウ内で大域整合をとれる。
  - SDF を陽に持つことで、occupancy 更新と最適化の二度漬けを避け、コスト勾配が連続。
- 制約: **2D 限定**。3D 拡張は当面考えない（命名にも反映済み）。

## 2. 状態変数

ウィンドウ \(W = \{t_0, t_0+1, \dots, t_0+K-1\}\) を保持。

- **Pose**: \(T_t \in SE(2)\), \(t \in W\)。tangent space は \(\xi_t = (\delta x, \delta y, \delta\theta) \in \mathbb{R}^3\)。
- **Map (構造)**: ウィンドウローカルの 2D Truncated Signed Distance Field
  - \(\phi: \mathbb{Z}^2 \to \mathbb{R}\) を voxel グリッドで持つ（解像度 e.g. 0.05 m）。
  - 値域は \([-\tau, +\tau]\) で打ち切り (\(\tau\) は数 voxel 分)。
  - 重み \(w_v\) を保持し、観測不足セルは最適化対象から除外。
- 第一段では SDF を**固定パラメータ**として扱い (pose のみ最適化)、第二段で SDF も同時最適化する full BA に拡張する。

## 3. コスト関数

スキャン点 \(p_{t,i} \in \mathbb{R}^2\) (センサ座標) について

\[
r_{t,i}(\xi_t, \phi) = \phi\bigl(T_t \cdot p_{t,i}\bigr) - 0
\]

を 0 に押し込むのが point-to-SDF コスト。連続化のためには \(\phi\) を bilinear 補間。

- **データ項**: \(\sum_t \sum_i \rho\bigl(r_{t,i}\bigr)\)。\(\rho\) は Huber (外れ点抑制)。
- **モーション事前**: 連続 pose 間に \(\Vert \log(T_t^{-1} T_{t+1}) - u_t \Vert_{\Sigma}^2\)。オドメトリ or 定速予測。
- **境界事前 (marginalization prior)**: ウィンドウ先頭 \(t_0\) に対する Schur 形ガウシアン。これがウィンドウ外履歴の唯一の窓口。
- **SDF 滑らかさ (第二段以降)**: \(\sum_{(u,v) \in \mathcal{E}} \Vert \phi_u - \phi_v \Vert^2\) で隣接 voxel 間。

## 4. Fixed-lag 構造

- 新 scan が来たら window に push。古い \(t_0\) は `marginalize` で
  - その pose を Schur 補で消去し、隣接 pose と SDF 境界に prior を残す。
  - 出ていく観測点が触れた SDF voxel は global frozen SDF に焼き付け、ローカル grid からは外す。
- これにより**最適化サイズは K に対して有界**で、リアルタイム性が保証される。
- 第一実装では「marginalize = 古い pose に強い prior を貼って固定」の近似でよい（厳密 Schur は第二段）。

## 5. 最適化

- **Gauss-Newton / Levenberg-Marquardt** を CUDA 上で実装。
- 1 iteration の流れ:
  1. **Warp**: GPU で各点を \(T_t\) で世界座標に飛ばす。 [N_points 並列]
  2. **Sample**: SDF を bilinear 補間して \(r\) と勾配 \(\nabla\phi\) を取得。
  3. **Per-point Jacobian**: \(J_{t,i} = \nabla\phi^\top \cdot \partial(T_t p) / \partial \xi_t\) を計算。SDF 同時最適化時は SDF voxel 4 セルへの hat も書き出す。
  4. **JᵀJ / Jᵀr accumulation**: pose ブロックは K×3 次元と小さいので thread block ごとに shared memory で reduce → atomicAdd で全体に集約。
  5. **Solve**: pose-only なら 3K×3K の dense Cholesky で OK (cuSOLVER)。SDF 同時最適化は voxel 数 V が支配的なので、SDF ブロックを Schur で消去してから pose を解く。
- 収束判定は \(\Vert \Delta \xi \Vert + \alpha \Vert \Delta \phi \Vert < \epsilon\)。

## 6. CUDA 並列化プラン

レイアウト:

| 構造 | デバイスメモリ | アクセスパターン |
|------|----------------|------------------|
| pose array \(T_t\) | `float[K][6]` (SE(2) 6 entry: cos, sin, x, y, …) | broadcast 多い → constant memory に流すか |
| scan point cloud | per-scan SoA `float* px, py` + offset table | 残差計算で連続 read |
| TSDF grid | `float* phi`, `float* w`, key = (ix, iy) → linear | 散在 read。texture memory で bilinear ハードウェア補間活用 |
| residual / J 行 | `float* r`, `float* J_pose[3]`, `int* sdf_idx[4]`, `float* sdf_w[4]` | per-point に 1 行 |

kernel 群:

1. `kernel_residual_and_jacobian` — 点単位、N_points スレッド。
2. `kernel_reduce_pose_block` — 各 pose ブロックの 3x3 / 3x1 を block-level reduction。
3. `kernel_solve_pose` — cuSOLVER LDLT (K が小さいので host 側でも可)。
4. `kernel_update_sdf` — SDF 同時時のみ。voxel 並列で diagonal block を構築。
5. `kernel_apply_increments` — pose / sdf を更新。

## 7. Map 表現

- 第一段: 単一の dense 2D grid (e.g. 200 m × 200 m × 0.05 m = 16M セル ≈ 64 MB の float)。
- ウィンドウから外れた領域は frozen 化して dense static map に格納。アクティブ領域はテクスチャ memory にコピー。
- 第二段で動的にハッシュ voxel に変更する余地は残すが、2D なら dense で十分。

## 8. 段階的ロードマップ

- **P0 (1 週目)**: 既存 Python pipeline で **CPU 実装の scan-to-SDF 最適化** を 1 scan 単位で書く（pose 3 dof のみ）。リファレンス実装。
- **P1**: P0 を K=10 のウィンドウに拡張、CPU で動かす。motion prior と marginalize-as-prior。【done】
- **P1.5**: `ScanBaEngine` を `slamx replay` に統合。online ローカル sliding TSDF で実 bag 追従。【done】
- **P2**: CUDA 移植。kernel_residual_and_jacobian + 3K×3K Cholesky まで GPU。【一部done: cupy 版 data-block (`scan_ba/cuda.py`) が CPU と一致】
- **P3**: SDF を変数化、Schur 込みで joint BA。
- **P4**: 厳密 marginalization (Schur で先頭 pose を消し、隣接 pose と SDF 境界に prior 残す) と loop closure 連動。

### P2 ベンチ所見 (2026-05-24, RTX 4070 Ti SUPER, cupy 14.1, CUDA 12.0)

per-scan data-block (sample+Jacobian+JtWJ/JtWr) を cupy で実装、TSDF 常駐・30反復平均:

| 点数 N | CPU (numpy) | GPU (cupy) | speedup |
|--------|-------------|------------|---------|
| 2,000   | 0.38 ms | 3.01 ms | 0.1x |
| 20,000  | 2.21 ms | 2.74 ms | 0.8x |
| 100,000 | 11.66 ms | 3.00 ms | 3.9x |

- **2D の 1 scan 規模 (~2000 点) では GPU が遅い**。原因は per-call の host 同期 (`n_valid`/`cost` の `.get()`) と小さい reduction のカーネル launch overhead。
- GPU が勝つのは 100k 点級。現状の素朴 cupy 移植は「大きい問題」向け。
- **示唆**: 実用 2D scan-BA で GPU を活かすには (a) ウィンドウ全 scan の点を 1 カーネルにバッチ、(b) LM 反復中の host 同期を排除し全反復を on-device 化、(c) sample+Jacobian+reduction を 1 つの fused custom kernel に。CUDA_PATH は `/usr` (cupy JIT が headers を要求、`scan_ba/cuda.py` が自動 setdefault)。
- CPU フォールバックは維持。`cuda.is_available()` 偽なら自動 skip。

## 9. オープン問題 / リスク

- **SDF の初期化**: スキャン到着初期は SDF がスカスカで残差が立たない。最初の \(K_0\) scan は scan-matching ベースで前段 estimate を作って SDF を埋める phase が要る。
- **Huber と LM の相互作用**: weight 更新は反復ごとに行うので、収束が遅くなる場合 IRLS の固定回数化を検討。
- **ウィンドウ境界のドリフト**: prior の不確かさ過小だと窓内が硬直、過大だと drift 加速。共分散の bake は実験で決める。
- **既存 benchmark との互換**: 現在の Cartographer parity / IILABS ベンチをそのまま使えるよう、`slamx` CLI レイヤは変えない方針。フロントエンド差し替えだけで eval が走るようインタフェース固定。
- **GPU 依存の混入**: CPU フォールバックを最低 P1 段階まで残し、CI で動作担保する。

## 10. 命名と影響範囲

- repo: `lidar_slam_2d` → `lidar_slam_2d_cuda` (既に rename 済)。
- パッケージ名 `slamx` は変更しない（CLI 互換維持）。
- 新モジュールは `src/slamx/cuda/` 配下を想定。
