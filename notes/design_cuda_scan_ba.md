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
- **P3**: SDF を変数化、Schur 込みで joint BA。【done: P3.0 dense → P3.1 疎 Schur → P3.2 平滑化 → P3.3 engine 配線+ATE → P3.4 GPU Schur(負) → P3.5 full-GPU assemble(交差点あり/小窓は負) → P3.6 splu→Jacobi-PCG で分解の壁撤去（solve V=40k で 49.9x、律速は assemble へ移行）→ P3.7 assemble を fused atomicAdd RawKernel に（assemble 単体 ~70x、end-to-end 1.2x、律速は PCG solve+host roundtrip へ）】
- **P4**: 厳密 marginalization (Schur で先頭 pose を消し隣接 pose に prior 残す)。【done: P4 アルゴ리즘 + P4.1 engine hot loop 採用 `use_marginalization`】
- **P-map**: 永続グローバル TSDF マップ（ループ閉じ込みで補正後 pose から再構築）。【done: `GlobalTsdfMap`, `build_global_map`】
- **P-eval**: 大規模 ATE（600 scan × 4 変種）で joint / marginalization の利得を定量化。【done: joint は大規模でも頑健に効く（−24% rmse）、厳密 marginalization は実走行で利得なし（正直な負の結果）】

### P3.0 所見 (2026-05-25): joint pose+SDF BA (CPU dense リファレンス)

`optimize_window_joint` (`scan_ba/joint.py`)。ウィンドウ点の bilinear 近傍ボクセル φ を pose と同時に変数化。点残差 `r = Σ_c w_c φ_{v_c}` の Jacobian は pose 部 (grad·∂(Tp)/∂ξ, 3) と SDF 部 (bilinear 重み w_c, 4 ボクセル)。各 active ボクセルに fold 時値 φ0 への prior を貼り自明解 φ=0 を防ぐ。full dense (3K+V) 正規方程式を直接 LM solve、refine 後の φ を `tsdf` に書き戻す。

検証 (L-room, SDF にノイズ σ=0.05 を注入した劣化マップ):

| | final cost |
|--|-----------|
| pose-only (`optimize_window`) | 0.669 |
| joint (pose+SDF) | **0.524** |

- 劣化マップ上で joint は **pose-only より低コスト** (φ も refine してデータ残差を下げる)。active ボクセル 560、pose は GT 近傍維持 (max dev 0.015 m)。
- 制約: dense (3K+V)² 直接 solve なので小ウィンドウ向けリファレンス。**P3.1**: H_φφ を Schur 消去 (SDF ブロックは data の 4×4 + prior 対角 + 平滑化の疎構造) して 3K pose 系に縮約、その後 GPU 化。**P3.2**: SDF 平滑化項を追加。

### P3.1 所見 (2026-05-25): SDF ブロックの疎 Schur 消去

`optimize_window_joint(backend="schur")`。ブロック組み立て (Hxx 3K×3K dense, Hxp 3K×V dense, H_φφ を scipy.sparse COO) に変更し、各 LM 反復で H_φφ_lm = data + (sdf_prior_info+lam)·I を `splu` で因子分解、Y = H_φφ⁻¹[H_φx | b_φ] を解いて pose 系に縮約 (S = Hxx_lm − Hxp·Y_H, dxx = S⁻¹·rhs, dφ = −(Y_b + Y_H·dxx))。dense full-solve と**完全一致** (pose/φ 差 0.0)。

| backend | 560-voxel 窓の solve | 結果 |
|---------|----------------------|------|
| dense (full 3K+V) | 26.56 s | cost 0.523798 |
| **schur (sparse)** | **0.12 s** | cost 0.523798 |

- **~220x 高速**で同一解。dense は (3K+V)³ ∝ V³ だが Schur は疎 H_φφ の因子分解 + 3K×3K の小 solve なので V に対しスケール。joint BA が実用ウィンドウサイズで回る。
- 次 (**P3.1 GPU / P3.2**): Schur の H_φφ 因子分解を GPU (cuSOLVER sparse / 反復法) に、SDF 平滑化項追加、engine への joint backend 配線。

### P3.2 所見 (2026-05-25): SDF 平滑化項

`optimize_window_joint(sdf_smooth_info=λ)` (既定 0.0=off)。隣接 active ボクセル間 (4 近傍の右/下) に残差 `r = φ_u − φ_v` を加える Laplacian 正則化。H_φφ に対角 +λ / 非対角 −λ を足す疎構造なので、P3.1 の Schur 経路にそのまま COO 追加で乗る。schur と dense は平滑化 ON でも完全一致 (φ 差 0.0)。L-room 劣化マップで refine 後 φ の総変動 (roughness): no-smooth 950.3 → λ=2.0 で 923.4 と低下。観測の薄い領域の SDF を整える。accept/reject 用 `total_cost` にも平滑化項を含めて整合。

### P3.3 所見 (2026-05-25): engine への joint 配線 + ATE 定量評価

`ScanBaEngineConfig.use_joint`（+ `joint_sdf_prior_info`/`joint_sdf_smooth_info`）で固定ラグ窓の solve を `optimize_window_joint` に差し替え。joint は CPU 専用で `use_cuda` より優先（device 常駐 GPU マップは in-place SDF refine と非互換）。`JointWindowResult.diagnostics["inliers_per_scan"]` を追加し engine の gate をそのまま流用。replay CLI 設定にも反映。窓ごとに局所マップを再 fold するため refine した φ は揮発的（その scan の registration を鋭くする効果が主）。

Cartographer backpack_2d を 300 scan replay → Cartographer 軌跡（疑似GT, 5581点）に時刻対応づけ、Umeyama 整列 ATE（`tools/eval_ate_scan_ba.py`, 258 マッチ）:

| variant | rmse [m] | mean | p50 | p90 | max |
|---------|----------|------|-----|-----|-----|
| pose_only | 0.160 | 0.123 | 0.113 | 0.189 | 0.610 |
| **joint** | **0.116** | 0.109 | 0.110 | 0.161 | **0.205** |

- joint pose+SDF BA は **RMSE を 28% 改善**（0.160→0.116）、**最悪誤差を約 1/3**（0.610→0.205）に抑制。実データで「効く」ことを数字で確認。

### P3.4 所見 (2026-05-25): Schur の GPU 化は割に合わない（正直な負の結果）

`optimize_window_joint(backend="schur_gpu")`。H_φφ を device 上で `cupyx.scipy.sparse.linalg.splu`（cuSOLVER）で因子分解し、縮約まで GPU で実行。CPU schur と**数値完全一致**（cost/pose/φ machine-eps）。だが速くならない:

- **solve は律速ではない**: 12 反復・1270 active-vox の窓を cProfile → 全 78 ms の内訳は gather 22 ms / assemble 17 ms / bilinear 15 ms。`splu` solve は数 ms 未満で上位に現れない。
- **cuSOLVER sparse LU < scipy SuperLU**: 帯状 SPD（SDF 結合）の単体ベンチで V=1k〜50k 全域 GPU が 3〜4x **遅い**（V=50k: CPU 101 ms vs GPU 319 ms）。直接疎分解は scipy SuperLU が極めて強い。
- 結論（P2.9 と同じ教訓）: joint BA の GPU 余地は線形 solve ではなく **gather/assemble のデータ項 Jacobian 蓄積**（pose-only の fused カーネル相当）。`schur_gpu` は正しく検証済みだが大規模 V 用・将来部品として温存。

### P4 所見 (2026-05-25): 厳密な sliding-window marginalization

`slide_window` の既定は「窓を出る最古 pose を強 AnchorPrior に焼き込む」近似で、その pose が持っていた情報を捨て恣意的な `info_xy/theta` に置換していた。P4 は原理的な置換 — 落とす pose を **Schur 補元で marginalize** する（`marginalize.py`）。

- この固定ラグ窓では最古 pose（pose 0）は pose 1 とのみ factor を共有（motion prior）。data 項・anchor・前回の marginal prior は pose 0 の unary。よって pose 0 を消すと pose 1 に **3×3 の `MarginalizationPrior`** が残る。
- `E(x) = ½(x−x_lin)ᵀΛ(x−x_lin) + gᵀ(x−x_lin)`。`window._evaluate` の (H += Λ, b += Λδ+g) 規約に一致。`WindowState.marg_prior` として畳み込む。
- **厳密性をテストで証明**: 実 data+motion+anchor で組んだフル窓を 1 GN ステップ解いた retained pose の増分と、pose 0 を marginalize して縮約窓を解いた増分が**完全一致**（単段・再帰 marginalization とも atol 1e-9）。線形/ガウスでは first-estimate Jacobians により厳密。
- `slide_window(marginalize=True, tsdf=...)` で再帰的に marginal prior を更新。

### P4.1 所見 (2026-05-25): marginalization を engine hot loop へ採用

`ScanBaEngine` に `use_marginalization` を追加。窓が pose を落とすたびに、その pose を Schur 消去して新しい最古 pose 上の `MarginalizationPrior` を作り（`_update_marginalization`）、強 anchor を置き換える。`_marg_prior`/`_marg_idx` を保持し再帰更新。

- 落とす pose の data 項は**そのステップのローカルマップで線形化（first-estimate Jacobians）**。次ステップでローカルマップは作り直されるが prior の Jacobian は固定 — 標準的な FEJ。
- 窓成長中（まだ pose を落とさない）は anchor のまま。最初に落とす瞬間の marginalization に元の anchor を含めるので**ゲージは保存**される。
- CPU / joint パス専用（GPU 常駐マップはホスト上に TSDF が無く線形化できないので `use_cuda` 時は無効）。joint ソルバの正規方程式・コストにも prior を配線。
- クリーンな合成走行では anchor とほぼ同一軌跡（anchor 自体が tight な pin のため）。利得は長時間・ループ閉じ込み時の情報整合性で出る想定。

### P3.5 所見 (2026-05-25): joint の gather+assemble を GPU 化（backend="gpu"）— 交差点はあるが小窓では負け

P3.4 で「solve の GPU 化は割に合わない・assemble が支配的」と分かったので、今回は **gather+assemble を丸ごと device 化**。cupy で bilinear サンプリング・scan ごとの Jacobian 縮約・`H_xphi`/`b_phi` の scatter・疎 `H_phiphi` の COO 構築、Schur solve も device（cuSOLVER）。

- **数値一致を検証**: backend="schur" と bit 単位で一致（cost ~1e-10, pose ~1e-15, phi 完全一致）。
- **正直なベンチ**（RTX 4070 Ti SUPER）: 大窓では GPU が勝つ（~40k 窓点で 1.74x、~20k で互角）。だが engine が実際に使う固定ラグ小窓（K~10, ~4–5k 点）では **GPU が ~2x 遅い**。
- **プロファイル内訳（4385 点, V=1068, /iter）**: bilinear 1.73ms / unique+searchsorted 0.37ms / scatter(16×`cp.add.at`) 1.57ms / COO 0.92ms / **splu solve 10.1ms**。→ assemble ではなく **cuSOLVER の疎 LU 分解が壁**（P3.4 と同根）。fused カーネルで scatter を潰しても solve 律速は変わらない。
- 真の梃子は「より速い疎分解」or「分解の回避」（H_phiphi は強 SDF prior で対角優勢 → CG/Jacobi 反復で splu を置換できる可能性）。opt-in (`backend="gpu"`) として温存。SDF 平滑化項は GPU パス未対応（明示的に NotImplementedError）。

### P3.6 所見 (2026-05-25): 分解の壁を撤去 — splu → Jacobi-PCG（gpu_solver="pcg"、デフォルト）

P3.5 が指した「真の梃子＝分解の回避」を実装。`H_phiphi` は SPD かつ SDF prior 下で強く対角優勢なので、**Jacobi 前処理付き CG**（`_pcg_spd_multi`、cuSPARSE spmm + 縮約のみ、因子分解なし）で `splu`（cuSOLVER 疎 LU）を置換。複数 RHS（3K+1 列）を列方向にベクトル化して一括反復、全列の相対残差 < tol まで回す。`backend="gpu"` の既定を `gpu_solver="pcg"` に（`"splu"` は比較用に温存）。

- **数値一致を保証**: PCG は厳密解に収束 — `test_joint_gpu_pcg_matches_splu` で splu と一致（cost <1e-7, pose 1e-7, phi 1e-5）、`test_joint_full_gpu_matches_cpu` 経由で CPU schur とも一致。マイクロベンチでは splu との最大差 ~1e-13。
- **分離ベンチ（solve のみ, m=16 RHS, RTX 4070 Ti SUPER, `tools/bench_joint_gpu_solver.py` 系）**: splu は V に対し線形〜超線形（V=1k:11ms → 5k:62ms → 10k:131ms → 40k:539ms）、**PCG はほぼ一定 ~8–11ms**（対角優勢で反復数が V に依らない）。speedup V=1k:1.39x → 5k:7.4x → 10k:16x → **40k:49.9x**。P3.5 で「splu が ~10ms/iter で壁」と言った数値は PCG では消える。
- **end-to-end（full joint GPU solve, K=5, 12 iters）**: V_active>~1000 で勝ち始め 1.08–1.14x、~700 以下では splu の安い分解が勝つ。end-to-end の利得が薄いのは gather+assemble が支配するため — **律速が分解から assemble へ完全に移った**（P2.9/P3.4 以来の一貫した教訓「線形ソルブは律速ではない」を最終的に裏取り）。大窓ほど効く（P3.5 で GPU が勝ち始めた V≥10k 域では solve 131→8ms）。
- 結論: 文書化していた「最後の真の梃子」を消化。分解の壁は撤去済み。残る joint GPU の梃子は assemble の fused カーネル化（→ P3.7 で実施）。

### P3.7 所見 (2026-05-25): assemble を 1 本の atomicAdd RawKernel に融合（gpu_assemble="fused"、デフォルト）

P3.6 が「律速は assemble へ移った」と指した点を実測で割り、潰した。**まず内訳プロファイル**（`tools/prof_joint_assemble.py`、1 LM iter を sub-step ごとに同期計測, V~1k–1.9k）:

| sub-step | ms | 備考 |
|--|--|--|
| bilinear gather | ~1.7 | |
| pose-block reduce (Hxx,bx) | ~1.6–2.2 | 6 本の bincount |
| Hxp+bp scatter | ~1.5–1.6 | **16 本の `cp.add.at`**（各々 sorted/atomic pass） |
| H_phiphi 三重項構築 | ~0.85 | |
| COO→CSR coalesce | ~1.4 | |
| **PCG solve** | **~5.6–7.0** | **単一で最大**（V=1.4k で 11 反復、毎反復 D2H 同期） |

→ **どこか 1 箇所が compute で支配的なのではなく、全面的に launch/latency-bound**（各 sub-step ~1.5ms はカーネル起動オーバヘッド、点数は ~3k と小さい）。P3.6 の「assemble が律速」は半分正しく、正確には「PCG solve が単一最大、assemble は複数 launch の合算で同程度」。

**実装**: pose 3x3 ブロック・b_x・H_xphi・b_phi を 1 本の `cp.RawKernel`（`_fused_assemble_gpu`、float64 atomicAdd, 1 thread/point）で構築。6 本の bincount + 16 本の `cp.add.at` を **1 launch** に置換。H_phiphi 三重項と CSR coalesce は両 path 共通で残す（PCG が CSR spmm を要するため）。

- **assemble サブステップ単体**: poseblk+scatter 3.1–3.7ms → **fused 0.05ms（~60–72x）**。
- **end-to-end（full joint GPU solve, K=5, 12 iters, pcg）**: vectorized 250–263ms → **fused 179–215ms（1.20–1.23x）**。残りは PCG solve（~6ms/iter）+ 毎反復の host↔device pose roundtrip と trial-cost 再評価（data_cost の再 gather）が支配。
- **数値**: float64 atomicAdd は和の順序を変えるため bit-exact ではなく丸め誤差レベル一致。`test_joint_gpu_fused_matches_vectorized` で vectorized と一致（cost <1e-7, pose 1e-7, phi 1e-5）、`test_joint_full_gpu_matches_cpu`（cost 1e-6, phi 1e-4）も fused 既定で pass。全 135 passed。
- **次の梃子（P3.8 候補, 未実施）**: 律速は PCG solve（11 反復 × 〔spmm + 数個の縮約 + 毎反復同期〕で latency-bound）と毎反復の host roundtrip/trial 再評価。PCG の収束判定を数反復ごとに（毎反復 D2H 同期を削減）、または LM の trial 評価を on-device 化して host roundtrip を畳むのが次の一手。ただし固定ラグ小窓では CPU schur で既に十分速く、ROI は逓減（正直な注記）。

### P-map 所見 (2026-05-25): 永続グローバル TSDF マップ（ループ閉じ込み整合）

トラッカは意図的に**毎スキャン作り直す鮮明なローカルサブマップ**で走る（永続蓄積はロボット移動でボケて lag を生む）。そのため全走行の一貫マップが今まで存在しなかった。`GlobalTsdfMap` がその欠落物 — 受理スキャンを現 pose で畳み込む大きな永続 TSDF を**トラッキングとは分離**して保持（マップ構築がトラッキングに feedback しない）。

- ループ閉じ込みで軌跡が補正されたら、**補正後 pose から全スキャンを再畳み込み**（`rebuild`）。online の逐次畳み込みだけだと閉じ込み前のドリフトが焼き込まれる。
- `to_occupancy_u8`: TSDF を ROS 流の 8bit occupancy（254=free / 0=occupied / 205=unknown）に描画。CLI が `global_map.pgm/yaml` を出力（`build_global_map`）。`finalize_global_map` は最終最適化 pose から再構築。

### P-eval 所見 (2026-05-25): 大規模 ATE（600 scan × 4 変種）

P3.3 の 300 scan を倍に伸ばし、joint と marginalization の利得を大規模で検証。Cartographer backpack_2d を 600 scan replay → 疑似GT（5581点）へ時刻対応づけ、Umeyama 整列 ATE（`tools/eval_ate_scan_ba.py --max-scans 600 --variants pose_only marg joint joint_marg`、558 マッチ）:

| variant | rmse [m] | mean | p50 | p90 | max | ms/scan |
|---------|----------|------|-----|-----|-----|---------|
| pose_only | 0.184 | 0.165 | 0.147 | 0.246 | 0.640 | 965 |
| marg | 0.198 | 0.176 | 0.160 | 0.253 | 0.705 | 907 |
| **joint** | **0.139** | 0.121 | 0.107 | 0.237 | **0.318** | 1133 |
| joint_marg | 0.148 | 0.130 | 0.117 | 0.251 | 0.333 | 1104 |

- **joint pose+SDF BA は大規模でも頑健に効く**: pose_only 比で **rmse −24%**（0.184→0.139）、**最悪誤差を約半分**（0.640→0.318）。300 scan の −28% / 最悪 1/3 と整合し、ウィンドウ内で SDF を同時 refine して registration を鋭くする効果がスケールしても持続することを確認。
- **厳密 marginalization は実走行で利得なし（正直な負の結果）**: pose_only→marg は rmse +8%（0.184→0.198）・最悪 +10%、joint→joint_marg も rmse +6%（0.139→0.148）と、両軸でわずかに**悪化**。FEJ により理論的には情報整合性で優れるはずだが、この実バッグでは (a) 既存の強 anchor ヒューリスティックが既にほぼ最適、(b) FEJ がドリフト走行の初期線形化誤差を prior に焼き込んで硬直、(c) 固定ラグ窓が短く整合性の利得が小さい一方 FEJ の硬さが勝つ、が重なったと解釈。**安価な anchor が実データでは exact marginalization に勝つ** — 実装は正しい（テストで bit 一致を証明済み）が、この問題設定では使う動機が薄いという知見。
- **per-scan コストは走行長でほぼ一定**: 600 scan で 907〜1133 ms/scan と、120 scan スモーク（855 ms/scan）から約 +13% に留まる。ループ閉じのポーズグラフ最適化（O(n)）と履歴増大による緩やかな成長で、破綻的な増加はない。joint は SDF 同時最適化分で pose_only 比 +17% 程度の上乗せ。

### P-quality 所見 (2026-05-25): pose_only のコールドスタート誤差は一回限りの bootstrap 再最適化では消せない（正直な負の結果）

joint が pose_only に勝つ源泉を切り分けるための調査。200 scan の ATE では pose_only rmse 0.183 / max 0.615 に対し **joint rmse 0.076 / max 0.165（−58%）** と差が大きく、誤差プロファイルを見ると pose_only の誤差は**最初の ~20 フレーム（疎な初期マップに対する registration）に集中**＝コールドスタート支配。

- **仮説**: 初期ウィンドウ（n=window_size 到達時点）を一度だけ強 anchor + 全 scan で再最適化すれば、初期 pose がスパースマップの局所解から脱出して joint 級の ATE を回収できるのでは → `engine.cfg.bootstrap_refine` + `_bootstrap_refine()` を試作。
- **結果はすべて完全な no-op（ATE が bit 一致）**。2 通り試した:
  1. pose-only ウィンドウソルバで再最適化 → `refine=False/True` で rmse 0.183/max 0.615 がビット一致。
  2. joint ソルバ（マップも同時 refine）で再最適化 → pose_only でも joint でも `refine=False/True` がビット一致（pose_only 0.183、joint 0.076 のまま不変）。
- **根本原因**: (a) ウィンドウ成長中に pose は既に局所最適に達しており、同じ目的関数の再最適化は何も動かさない。(b) トラッカは P-map の方針どおり**毎スキャン ローカルサブマップを作り直す**ため、bootstrap で初期マップを幾ら refine しても次ステップで破棄される — pose_only のトラッキングマップは ephemeral。よって「マップ品質を上げてコールドスタートを救う」筋は、一回限りの介入では原理的に効かない。
- **結論 / 推奨**: コールドスタート誤差はポーズ最適化不足ではなく**初期マップ品質**に起因し、それを解けるのは **joint の「窓内でマップと pose を毎反復 同時最適化」だけ**（bootstrap の一発ではなく継続的な co-refinement が必要）。安価な再最適化での緩和は不可。実データで精度が要るなら `use_joint=True` を使う、が素直な処方箋。試作した engine.py の `bootstrap_refine` 一式は revert（出荷しない）。

### P-loop 所見 (2026-05-26): ループ閉じ込みのロバスト化 — 辺重み + ロバストカーネル

既存のループ閉じ込みは「距離ベース候補 → サブマップ TSDF へ単発 align → inlier/rms/corr の 3 ゲート → エッジ追加 → 全体 pose-graph solve」。**最適化側が無防備**だった: `PoseGraph` の全辺が等重みの素の L2 で解かれ（情報量も重みも無し）、ロバストカーネルも無い。検出ゲートをすり抜けた誤検出ループ辺が 1 本でも入ると、軌跡全体が引きずられて壊れる（古典的な単発 false-positive 問題）。

- **実装**: `Edge.weight`（辺ごとの sqrt-information、既定 1.0）と `PoseGraphConfig.robust_loss`/`robust_f_scale` を追加。`optimize()` で残差・ヤコビアンに辺重みを左から掛け（`(E,3)` を行優先で平坦化し `np.repeat(weights,3)` でブロードキャスト）、`scipy.least_squares` に `loss`/`f_scale` を渡す。**既定は `weight=1.0` + `loss="linear"` で従来と bit 一致**（後方互換、既存テストそのまま green）。
- **engine 配線**: `ScanBaEngineConfig.loop_robust_loss="cauchy"` / `loop_robust_f_scale=0.5` を既定とし `PoseGraph` へ渡す。**ループ辺は `weight=inlier_ratio`** で重み付け（ゲートぎりぎり 0.4 の弱い一致は信頼できるオドメトリ 1.0 より効きを下げる）、オドメトリ辺は 1.0。二段構え＝「弱い辺は最初から軽く、それでも不整合な辺はカーネルがクリップ」。
- **f_scale=0.5 の根拠**: 受理ループ辺の残差は rms ゲート ≤0.3 m。f_scale=0.5 なら良辺（rms≤0.3）は cauchy のほぼ二次域（z≤0.36, weight≥0.73）に残しつつ、メートル級にずれた粗大外れ値だけを強く減衰。0.3 まで下げると良辺も削り始めるので不可。
- **定量（単発の粗大偽辺ストレステスト, 6 pose ループ + 完全オドメトリ + 真ループ辺、偽辺が pose3≡pose0 を主張）**: 最大軌跡偏差 = **素の L2 で 1.87 m → 重み 0.4 + cauchy(0.5) で 0.31 m（約 6 倍改善）**。f_scale=0.3 なら 0.19 m。π フリップの極端な単発外れ値なので「真値完全復元」までは行かないが、軌跡破壊は確実に防げる。test_pose_graph に `test_robust_loss_rejects_false_loop_edge` / `test_edge_weight_down_weights_conflicting_edge` を追加（137 passed）。

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

### P2.5 所見 (2026-05-25): ウィンドウ LM を on-device 化

(a)+(b) を `optimize_window_cuda` (`scan_ba/cuda.py`) として実装。TSDF・全 scan 点 (concat + segment id)・poses・3K×3K 正規方程式を device 常駐にし、各反復は全点 1 パスで評価 (per-point の H/b 寄与を `bincount` で scan ごとに segment-reduce → block-diagonal を組み立て → `cp.linalg.solve`)。host へ渡すのは accept/reject の cost スカラと最終 poses のみ。CPU `optimize_window` と数値完全一致 (poses 差 ≤ 4e-16、反復数・cost・inlier 一致)。

K=10、25 反復フル solve のベンチ (`tools/bench_scan_ba_cuda.py`):

| window 点数 | CPU (numpy) | GPU (cupy) | speedup |
|-------------|-------------|------------|---------|
| 2,000    | 10.8 ms | 41.4 ms | 0.26x |
| 10,000   | 16.5 ms | 41.9 ms | 0.40x |
| 50,000   | 177 ms  | 178 ms  | 0.99x |
| 200,000  | 550 ms  | 209 ms  | 2.64x |

- 素朴 P2 (per-block host sync) では全域で負けていたのに対し、**損益分岐が ~50k 点へ低下、200k で 2.64x**。host 同期は反復あたり O(1) に削減済み。
- ただし実用 2D ウィンドウ (K=10 × 200〜1000 点 = 2〜10k 点) はまだ分岐点以下で CPU 有利。残るボトルネックは**反復あたり数百回の小カーネル launch** (bincount×11・小 slice 演算・solve)。
- **次の一手 (c)**: warp+sample+Jacobian+per-block reduction を 1 つの fused RawKernel に畳み、反復あたりの launch を数回に。これで分岐点を実用サイズまで下げるのが P2.5→P2.9 の目標。

### P2.5c 所見 (2026-05-25): fused データ項カーネル

(c) を `_DATA_KERNEL_SRC` / `optimize_window_cuda(backend="fused")` として実装。1 点 1 スレッドで warp→bilinear sample→Jacobian→Huber を計算し、scan ごとの累算器 (K×11: 上三角 H 6 + b 3 + cost + inlier) へ float64 `atomicAdd`。bincount×11＋多数の elementwise 演算が **evaluate あたり 1 カーネル launch** に畳まれる。CPU と数値完全一致 (poses 差 ≤ 3e-17、反復・cost・inlier 一致)。

| window 点数 | CPU | bincount | fused | fused speedup |
|-------------|------|----------|-------|---------------|
| 2,000    | 13.6 ms | 65.6 ms | 29.5 ms | 0.46x |
| 10,000   | 20.0 ms | 60.2 ms | 26.8 ms | 0.75x |
| 50,000   | 204 ms  | 202 ms  | 88.5 ms | 2.30x |
| 200,000  | 566 ms  | 221 ms  | 84.6 ms | 6.69x |

- fused は bincount の **約 2.2x**、損益分岐が ~50k → ~15-20k 点へ低下、200k で 6.69x。
- fused 時間は 2k で 29 ms・200k で 85 ms とほぼ一定 = データ項はもはやボトルネックでなく、**反復あたりの固定オーバーヘッド律速**。次段 P2.9 で内訳をプロファイルして潰す。
- CPU と bincount フォールバックは維持 (`backend=` で選択)。

### P2.9 所見 (2026-05-25): 反復オーバーヘッドのプロファイルと motion prior ベクトル化

evaluate あたりの内訳を実測 (K=10, 2k 点):

| 段 | 時間 |
|----|------|
| fused kernel + acc | 0.016 ms |
| H/b 組み立て | 0.247 ms |
| **motion prior Python ループ (K-1 反復)** | **1.764 ms** |
| 30×30 `cp.linalg.solve` | 0.136 ms |
| host cost sync | 0.051 ms |

ボトルネックは solve でも CUDA graph 不在でもなく、**motion prior を 1 prior ずつ回す Python ループ (~50 個の極小カーネル launch)** だった。block-tridiagonal 構造を index 配列で一括 scatter する形にベクトル化 (対角ブロックは node ごとに左右 prior の W を加算、off-diagonal は distinct な (i,i+1) ペアへ一括) → ループ消滅。CPU と数値完全一致のまま:

| window 点数 | CPU | bincount | fused | fused speedup |
|-------------|------|----------|-------|---------------|
| 2,000    | 11.2 ms | 33.4 ms | 10.4 ms | **1.07x** |
| 10,000   | 16.6 ms | 32.6 ms |  9.8 ms | **1.70x** |
| 50,000   | 180 ms  | 141 ms  |  38 ms | 4.74x |
| 200,000  | 571 ms  | 176 ms  |  40 ms | 14.22x |

- **損益分岐が 2k 点未満へ低下 = 実用 2D ウィンドウ (K=10 × 200〜1000 点) で GPU が CPU を上回る**。fused 時間は 10〜40 ms とほぼ一定で良く償却。
- P2 (素朴 per-block port, 全域で負け) → P2.5 (on-device, ~50k 分岐) → P2.5c (fused kernel, ~15-20k) → P2.9 (assembly ベクトル化, <2k) と段階的に分岐点を下げ切った。
- 教訓: 「GPU が遅い」の主因は素朴に思える small-op の Python ループ launch であり、custom kernel より先にプロファイルすべきだった。

### engine 統合所見 (2026-05-25)

`ScanBaEngineConfig.use_cuda` を追加し、`handle_scan` のウィンドウ solve を `optimize_window_cuda` に差し替え可能に (CLI: `slam.scan_ba.use_cuda`、cupy/CUDA 不在なら CPU フォールバック)。実 backpack 80 scans の end-to-end:

| | 時間 | 軌跡 |
|--|------|------|
| CPU | 70.6 s | — |
| CUDA | 63.1 s | CPU と 7e-16 m 一致 |

- **end-to-end は ~1.1x に留まる**。per-scan コストはウィンドウ solve でなく **CPU 側の `_rebuild_local_map` (直近 20 scans を 1.44M セル TSDF へ畳む) + preprocess** が支配的で、solve の 1.07〜1.7x が全体の ~10% にしか効かない。
- **次のボトルネック = TSDF 再構築 (`update_tsdf_from_scan`) の GPU 化**。ローカルマップ折り込みも device 常駐にすれば、毎スキャンの TSDF upload (約 23 MB) も不要になり end-to-end の speedup が出る。→ 次節で実施。

### device 常駐ローカルマップ所見 (2026-05-25): end-to-end 6.7x

`update_tsdf_from_scan` を cupy 移植 (`update_tsdf_from_scan_cuda`, `cp.add.at` で散布、float32 store 踏襲 → CPU と bit 一致)。engine が `use_cuda` 時に phi/weight を float32 device 配列として常駐させ、毎スキャンの `_rebuild_local_map` (直近 `map_window` scan の折り込み) も `optimize_window_cuda` も**同じ device TSDF を共有**。host upload は消滅 (bootstrap の最初の 2 scan だけ単発 align 用に device→host ダウンロード)。実 backpack 80 scans:

| | 時間 | ms/scan | 軌跡 |
|--|------|---------|------|
| CPU | 68.1 s | 851 | — |
| CUDA (device 常駐) | **10.1 s** | **126** | CPU と 7e-16 m 一致 |

- **end-to-end 6.74x**。solve だけ GPU の ~1.1x から、fold も device 常駐化して跳ね上がった。CPU の per-scan を支配していた `_rebuild_local_map` (20 fold × 1.44M セル全配列演算 + `np.add.at`) が GPU 散布に置き換わり、かつ毎スキャンの 23 MB upload も消えた。
- loop closure / bootstrap の単発 align は CPU のまま (低頻度)。`cuda.is_available()` 偽なら自動 CPU フォールバック。
- 残: loop closure 検証 TSDF も device 化、SDF 同時最適化 (P3)。

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
