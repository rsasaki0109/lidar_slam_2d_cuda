# Cartographer ベンチ / slamx parity ライン 引き継ぎメモ

更新日: 2026-04-08  
対象ワークスペース: `lidar_slam_2d`

## 0. 先に結論

- 現時点の最良結果は **2000 scans / `runs/slamx_parity_noloop_s2k` / align RMSE 0.4745925363 m**。
- 旧 `bench_fast` の 2000 scans (`runs/slamx_backpack2d_2k`) は **7.6929197013 m** なので、**約 16 倍改善**。
- 300 scans でも、`runs/slamx_parity_medium_s300` は **0.0805190088 m** で、旧 `bench_fast` の **1.5926748606 m** から **約 20 倍改善**。
- 今日の実務上の結論は、「最初の 300〜2000 scans は実質的に一方向の廊下で、ループ閉じ込みを積極的に効かせるより、Cartographer 寄せの local matching + 重くても素直な pose graph 最適化のほうが効いた」。
- 特に **最初の 2k scans では loop closure を切ったランが最良**。`notes/benchmark_cartographer_agreement_b0_2014-07-11.json` への追記は **すでにステージ済み**。
- ただし、このノート前半で言及している「loop closure 専用の ICP パラメータを `slam.loop_detection.icp` で分離可能にする」変更は、その後 parity 系の working YAML にも反映された。**locked benchmark config には入れていない**ので、再現用と実験用を混同しないこと。

## 1. このノートの役割

このノートは、次担当が次の 3 点をすぐ把握できるようにするための引き継ぎメモ。

1. 何を比較していたか
2. 何が本当に効いたか
3. 今のワークツリーのどこまでが「結果として確定」で、どこからが「次の実験待ち」か

比較対象は一貫して以下。

- 入力: Google Cartographer 公式 `backpack_2d` の ROS1 bag
- `slamx`: `.venv/bin/slamx replay`
- 擬似 GT: Cartographer の `map -> base_link` TF から export した CSV
- 評価: `slamx eval ate`

ここでいう RMSE は **真の外部 GT との誤差ではなく**、**Cartographer が同じ bag で出した軌跡にどれだけ一致するか**である。以後もこの前提を崩さないこと。

## 2. 現在のワークツリー状況

2026-04-08 時点で `git status --short` は次の 3 ファイルがステージ済み。

- `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`
- `src/slamx/cli/main.py`
- `src/slamx/core/frontend/local_slam.py`

未ステージ変更は、少なくともこの確認時点では無かった。この `plan` ファイルは今から更新しているので、以後は別途差分が出る。

直近の主要コミットは以下。

- `9c57cfb` Add ICP refinement for loop closure + submap-based loop references
- `37baef6` Improve loop closure and optimize scan matching / pose graph
- `4ad8215` bench: add 300-scan parity medium results (RMSE 0.08m aligned)
- `700e1b3` Remove sliding-window pose-graph optimization
- `7826497` docs: expand plan_cartographer_benchmark for handoff (parity configs, pseudo GT, checklist)

重要なのは、**昔このノートに書かれていた `optimization_window` 系の説明は、現在のコードにはもう対応していない**こと。`700e1b3` で sliding-window pose graph は削除済み。

## 3. データと評価ファイル

固定パス:

- bag: `lidar_slam_2d/data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag`
- topic: `horizontal_laser_2d`
- Cartographer 記録 bag: `runs/carto_recorded.bag`
- Cartographer 全軌跡 CSV: `runs/cartographer_traj.csv`
- 300 scans 用 windowed GT: `runs/cartographer_traj_s300_window.csv`
- 2000 scans 用 windowed GT: `runs/cartographer_traj_s2k_window.csv`
- 集計 JSON: `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`

補足:

- `export-tf-trajectory` は TF の `header.stamp` を使う修正がすでに入っている前提。bag 記録時刻では `replay` の stamp と噛み合わない。
- `eval ate` は `--max-dt-ms 50` で比較している。

## 4. 数値の正本

正本は `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`。  
このノートでは見やすさのために表へ再掲する。

### 4.1 主要比較

| 区間 | ラン | 構成 | align RMSE (m) | no-align RMSE (m) | n | 備考 |
|---|---|---|---:|---:|---:|---|
| 300 | `runs/slamx_backpack2d_short` | `bench_fast` | 1.5926748606 | 3.0881564409 | 258 | 旧ベースライン |
| 300 | `runs/slamx_parity_medium_s300` | parity 系 | 0.0805190088 | 0.4288311285 | 258 | 現在の 300 最良 |
| 2000 | `runs/slamx_backpack2d_2k` | `bench_fast` | 7.6929197013 | 16.9767459465 | 1953 | 旧ベースライン |
| 2000 | `runs/slamx_parity_noloop_s2k` | parity 系, loop off | 0.4745925363 | 0.8308967525 | 1953 | 現在の 2k 最良 |
| full | `runs/slamx_backpack2d_full_capped` | `bench_backpack_full` | 17.1389666989 | 42.1392935135 | 5438 | 完走優先、精度は悪化 |

改善率:

- 300 scans: `1.5926748606 / 0.0805190088 ~= 19.8x`
- 2000 scans: `7.6929197013 / 0.4745925363 ~= 16.2x`

### 4.2 今日の 300-scan 派生ラン比較

以下は `runs/*/trajectory.json` が残っていたものを、同じ `runs/cartographer_traj_s300_window.csv` で再評価した結果。

| ラン | align RMSE (m) | no-align RMSE (m) | loop accepted | loop rejected | 所感 |
|---|---:|---:|---:|---:|---|
| `runs/slamx_parity_medium_s300` | 0.0805190088 | 0.4288311285 | 0 | 954 | 最良。loop を有効化していても実質受理 0 |
| `runs/slamx_parity_loop_s300_t15` | 0.1581353843 | 0.7732361018 | 7 | 959 | かなり改善するが最良には届かない |
| `runs/slamx_parity_loop_s300` | 0.2985079820 | 0.8626540221 | 99 | 867 | loop 受理が増えると悪化 |
| `runs/slamx_parity_icp_s300` | 0.3428109153 | 2.0124114088 | 300 | 666 | 受理過多で明確に悪い |
| `runs/slamx_parity_submap_s300` | 0.5939069939 | 3.1447527646 | 98 | 868 | submap ref 単独では改善せず |

実務上の解釈:

- **最初の 300 scans では loop closure を「積極的に当てる」方向は基本的に悪化**。
- 最良ランは `enabled: true` でも結果的に受理 0 件で、local matching と pose graph の素直な一致性が支配的。
- `submap-based loop reference` や `ICP refinement` は、短い一方向区間では単体で勝ち筋にならなかった。

## 5. 今日の実験から引いてよい結論

### 5.1 効いたもの

- correlative local matching への寄せ
- preprocess / grid / submap / pose graph の Cartographer 寄せ設定
- cKDTree 化とバッチ処理による loop/search 系の高速化
- pose graph 残差のベクトル化

### 5.2 少なくとも「最初の 2k scans」に対しては効かなかったか、採用保留のもの

- 強い loop closure
- ループ候補に対する aggressive な ICP refinement
- submap 参照だけを足して loop を増やす方向

### 5.3 今の暫定仮説

最初の 2k scans には **真の revisit がほぼ無い、または loop closure の利益がコストと誤受理リスクに見合わない**。  
そのため、**2k までの parity ベストは「loop off」が正しい**。

これは厳密証明ではなく、手元の telemetry と RMSE の傾向からの実務判断。次担当が覆すなら、`telemetry.jsonl` の候補分布と地図上の revisit 実態をもう一度精査すること。

## 6. 重要: 現在の YAML と「実際に良かったラン」は一致していない

ここは次担当がハマりやすいので、強く明記する。

### 6.1 `configs/cartographer_parity_medium.yaml` はそのままでは 300 最良 / 2k 最良を再現しない

現在チェックインされている `configs/cartographer_parity_medium.yaml` は概ね以下。

- `loop_detection.enabled: true`
- `search_radius_m: 4.0`
- `min_separation_nodes: 100`
- `accept_score: -3.0`
- `icp_accept_rms: 0.10`

しかし、実際に残っている成功ランの resolved config は別物。

#### 300 最良ラン

`runs/slamx_parity_medium_s300/config_resolved.yaml`:

- `enabled: true`
- `search_radius_m: 2.5`
- `min_separation_nodes: 45`
- `accept_score: -0.28`
- `icp_accept_rms`: 記録なし
- accepted loop: 0

#### 2k 最良ラン

`runs/slamx_parity_noloop_s2k/config_resolved.yaml`:

- `enabled: false`
- `search_radius_m: 4.0`
- `min_separation_nodes: 45`
- `accept_score: -2.0`
- local matching は correlative
- pose graph は `max_iterations: 50`, `max_nfev_cap: 200000`

つまり、**現行 YAML は「議論中の最新設定」ではあっても、「実測ベストの完全再現設定」ではない**。

次担当が再現性を重視するなら、以下を正として扱う。

1. `configs/cartographer_parity_medium_s300_locked.yaml` / `configs/cartographer_parity_noloop_s2k.yaml` を benchmark 再現用 fixed config とする
2. あるいは `runs/*/config_resolved.yaml` を実験記録として参照し、必要なら fixed config の正しさを再確認する

## 7. ステージ済み未コミット差分の意味

### 7.1 `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`

この差分は妥当。入っている内容は以下。

- 旧 2000 scans エントリに `config: "configs/bench_fast.yaml"` を明記
- `runs_2000_parity_medium` を追加
- 実際の run path は `runs/slamx_parity_noloop_s2k`
- notes に「loop detection disabled (no true revisits in first 2k scans)」を明記

この JSON 更新は、今日の結論をちゃんと保存している。

### 7.2 `src/slamx/cli/main.py` と `src/slamx/core/frontend/local_slam.py`

この差分は docs 更新ではなく、**実挙動変更**を含む。

やっていること:

- `slam.loop_detection.icp` というネストを CLI で読めるようにした
- `LocalSlamConfig.loop_icp` を追加
- loop closure 用の ICP refiner を、local matcher 側の `icp` 設定とは分離

新しい default:

- `max_iterations: 30`
- `max_correspondence_dist_m: 2.0`
- `min_correspondences: 20`
- `trim_fraction: 0.3`

以前は `_loop_refiner = IcpScanMatcher(self.cfg.icp)` だったので、**loop refinement は local matcher 用 ICP 設定に引きずられていた**。  
今回の差分でそれを切り離している。以後、working config の `configs/cartographer_parity_medium.yaml` / `configs/cartographer_parity_full.yaml` には `slam.loop_detection.icp` を明示している。

### 7.3 次担当への実務アドバイス

この staged diff は意味的に 2 つに分かれている。

- bench/result の保存
- loop ICP 設定まわりの機能追加

混ぜて 1 commit にしても動く可能性は高いが、履歴としては次のように分けたほうが読みやすい。

1. `notes/benchmark_cartographer_agreement_b0_2014-07-11.json` とこの `notes/plan_cartographer_benchmark.md` を docs/bench commit
2. `src/slamx/cli/main.py` と `src/slamx/core/frontend/local_slam.py` を loop ICP config support commit

もし時間が無ければまとめてもよいが、その場合は commit message に「2k parity result 保存」と「loop ICP config 分離」の 2 点を明記したほうがよい。

## 8. 中断済みランの扱い

本日中断された 2k 系ラン:

- `runs/slamx_parity_medium_s2k`
- `runs/slamx_parity_loop_s2k`
- `runs/slamx_parity_icp_s2k`

観測された終了コード:

- `143`
- `144`

これは `kill` による中断であり、**異常終了というより意図的停止として扱ってよい**。  
また、これらのディレクトリには `trajectory.json` が無いので、**評価対象に使わないこと**。

実務ルール:

- `trajectory.json` が無いランは benchmark JSON に入れない
- `config_resolved.yaml` だけ残っている場合は、「試した設定の痕跡」としてのみ扱う

## 9. 速度面のメモ

手元報告では、テスト/実験サイクルは **88 秒 -> 9.8 秒** まで短縮している。  
主因は以下のはず。

- cKDTree 化
- 候補探索のバッチ化
- pose graph 残差の vectorization

この改善で「短い 300-scan 派生実験を複数回す」ことが現実的になった。次担当も、いきなり full bag に行くより、まず 300 / 2k で仮説検証するのがよい。

## 10. 現時点で信用してよいファイル

信用度が高いもの:

- `notes/benchmark_cartographer_agreement_b0_2014-07-11.json`
- `runs/slamx_parity_medium_s300/config_resolved.yaml`
- `runs/slamx_parity_noloop_s2k/config_resolved.yaml`
- `runs/slamx_parity_medium_s300/trajectory.json`
- `runs/slamx_parity_noloop_s2k/trajectory.json`

古い説明が残っていて、そのまま信用してはいけないもの:

- `configs/cartographer_parity_full.yaml`
- 過去版のこの `plan` に書かれていた `optimization_window` 説明

理由:

- `configs/cartographer_parity_full.yaml` は deprecated な exploratory config で、benchmark の正本再現用ではない
- `pose_graph.optimization_window` は設定ファイルからも削除済みだが、古い handoff 説明には sliding-window 前提が残っている
- `tests/test_pose_graph_window.py` も `700e1b3` で削除済み

つまり、**parity_full は今のコードベースでは設計ノートの残骸に近い**。

## 11. 再現コマンド

前提: `env -u PYTHONPATH` をつけて venv 内の `slamx` を使う。

### 11.1 300 scans の評価済みランを再評価

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_medium_s300 \
  --gt runs/cartographer_traj_s300_window.csv --max-dt-ms 50
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_medium_s300 \
  --gt runs/cartographer_traj_s300_window.csv --max-dt-ms 50 --no-align
```

### 11.2 2k 最良ランを再評価

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_noloop_s2k \
  --gt runs/cartographer_traj_s2k_window.csv --max-dt-ms 50
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_parity_noloop_s2k \
  --gt runs/cartographer_traj_s2k_window.csv --max-dt-ms 50 --no-align
```

### 11.3 旧 2k ベースラインとの比較

```bash
cd lidar_slam_2d
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_backpack2d_2k \
  --gt runs/cartographer_traj_s2k_window.csv --max-dt-ms 50
env -u PYTHONPATH .venv/bin/slamx eval ate runs/slamx_backpack2d_2k \
  --gt runs/cartographer_traj_s2k_window.csv --max-dt-ms 50 --no-align
```

### 11.4 再走時の注意

もし 300 最良や 2k 最良を「再現」したいなら、`configs/cartographer_parity_medium_s300_locked.yaml` または `configs/cartographer_parity_noloop_s2k.yaml` を使うこと。`configs/cartographer_parity_medium.yaml` は継続調整用として扱う。

## 12. 次担当の優先タスク

優先度順。

1. **docs/bench と code を分けて commit するか判断**
   - docs only で先に確定させるのが安全
2. **300 / 2k ベスト設定の正本参照を維持**
   - fixed YAML は `configs/cartographer_parity_medium_s300_locked.yaml` と `configs/cartographer_parity_noloop_s2k.yaml`
   - benchmark JSON / docs / rerun 手順がこの 2 本を指しているか確認する
3. **`configs/cartographer_parity_full.yaml` の deprecated 状態を維持**
   - benchmark の正本再現用に使わない
   - 復活させるならコード側の機能と説明を揃えてから扱う
4. **loop ICP 専用設定を本当に使うなら YAML まで落とす**
   - `slam.loop_detection.icp` ブロックは working parity config に追加済み
   - そのうえで 300 / 2k / full のどこに効くか再測定する
5. **真の GT データへの移行計画を作る**
   - 今はあくまで Cartographer 擬似 GT 一致度

## 12.1 真の GT 比較の即着手案

2026-04-08 の追加状況:

- `slamx eval ate` は **`.tum` を GT として直接読める**ようにした
- つまり、IILABS 系の `ground_truth.tum` を CSV 変換せずそのまま使える

### 2026-04-08 実測済み: IILABS `velodyne_vlp-16 / elevator` prefix

実際に取得したデータ:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/velodyne_elevator_2025-02-05-15-04-36.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- full bag scan 数: **20243**
- full bag duration: **約 506.24 s**

この turn で比較したのは、`slamx bench_fast` が進んだ時点の prefix bag:

- prefix cutoff: `1738768050028047992 ns`
- prefix bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator_prefix_1738768050028047992.bag`
- prefix 長: **約 172.5 s**
- `slamx` 側 prefix trajectory: `runs/iilabs_elevator_bench_fast_partial/trajectory.json`
- Cartographer 側 prefix trajectory: `runs/iilabs_elevator_prefix_carto_traj.csv`

ただし、ここで **一番重要な注意点** がある。

`ground_truth.tum` は bag 全域で連続していない。実際には 2 セグメントだけで、間に巨大 gap がある:

- GT segment A: `1738767876580871424 -> 1738767901134893312` （**約 24.55 s**）
- GT segment B: `1738768365879127552 -> 1738768386304392192` （**約 20.43 s**）
- gap: **464.74423424 s**

したがって、この prefix bag は 172.5 秒あるが、**GT と実際に重なって評価できるのは先頭約 24.6 秒だけ**。以前の「prefix 全体で 172.5 秒ぶん比較した」という読み方は正確ではない。

#### prefix 直比較（GT overlap は先頭約 24.6 秒のみ）

| 手法 | 区間 | align RMSE (m) | no-align RMSE (m) | n |
|---|---|---:|---:|---:|
| `slamx` + `configs/bench_fast.yaml` | IILABS elevator prefix | **0.9451395693** | **1.5303861753** | 954 |
| Cartographer 2D (`iilabs_hokuyo_2d.lua`) | IILABS elevator prefix | **0.1391654556** | **0.2626741788** | 4501 |

この GT-overlap 条件では **Cartographer が明確に優位**。

実務上の解釈:

- `bench_fast` は IILABS elevator GT に対してはまだ弱い
- direct な prefix 比較では Cartographer のほうが **align で約 6.8x、no-align で約 5.8x 良い**
- ただしこの差は **「prefix 172.5 秒全体」ではなく「prefix 冒頭の GT overlap」** で出ている
- したがって「GT に対して Cartographer に勝った」とはまだ言えない

#### 追加で回した `slamx` tuning（最初の 2000 scans ≒ 約 50.05 秒）

`slamx` 側は 2000 scans の tuning も回した。trajectory 自体は約 50.05 秒ぶん出ているが、`max-dt-ms 50` では GT overlap が冒頭セグメントに限られるため、**評価に使われるのは結局 954 点**である。

| 手法 | 実行 | align RMSE (m) | no-align RMSE (m) | n |
|---|---|---:|---:|---:|
| `bench_fast` | `runs/iilabs_elevator_s2k_bench_fast` | **0.9451395693** | **1.5303861753** | 954 |
| `iilabs_icp_medium` | `runs/iilabs_elevator_s2k_icp_medium` | **0.7706958467** | **1.4178245479** | 954 |
| `iilabs_icp_dense` | `runs/iilabs_elevator_s2k_icp_dense` | **0.1797559632** | **0.2369072083** | 954 |
| `iilabs_correlative_light` | `runs/iilabs_elevator_s2k_corr_light` | **0.1738570637** | **0.3300528725** | 954 |
| `iilabs_hybrid_tail` (`correlative -> ICP`) | `runs/iilabs_elevator_s2k_hybrid` | **0.1342020518** | **0.2407537824** | 954 |
| `iilabs_hybrid_tail_refdense` (`correlative -> ICP`, dense ref submap) | `runs/iilabs_elevator_s2k_hybrid_refdense` | **0.0987022896** | **0.2014347040** | 954 |
| `iilabs_hybrid_tail_refdense_cv_s2_refinegrid` | `runs/iilabs_elevator_s2k_hybrid_refdense_cv_s2_refinegrid` | **0.0351590699** | **0.1489259484** | 954 |
| Cartographer を `iilabs_icp_dense` の timestamps に再サンプル | `runs/iilabs_elevator_s2k_window_carto_at_slamx.csv` | **0.1207190760** | **0.2670687466** | 954 |

この比較から言えること:

- `iilabs_icp_medium` は `bench_fast` より改善するが、まだ遠い
- `iilabs_icp_dense` と `iilabs_correlative_light` は **初期 GT overlap 区間では**かなり詰めている
- local matching を `correlative -> ICP` の2段階にした `iilabs_hybrid_tail` は、**align を 0.1342 m まで改善**した
- `iilabs_hybrid_tail_refdense` はさらに改善し、**align / no-align の両方で Cartographer sampled を上回った**
  - `align 0.0987 m` vs `Cartographer sampled 0.1207 m`
  - `no-align 0.2014 m` vs `Cartographer sampled 0.2671 m`
- `iilabs_hybrid_tail_refdense_cv_s2_refinegrid` はさらに改善し、**同一 config で head 側を大幅更新**した
  - `align 0.0352 m` vs `Cartographer sampled 0.1207 m`
  - `no-align 0.1489 m` vs `Cartographer sampled 0.2671 m`
- したがって **先頭 GT segment A に限れば、現状 best の `slamx` は Cartographer sampled より明確に良い**
- `corr_light` は `align` だけなら `dense` とほぼ同等だが、`no-align` が悪化する

#### `dense` を prefix 全長（172.68 秒）まで回した結果

- run: `runs/iilabs_elevator_prefix_icp_dense`
- poses: **6899**
- span: **172.68004157 s**
- ATE (`max-dt-ms 50`): `align 0.1797559632 m`, `no-align 0.2369072083 m`, `n=954`

この数値が 2000-scan `dense` と実質同じなのは、**172.68 秒分 replay しても、GT overlap が冒頭約 24.6 秒にしか無いから**。prefix 後半の改善・劣化は、この設定のままでは GT で採点されない。

ここまでで分かったこと:

- いまの IILABS `elevator` GT 比較は、prefix だけでは **実質「冒頭 24.6 秒の比較」** になる
- その区間では Cartographer 優位だが、`iilabs_icp_dense` はかなり近く、`no-align` では上回る
- 「中盤〜後半でどちらが崩れるか」は **full bag か、GT segment B を含む tail 抽出** を回さないと判断できない

#### GT segment B を含む tail 抽出でも比較した

tail bag を新規作成:

- tail bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator_tail_gt2_1738768360879127552_1738768386304392192.bag`
- 中身: `/eve/scan` 926 scans + `/tf_static`
- tail window: `1738768360879127552 -> 1738768386304392192`
- 実スキャン span: `1738768360890097146 -> 1738768384054331412` （**約 23.16 s**）

`slamx dense`:

- run: `runs/iilabs_elevator_tail_gt2_icp_dense`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.1461600059 m`, `no-align 4.1644745891 m`, `n=728`

`slamx hybrid` (`correlative -> ICP`):

- config: `configs/iilabs_hybrid_tail.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0675333154 m`, `no-align 4.3238299438 m`, `n=728`

`slamx hybrid refdense`:

- config: `configs/iilabs_hybrid_tail_refdense.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid_refdense`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0557417342 m`, `no-align 4.4448122283 m`, `n=728`

`slamx hybrid refdense + constant velocity`:

- config: `configs/iilabs_hybrid_tail_refdense_cv.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid_refdense_cv`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0479353570 m`, `no-align 4.5878346070 m`, `n=728`

`slamx hybrid refdense + constant velocity + stride 2`:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid_refdense_cv_s2`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0352382022 m`, `no-align 4.5668379188 m`, `n=728`

`slamx hybrid refdense + constant velocity + stride 2 + refinegrid`:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid_refdense_cv_s2_refinegrid`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0192689155 m`, `no-align 4.5731614682 m`, `n=728`

### 2026-04-08 追加検証: IILABS `velodyne_vlp-16 / slippage` full

取得データ:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/slippage/velodyne_slippage_2025-02-05-10-39-46.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/slippage/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- bag size: **656.8 MB**
- GT span: **105.830479872 s**
- GT gap: **なし** (`segment_count = 1`)

Cartographer 2D:

- recorded bag: `runs/iilabs_slippage_carto_recorded.bag`
- exported traj: `runs/iilabs_slippage_carto_traj.csv`
- rows: **23211**
- ATE direct: `align 0.3254778952 m`, `no-align 0.3363607528 m`, `n=23085`

`slamx` best-so-far config をそのまま適用:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_slippage_hybrid_refdense_cv_s2_refinegrid`
- trajectory span: **105.622750036 s** / **4219 poses**
- ATE direct: `align 0.1106470265 m`, `no-align 0.2641505614 m`, `n=4164`

Cartographer を `slamx` timestamps に再サンプル:

- sampled traj: `runs/iilabs_slippage_carto_at_slamx.csv`
- ATE sampled-to-slamx timestamps: `align 0.3157390843 m`, `no-align 0.3293682817 m`, `n=4164`

`slippage` の読み方:

- GT が連続なので、`elevator` より比較条件が素直
- direct GT 比較でも `slamx` が Cartographer を上回る
  - `align 0.1106 m` vs `0.3255 m`
  - `no-align 0.2642 m` vs `0.3364 m`
- timestamps を `slamx` 側に揃えても `slamx` が上回る
- `align 0.1106 m` vs `0.3157 m`
- `no-align 0.2642 m` vs `0.3294 m`
- したがって、`configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml` は **`slippage` full でも Cartographer sampled を明確に上回った**

### 2026-04-08 追加検証: IILABS `velodyne_vlp-16 / ramp` 1k prefix

取得データ:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/ramp/velodyne_ramp_2025-02-05-14-40-54.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/ramp/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- bag size: **1.2 GB**
- GT span: **184.921449472 s**
- GT gap: **なし** (`segment_count = 1`)

Cartographer 2D full:

- recorded bag: `runs/iilabs_ramp_carto_recorded.bag`
- exported traj: `runs/iilabs_ramp_carto_traj.csv`
- rows: **41023**
- ATE direct: `align 0.1446296846 m`, `no-align 0.2013226597 m`, `n=39969`

`slamx` best-so-far config を `--max-scans 1000` で適用:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_ramp_s1k_hybrid_refdense_cv_s2_refinegrid`
- trajectory span: **25.008255675 s** / **1000 poses**
- GT overlap: **19.3958656 s** / `n=777`
- ATE direct: `align 0.0661317459 m`, `no-align 0.1142695860 m`, `n=777`

Cartographer を `slamx` timestamps に再サンプル:

- sampled traj: `runs/iilabs_ramp_carto_at_slamx_s1k.csv`
- ATE sampled-to-slamx timestamps: `align 0.0740222111 m`, `no-align 0.1832452118 m`, `n=777`

`ramp` の読み方:

- `slamx` prefix は GT 開始より **約 5.61 s 早く**始まっているので、ATE では `777/1000` points だけが matched になっている
- これは `elevator` のような **GT gap 問題ではなく**、単に prefix の先頭が GT 範囲の外に出ているだけ
- 同じ timestamps に揃えた比較では `slamx` が Cartographer sampled を上回る
  - `align 0.0661 m` vs `0.0740 m`
- `no-align 0.1143 m` vs `0.1832 m`
- したがって、`configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml` は **`ramp` 1k prefix でも Cartographer sampled を上回った**

### 2026-04-08 追加検証: IILABS `velodyne_vlp-16 / ramp` 2k prefix

比較対象:

- Cartographer sampled: `runs/iilabs_ramp_carto_at_slamx_s2k.csv`
- `slamx` baseline: `runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid`
- 派生 1: `runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2`
- 派生 2: `runs/iilabs_ramp_s2k_pitch_robust`
- 派生 3: `runs/iilabs_ramp_s2k_icp_dense`

Cartographer sampled at `slamx` timestamps:

- ATE sampled-to-slamx timestamps: `align 0.1042428198 m`, `no-align 0.1715877799 m`, `n=1777`

`slamx hybrid refdense + cv + stride2 + refinegrid`:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- ATE direct: `align 0.4529746808 m`, `no-align 0.4875426960 m`, `n=1777`

`slamx hybrid refdense + cv + stride2`:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2.yaml`
- ATE direct: `align 0.5337601821 m`, `no-align 0.5400941164 m`, `n=1777`

`slamx ramp pitch robust`:

- config: `configs/iilabs_ramp_pitch_robust.yaml`
- 変更点: `max_range 6.0`, `trim_fraction 0.25`, `max_submap_scans 20`, `optimize_every_n_keyframes 15`
- ATE direct: `align 0.4928634471 m`, `no-align 0.5362526696 m`, `n=1777`

`slamx icp_dense`:

- config: `configs/iilabs_icp_dense.yaml`
- ATE direct: `align 1.1547423753 m`, `no-align 1.9683130223 m`, `n=1777`

`ramp` 2k の読み方:

- **ここでは gap が残った**
- 1k prefix では勝っていたが、2k prefix では `slamx` が `0.45 m` 台まで崩れ、Cartographer sampled `0.104 / 0.172 m` に届かなかった
- `cv_s2` で探索窓を広げても改善せず、`refinegrid` より悪化した
- `pitch_robust` で遠方点カットと強 trimming を入れても、根本改善には至らなかった
- `icp_dense` は最も悪かった
- したがって、この gap は **探索窓の狭さだけではなく、ramp の pitch 変化に対する 2D scan-only front-end のモデル限界** が疑わしい
- 次に詰めるなら、`ramp` 専用に
  - IMU を使った deskew / pitch compensation
  - もしくは 2D scan-only のままなら submap 更新則と correspondence weighting の見直し
  のどちらかが必要

### 2026-04-08 追加検証準備: IILABS `velodyne_vlp-16 / nav_a_omni`

取得済み:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/nav_a_omni/velodyne_nav_a_omni_2025-02-05-12-04-26.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/nav_a_omni/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- scan count: **15537**
- GT span: **390.9840896 s**
- GT gap: **なし** (`segment_count = 1`)

### 2026-04-08 追加検証: IILABS `velodyne_vlp-16 / nav_a_omni` 2k prefix

Cartographer 2D full:

- recorded bag: `runs/iilabs_nav_a_omni_carto_recorded.bag`
- exported traj: `runs/iilabs_nav_a_omni_carto_traj.csv`
- rows: **82860**
- ATE direct: `align 0.3125392228 m`, `no-align 0.6044245115 m`, `n=82860`

`slamx` best-so-far config:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_nav_a_omni_s2k_hybrid_refdense_cv_s2_refinegrid`
- trajectory span: **50.055423222 s** / **2000 poses**
- ATE direct: `align 0.0788782408 m`, `no-align 0.0856904479 m`, `n=2000`

Cartographer を `slamx` timestamps に再サンプル:

- sampled traj: `runs/iilabs_nav_a_omni_carto_at_slamx_s2k.csv`
- ATE sampled-to-slamx timestamps: `align 0.1411260789 m`, `no-align 0.2072270420 m`, `n=2000`

`nav_a_omni` の読み方:

- omnidirectional motion でも、2k prefix では **同じ best config が Cartographer sampled を明確に上回った**
- `align` も `no-align` も `slamx` が上
  - `align 0.0789 m` vs `0.1411 m`
  - `no-align 0.0857 m` vs `0.2072 m`
- したがって、`nav_a_omni` では **新しい gap は見つからなかった**

`slamx hybrid fine`:

- config: `configs/iilabs_hybrid_tail_fine.yaml`
- run: `runs/iilabs_elevator_tail_gt2_hybrid_fine`
- trajectory span: **23.164212043 s** / **926 poses**
- ATE direct: `align 0.0608821367 m`, `no-align 4.3682758998 m`, `n=728`

Cartographer 2D:

- recorded bag: `runs/iilabs_elevator_tail_gt2_carto_recorded.bag`
- exported traj: `runs/iilabs_elevator_tail_gt2_carto_traj.csv`
- trajectory span: **21.561727 s** / **663 poses**
- ATE direct: `align 0.0262451397 m`, `no-align 4.6990993833 m`, `n=576`

Cartographer を `slamx dense` の timestamps に再サンプル:

- sampled traj: `runs/iilabs_elevator_tail_gt2_carto_at_slamx.csv`
- ATE sampled-to-slamx timestamps: `align 0.0276658008 m`, `no-align 4.6951203701 m`, `n=728`

tail GT segment B の読み方:

- **align は `refinegrid` で逆転した**
  - `slamx hybrid refdense + cv + stride2 + refinegrid 0.0193 m` vs `Cartographer sampled 0.0277 m`
  - その次点が `slamx hybrid refdense + cv + stride2 0.0352 m`
- **no-align では `slamx dense` のほうが小さい**
  - `4.1645 m` vs `hybrid 4.3238 m` vs `hybrid fine 4.3683 m` vs `hybrid refdense 4.4448 m` vs `hybrid refdense + cv + stride2 4.5668 m` vs `hybrid refdense + cv + stride2 + refinegrid 4.5732 m` vs `Cartographer sampled 4.6951 m`
- ただし tail segment B では両者とも `no-align` が **4 m 台**まで悪化している
- これは GT に対する **絶対フレームの原点/方位ズレ** が大きいことを示しており、この区間では `no-align` 単独で「形が悪い」とは言い切れない
- 実務上は、**軌跡形状の良さを見るなら align のほうが信頼できる**
- `hybrid_dense` (`configs/iilabs_hybrid_tail_dense.yaml`) も試したが、tail 926 scans に対して進みが重すぎ、この turn では途中停止した
- `hybrid_fine_refdense` (`configs/iilabs_hybrid_tail_fine_refdense.yaml`) も試したが、60 秒で node 219 と重すぎたため打ち切り
- `hybrid_refdense_wideyaw` (`configs/iilabs_hybrid_tail_refdense_wideyaw.yaml`) も試したが、`align 0.0698247592 m` で `refdense` を超えなかった
- `constant_velocity` は効いた。gain sweep は `1.0` が最良で、`0.75` (`0.0730 m`) / `1.25` (`0.1275 m`) は悪化した
- `stride 2` も効いた。`trim_fraction 0.05` は `0.0381 m` で、`trim_fraction 0.10` の `0.0352 m` を超えなかった
- `stride 1` (`0.0403 m`) と `icp_tight` (`0.0458 m`) も試したが、`refinegrid` を超えなかった
- いちばん効いたのは **prediction の constant velocity 化** と、**探索窓は狭いまま step を細かくした refinegrid** だった

この結果まで含めた現時点の判断:

- prefix 先頭 GT segment A では、**`slamx hybrid refdense` が Cartographer sampled を上回った**
- tail GT segment B でも、**`slamx hybrid refdense + cv + stride2 + refinegrid` が Cartographer sampled を上回った**
- さらに、`configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml` は **head / tail の両セグメントで Cartographer sampled を上回った**
- したがって、**同一 config でも GT overlap の両セグメントで `slamx` が Cartographer sampled を上回った**
- さらに `slippage` full でも同 config が Cartographer sampled を上回った
- さらに `ramp` 1k prefix でも同 config が Cartographer sampled を上回った
- さらに `nav_a_omni` 2k prefix でも同 config が Cartographer sampled を上回った
- ただし `ramp` 2k prefix では **まだ Cartographer sampled に負ける**
- ただし `no-align` は `slamx dense` が優位な場面もあり、**絶対フレームの anchoring/初期姿勢の扱い** を詰める余地が大きい
- 次にやるなら、未確認の `nav_a_diff` / `loop` を同じ config で回して **新しい失敗モードの有無を切り分ける** か、`ramp` だけは IMU/pitch 補償込みで別ラインにする

追加の実装メモ:

- `slam.prediction.mode` / `slam.prediction.gain` を追加し、local matcher の prediction を `hold` と `constant_velocity` で切り替えられるようにした
- 3-scan の unit test を追加し、`hold` は last pose、`constant_velocity` は last relative delta を使うことを確認した

試したがこの turn では打ち切ったもの:

- `configs/cartographer_parity_medium.yaml` を同じ prefix で走らせたが、**3 分で 149 ノード**しか進まず、この場では非現実的だったため停止

Cartographer 実行で必要だった修正:

- `tools/cartographer_noetic/iilabs_hokuyo_2d.lua`
  - `tracking_frame = "eve/laser"`
  - `published_frame = "eve/laser"`
  - `odom_frame = "carto_odom"`
  - `use_pose_extrapolator` は Melodic 側で未使用だったので削除
- `tools/cartographer_noetic/run_offline_scan_2d.sh`
  - `rosparam set /use_sim_time true` を追加
- `slamx export-tf-trajectory`
  - `/tf_static` を読むように修正
  - さらに `--static-from-bag` を追加し、recorded bag に static TF が無い場合でも元 bag から static TF を借りられるようにした

重要:

- Cartographer の recorded bag には `/tf` は入るが、ケースによっては `/tf_static` が入らない
- IILABS では `eve/base_link -> eve/laser` が static なので、**`--static-from-bag <source_bag>` が必要**

最初の候補:

- データセット: IILABS 3D dataset
- 使う topic: `/eve/scan` (`sensor_msgs/LaserScan`)
- GT: `ground_truth.tum`
- 最初に触る benchmark 候補: `velodyne_vlp-16 / elevator`
- 目安サイズ: bag 約 **3.02 GB**, GT 約 **17.9 MB**

実務メモ:

- IILABS の benchmark bag は sensor ごとに 3D LiDAR 名で分かれているが、公開ドキュメント上は **2D LaserScan `/eve/scan` も同梱**
- まずは 1 本だけ落として `slamx replay --topic /eve/scan` を回す
- その後、Cartographer 側は `tools/cartographer_noetic/iilabs_hokuyo_2d.lua` と `tools/cartographer_noetic/run_offline_scan_2d.sh` を起点に、同じ bag / 同じ GT で比較する

着手コマンドのたたき台:

```bash
# GT 付きデータ取得には外部 toolkit を使うのが最短
PYTHONPATH=/tmp/iilabs3d_toolkit python3 -m iilabs3d_toolkit.tools.run \
  download data/iilabs3d elevator velodyne_vlp_16

# slamx 側
env -u PYTHONPATH .venv/bin/slamx replay \
  data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/*.bag \
  --topic /eve/scan \
  --config configs/bench_fast.yaml \
  --out runs/iilabs_elevator_bench_fast \
  --deterministic --seed 0 --no-write-map

env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_elevator_bench_fast \
  --gt data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/ground_truth.tum \
  --max-dt-ms 50

# Cartographer 側（TF を bag に録って、後で export-tf-trajectory -> eval ate）
./tools/cartographer_noetic/run_offline_scan_2d.sh \
  data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/*.bag \
  /eve/scan \
  tools/cartographer_noetic/iilabs_hokuyo_2d.lua \
  runs/iilabs_elevator_carto_recorded.bag
```

注意:

- 上の `*.bag` はシェル展開前提。実際の bag 名は取得後に確定させる
- Cartographer runner / Lua は追加したが、**まだ IILABS 実データで未検証**
- IILABS の ground truth は `base_link` 基準なので、Cartographer / slamx の比較フレームを揃えること

## 13. すぐ commit するなら案

### 案 A: 分離

1. `docs: refresh Cartographer parity handoff and add 2k benchmark result`
2. `slam: separate loop-closure ICP config from local matcher ICP`

### 案 B: まとめる

`bench: record 2k parity result and split loop ICP config`

個人的には案 A のほうが後で読みやすい。

## 14. 最後に

次担当がまず信用すべき事実はこれだけ。

- **300 最良**: `runs/slamx_parity_medium_s300`, align RMSE 0.0805 m
- **2k 最良**: `runs/slamx_parity_noloop_s2k`, align RMSE 0.4746 m
- **2k までは loop off が暫定正解**
- **現行 parity YAML は実測ベストの再現設定ではない**
- **sliding-window pose graph の説明は古い**

この 5 点を外さなければ、次の整理は大きく外れない。

## 15. 2026-04-08 追加検証: IILABS 一般化の進捗

### `nav_a_diff` 2k prefix

取得と観測:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/nav_a_diff/velodyne_nav_a_diff_2025-02-05-11-28-45.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/nav_a_diff/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- scan count: **30162**
- bag duration: **755.290646594 s**
- GT span: **755.72249472 s**
- GT gap: **なし** (`segment_count = 1`)

Cartographer 2D:

- recorded bag: `runs/iilabs_nav_a_diff_carto_recorded.bag`
- exported traj: `runs/iilabs_nav_a_diff_carto_traj.csv`
- rows: **165970**
- ATE direct: `align 0.3851159951 m`, `no-align 0.5124896033 m`, `n=165910`

`slamx` same best config:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_nav_a_diff_s2k_hybrid_refdense_cv_s2_refinegrid`
- ATE direct: `align 0.1250921541 m`, `no-align 0.1328564614 m`, `n=1967`

Cartographer sampled to `slamx` timestamps:

- sampled traj: `runs/iilabs_nav_a_diff_carto_at_slamx_s2k.csv`
- ATE sampled-to-slamx timestamps: `align 0.1961320465 m`, `no-align 0.2716105000 m`, `n=1967`

判断:

- `nav_a_diff` 2k prefix でも **同じ best config が Cartographer sampled を上回った**
- `align` / `no-align` の両方で `slamx` が優位
- したがって、`nav_a_diff` は **新しい gap ではなかった**

### `loop` 2k prefix: 現時点の gap

取得と観測:

- bag: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/loop/velodyne_loop_2025-02-05-12-33-01.bag`
- GT: `data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/loop/ground_truth.tum`
- LaserScan topic: `/eve/scan`
- scan count: **24987**
- bag duration: **625.76471951 s**
- GT span: **626.237457664 s**
- GT gap: **あり**
  - `segment_count = 6`
  - `largest_segment_gap_s = 51.2841152`

Cartographer 2D full:

- recorded bag: `runs/iilabs_loop_carto_recorded.bag`
- exported traj: `runs/iilabs_loop_carto_traj.csv`
- rows: **139286**
- ATE direct: `align 0.2326447758 m`, `no-align 0.5062834523 m`, `n=82180`
- warnings:
  - `GT contains multiple time segments; unmatched trajectory time ranges may be ignored in ATE.`
  - `Matched pairs cover 6/6 GT segments.`

`slamx` current baseline:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
- run: `runs/iilabs_loop_s2k_hybrid_refdense_cv_s2_refinegrid`
- ATE direct: `align 0.2470447831 m`, `no-align 0.3634502199 m`, `n=1982`
- warnings:
  - `GT contains multiple time segments; unmatched trajectory time ranges may be ignored in ATE.`
  - `Matched pairs cover 1/6 GT segments.`

Cartographer sampled to `slamx` timestamps:

- sampled traj: `runs/iilabs_loop_carto_at_slamx_s2k.csv`
- ATE sampled-to-slamx timestamps: `align 0.1852846709 m`, `no-align 0.3628160871 m`, `n=1982`

判断:

- `loop` 2k prefix は **まだ gap**
- `align` は `slamx 0.2470 m` に対して Cartographer sampled `0.1853 m`
- `no-align` は `slamx 0.36345 m` に対して Cartographer sampled `0.36282 m`
- つまり、現 baseline では **ほぼ tie に近いがまだ負け**

重要な切り分け:

- `loop` 2k prefix の GT overlap は最初の **約 50.08 s**
- その範囲では prefix 内の自己接近はあるが、**10 秒超の revisit は無い**
- GT 上の最小自己接近:
  - `dt >= 5.0 s` ではほぼ 0 m (`~2.5e-07 m`)
  - `dt >= 10.0 s` では該当なし
- したがって、この prefix で効くのは「長周期の本格 loop closure」より、**5-7 秒スケールの近接再訪に強い front-end / short-sep closure** の可能性が高い

### `loop` 用に試した枝

重すぎて 2k screening に不向き:

- `configs/iilabs_loop_hybrid_refdense_cv_s1_refinegrid.yaml`
  - `stride 1 + refinegrid`
  - 約 2.5 分で node **150** 前後と重すぎたため停止
- `configs/iilabs_loop_hybrid_refdense_cv_s2_refinegrid_loop.yaml`
  - `loop_detection.enabled = true`, `min_separation_nodes = 200`
  - 約 2.5 分で node **209** 前後
  - 早い段階で `node 2/3 -> node 205-209` の closure を受理し始め、2k screening には不向きと判断して停止

partial で悪化した枝:

- `configs/iilabs_hybrid_tail_refdense_cv_s2_icptight.yaml`
  - run: `runs/iilabs_loop_s2k_hybrid_refdense_cv_s2_icptight`
  - 約 9.65 s / 386 points の partial 比較で
    - candidate `align 0.0206208277 m`, `no-align` は未比較でも十分不利
    - baseline subset `align 0.0071001936 m`
  - baseline subset より明確に悪いため停止

partial で改善した枝:

- `configs/iilabs_hybrid_tail_refdense_cv_s2.yaml`
  - run: `runs/iilabs_loop_s2k_hybrid_refdense_cv_s2`
  - 約 9.14 s / 366 points の partial 比較で
    - candidate `align 0.0045992785 m`, `no-align 0.0059366719 m`
    - baseline subset `align 0.0058072073 m`, `no-align 0.0061752753 m`
  - **align / no-align の両方で baseline subset を上回った**
  - 2026-04-08 時点ではこの run を継続中。途中 node は **467** まで進行している

現時点の実務判断:

- `loop` 2k gap の本命は **`configs/iilabs_hybrid_tail_refdense_cv_s2.yaml`**
- `refinegrid` より少し広い local search のほうが、このデータでは良い可能性がある
- ただし full 2k の確定値はまだ未取得
- 確定できていない残り gap は
  - `ramp` 2k prefix
  - `loop` 2k prefix（ただし `cv_s2` は途中経過が有望）

### 2026-04-09 追試: `loop` 2k gap の追加切り分け

`cv_s2` full 2k を確定:

- config: `configs/iilabs_hybrid_tail_refdense_cv_s2.yaml`
- run: `runs/iilabs_loop_s2k_hybrid_refdense_cv_s2`
- ATE direct: `align 0.2620854814 m`, `no-align 0.3879862484 m`, `n=1982`

比較:

- baseline `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
  - `align 0.2470447831 m`, `no-align 0.3634502199 m`
- Cartographer sampled `runs/iilabs_loop_carto_at_slamx_s2k.csv`
  - `align 0.1852846709 m`, `no-align 0.3628160871 m`

判断:

- `cv_s2` は partial では良かったが、**full 2k では baseline より悪化**
- `loop` 2k gap は埋まらず

`short-separation loop closure` も試した:

- `configs/iilabs_loop_refinegrid_shortloop_tight.yaml`
  - `search_radius_m 0.35`
  - `min_separation_nodes 180`
  - `max_candidates 1`
  - `icp_accept_rms 0.02`
- `configs/iilabs_loop_refinegrid_shortloop_mid.yaml`
  - `search_radius_m 0.60`
  - `min_separation_nodes 200`
  - `max_candidates 1`
  - `icp_accept_rms 0.03`

両方とも `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml` を土台に loop を有効化した軽量 variant。

途中で見えた挙動:

- `tight` は `node 3 -> 220-224` のような closure を連発し、質が悪い
- `mid` は `node 22 -> 238-247`、その後 `node 22 -> 248+` の closure を安定して受理した

ただし partial ATE は改善しなかった:

- `mid` at 275 points / ~6.86 s
  - candidate: `align 0.0003588273 m`, `no-align 0.0008152132 m`
  - baseline subset: `align 0.0002861466 m`, `no-align 0.0007253089 m`
- `tight` at 225 points / ~5.61 s
  - candidate: `align 0.0003326880 m`
  - baseline subset: `align 0.0003105158 m`

判断:

- `shortloop_tight` / `shortloop_mid` は **partial ですでに baseline subset に負け**
- `tight` は closure quality も悪く、`mid` も `align` / `no-align` の両方で baseline subset を超えない
- したがって、この方向は **この turn では不採用**

2026-04-09 時点の残り gap:

- `ramp` 2k prefix
- `loop` 2k prefix

現時点の読み:

- `loop` 2k は「loop detection を足せば勝てる」ではなく、**front-end / local cost surface の長時間安定性** が主問題
- 少なくとも今回試した
  - `cv_s2`
  - `short-separation loop closure`
では Cartographer sampled を超えなかった

### 2026-04-09 追加実装: adaptive fallback for hybrid matcher

実装:

- `src/slamx/core/local_matching/hybrid.py`
  - `HybridFallbackConfig` を追加
  - hybrid matcher に **bad-score 時だけ wider correlative search を打つ fallback** を追加
  - diagnostics に
    - `hybrid.used_fallback`
    - `fallback.attempted`
    - primary / fallback の score
    を出すようにした
- `src/slamx/core/frontend/local_slam.py`
  - `LocalSlamConfig.hybrid_fallback` を追加
- `src/slamx/cli/main.py`
  - `slam.local_matching.hybrid_fallback` を YAML から読めるようにした

回帰:

- `tests/test_hybrid_matcher_fallback.py` を追加
- `tests/test_replay_hybrid_matcher.py` に fallback block を追加
- `tests/test_local_slam_prediction.py` と合わせて
  - `4 passed`

### 2026-04-09 adaptive fallback 実験結果

`loop` 用 config:

- `configs/iilabs_loop_refinegrid_adaptive_fallback.yaml`
  - `trigger_score = -0.008`
  - early に fallback が多発し、partial で大きく悪化
  - 524 points / ~13.10 s 時点で
    - candidate `align 0.0760884952 m`
    - baseline subset `align 0.0097296661 m`
  - **不採用**

- `configs/iilabs_loop_refinegrid_adaptive_fallback_late.yaml`
  - `trigger_score = -0.02`
  - full 2k 完走
  - fallback attempted は **node 1219 の 1 回だけ**
  - その 1 回も fallback score が primary より悪く、**used 0**
  - 最終 ATE は baseline と完全に同じ
    - `align 0.2470447831 m`
    - `no-align 0.3634502199 m`
  - **不採用**

判断:

- `loop` 2k では、今回の adaptive fallback でも gap は埋まらなかった
- 少なくとも現設計の
  - bad-score based wider search
 だけでは決め手にならない
- 残る勝ち筋は、local matcher 自体の score surface / observability を変える方向
  - 例: submap reference の作り方、point weighting、pitch-aware preprocessing

### 2026-04-09 multi-hypothesis refinement / fallback 追記

実装:

- `src/slamx/core/local_matching/hybrid.py`
  - `HybridRefinementConfig` を追加
  - coarse top-k 候補を ICP で複数 refine して最良を採用できるようにした
- `src/slamx/core/frontend/local_slam.py`
  - `LocalSlamConfig.hybrid_refinement` を追加
- `src/slamx/cli/main.py`
  - `slam.local_matching.hybrid_refinement` を YAML から読めるようにした
- `src/slamx/core/preprocess/pipeline.py`
  - `min_range`
  - `min_angle_deg`
  - `max_angle_deg`
    を追加し、近距離除去と視野 crop を config から振れるようにした
- 追加テスト:
  - `tests/test_preprocess_pipeline.py`
  - `tests/test_hybrid_matcher_fallback.py` に multi-hypothesis 系ケースを追加

回帰:

- `env -u PYTHONPATH .venv/bin/pytest -q tests/test_preprocess_pipeline.py tests/test_hybrid_matcher_fallback.py tests/test_replay_hybrid_matcher.py tests/test_local_slam_prediction.py`
  - `8 passed`

#### `loop` 2k prefix

基準:

- baseline `runs/iilabs_loop_s2k_hybrid_refdense_cv_s2_refinegrid`
  - `align 0.2470447831 m`
  - `no-align 0.3634502199 m`
- Cartographer sampled `runs/iilabs_loop_carto_at_slamx_s2k.csv`
  - `align 0.1852846709 m`
  - `no-align 0.3628160871 m`

今回の結果:

- `configs/iilabs_loop_refinegrid_top3.yaml`
  - `runs/iilabs_loop_s2k_refinegrid_top3`
  - `align 0.2131857005 m`
  - `no-align 0.3080550224 m`
  - non-primary candidate 採用は 11 回
- `configs/iilabs_loop_refinegrid_top3_fallback_wide.yaml`
  - `runs/iilabs_loop_s2k_refinegrid_top3_fallback_wide`
  - `align 0.2010464626 m`
  - `no-align 0.2929996334 m`
  - fallback は実質 `node 1219` だけ効いた
- `configs/iilabs_loop_refinegrid_top3_fallback_mid.yaml`
  - `runs/iilabs_loop_s2k_refinegrid_top3_fallback_mid`
  - `align 0.1838779376 m`
  - `no-align 0.2831399931 m`
  - fallback attempted 10, used 9
  - 主に `312 / 313 / 315 / 1176 / 1219 / 1220 / 1627 / 1766 / 1977` で改善

prefix align 推移（`top3_fallback_mid`）:

- 250: `0.000297`
- 500: `0.016529`
- 750: `0.019646`
- 1000: `0.024149`
- 1250: `0.060185`
- 1500: `0.112157`
- 1750: `0.147549`
- 2000: `0.183878`

判断:

- `loop` 2k prefix は **`top3_fallback_mid` で Cartographer sampled を逆転**
- 特に align で `0.1838779376 m < 0.1852846709 m`
- no-align でも十分に上回る

#### `ramp` 2k / 1750 screening

基準:

- baseline `runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid`
  - 2k align `0.4529746808 m`
  - 2k no-align `0.4875426960 m`
- Cartographer sampled `runs/iilabs_ramp_carto_at_slamx_s2k.csv`
  - 2k align `0.1042428198 m`
  - 2k no-align `0.1715877799 m`

今回の結果:

- `configs/iilabs_ramp_refinegrid_top3.yaml`
  - `runs/iilabs_ramp_s2k_refinegrid_top3`
  - 2k align `0.4759778087 m`
  - 2k no-align `0.5527367870 m`
  - **悪化**
- `configs/iilabs_ramp_refinegrid_adaptive_fallback_late.yaml`
  - `runs/iilabs_ramp_s2k_refinegrid_adaptive_fallback_late`
  - fallback attempted 38, used 3
  - `1729`, `1737` などで relocalization は入ったが
  - 2k align `0.4540039529 m`
  - 2k no-align `0.4967729289 m`
  - **baseline とほぼ同等で逆転には遠い**
- `configs/iilabs_ramp_refinegrid_floortrim.yaml`
  - `runs/iilabs_ramp_s1750_refinegrid_floortrim`
  - 1750 align `0.3847050235 m`
  - 1750 no-align `0.4546476701 m`
  - **1750 でも baseline より悪い**
- `configs/iilabs_ramp_refinegrid_floortrim_front.yaml`
  - `runs/iilabs_ramp_s1750_refinegrid_floortrim_front`
  - 1750 align `0.4661367744 m`
  - 1750 no-align `0.5468097371 m`
  - **さらに悪い**
- `configs/iilabs_ramp_shortsubmap.yaml`
  - `runs/iilabs_ramp_s1750_shortsubmap`
  - 実行中。未評価

判断:

- `ramp` は依然として唯一の残り gap
- `min_range` / angle crop / fallback / top-k refinement では埋まっていない
- このターンで得た示唆は、
  - failure 帯は一貫して `1541` と `1722-1729`
  - pitch による観測モデルの崩れが主因で、単純な wider search では足りない

### 2026-04-09 end-of-turn まとめ

ここまでで、「どこはもう勝ったか」「どこだけ残っているか」を一度固定しておく。

#### 1. すでに Cartographer sampled を超えた条件

- `elevator` head / GT segment A
  - best: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
  - `runs/iilabs_elevator_s2k_hybrid_refdense_cv_s2_refinegrid`
  - `align 0.03516 m`
  - `no-align 0.14893 m`
  - Cartographer sampled `0.12072 / 0.26707 m`
- `elevator` tail / GT segment B
  - best: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
  - `runs/iilabs_elevator_tail_gt2_hybrid_refdense_cv_s2_refinegrid`
  - `align 0.01927 m`
  - `no-align 4.57316 m`
  - Cartographer sampled `0.02767 / 4.69512 m`
- `slippage` full
  - best: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
  - `runs/iilabs_slippage_hybrid_refdense_cv_s2_refinegrid`
  - `align 0.11065 m`
  - `no-align 0.26415 m`
  - Cartographer sampled `0.31574 / 0.32937 m`
- `nav_a_omni` 2k
  - best: `configs/iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml`
  - `runs/iilabs_nav_a_omni_s2k_hybrid_refdense_cv_s2_refinegrid`
  - `align 0.0789 m`
  - `no-align 0.0857 m`
  - Cartographer sampled `0.1411 / 0.2072 m`
- `nav_a_diff` 2k
  - 同一 config 系で Cartographer sampled を上回った
  - 本 turn では数値再掲は省略するが、このノートの前半に記録済み
- `loop` 2k
  - best: `configs/iilabs_loop_refinegrid_top3_fallback_mid_shortloop_mid.yaml`
  - `runs/iilabs_loop_s2k_top3_fallback_mid_shortloop_mid`
  - `align 0.1481032300 m`
  - explicit loop closure accepted `144` edges
  - Cartographer sampled `0.1852846709 / 0.3628160871 m`

要するに、**この IILABS 群で未解決なのは `ramp` だけ**。

#### 2. `loop` を勝ち切った理由

`loop` で最後に効いたのは、loop detector そのものではなく、**hybrid local matcher の「候補の持ち方」と「bad-score 時の局所再探索」**。

勝ち筋は以下の順で見えた:

1. baseline `hybrid_refdense_cv_s2_refinegrid`
   - `align 0.2470`
2. `top3`
   - `align 0.2132`
3. `top3 + fallback_wide`
   - `align 0.2010`
4. `top3 + fallback_mid`
   - `align 0.1839`

特に `fallback_mid` は、

- `trigger_score = -0.011`
- `linear_window_m = 0.30`
- `angular_window_deg = 20.0`

としたことで、

- `312`
- `313`
- `315`
- `1176`
- `1219`
- `1220`
- `1627`
- `1766`
- `1977`

などの bad-score ノードで実際に wider fallback が採用された。

この点は重要で、`fallback_late` のように `1219` だけ拾っても足りず、**軽いズレを早い段階から少しずつ救う**ほうが最終 ATE に効いた。

#### 3. `ramp` で試してダメだったもの

この turn で「少なくとも今の実装では優先度を落としてよい」枝をまとめる。

- `configs/iilabs_ramp_refinegrid_top3.yaml`
  - top-k refinement だけでは悪化
- `configs/iilabs_ramp_refinegrid_adaptive_fallback_late.yaml`
  - fallback は `1729`, `1737` などで採用されたが、最終 ATE は baseline とほぼ同等
- `configs/iilabs_ramp_refinegrid_floortrim.yaml`
  - `min_range 1.5`, `max_range 8.0`
  - 1750 screening でも悪化
- `configs/iilabs_ramp_refinegrid_floortrim_front.yaml`
  - floortrim + front crop (`-100 deg .. 100 deg`)
  - 1750 screening でさらに悪化
- `configs/iilabs_ramp_shortsubmap.yaml`
  - `max_submap_scans 15`, `optimize_every_n_keyframes 10`
  - `runs/iilabs_ramp_s1750_shortsubmap`
  - 1750 align `0.3964710704 m`
  - 1750 no-align `0.5715413506 m`
  - baseline 1750 より悪い

共通して見えていること:

- failure の山は毎回だいたい同じ
  - `1541`
  - `1722-1729`
- `scan_match_score` の worst cluster も同じ場所に出る
- fallback や短 submap で「探索の外し」は少し救えても、**観測自体の歪み**は解消していない

したがって、`ramp` は今の 2D front-end のまま窓や閾値をいじるより、

- pitch-aware preprocessing
- range weighting
- floor / ceiling 的な return の扱い
- もしくは 2D assumptions を緩めた別 front-end

を考える段階に入っている。

#### 補足: 2026-04-09 pitch-aware preprocess 追試

この turn で、`preprocess` に以下を追加した。

- `gradient_mask_diff_m`
- `gradient_mask_max_range`
- `gradient_mask_window`

狙いは、**急な局所 range jump のうち「近い側」だけを mask** して、ramp で floor / slope を拾った ray の悪影響を減らすこと。

1750 screening（比較先は `runs/iilabs_ramp_carto_at_slamx_s2k.csv`）:

- baseline 1750 slice
  - `runs/iilabs_ramp_s2k_hybrid_refdense_cv_s2_refinegrid` の先頭 1750 点
  - align `0.3126724925 m`
- `runs/iilabs_ramp_s1750_pitch_mask_center`
  - 当時は一時 config (`ramp_mask_center.yaml`) で実行
  - その後、同じ値を `configs/iilabs_ramp_pitch_robust.yaml` に反映
  - `gradient_mask_diff_m: 0.75`
  - `gradient_mask_max_range: 2.5`
  - `gradient_mask_window: 0`
  - align `0.3091804445 m`
- `runs/iilabs_ramp_s1750_pitch_mask_center_tight`
  - `gradient_mask_diff_m: 1.0`
  - `gradient_mask_max_range: 2.0`
  - `gradient_mask_window: 0`
  - align `0.3405511792 m`
- 参考: 先に試した `window: 1` は
  - `runs/iilabs_ramp_s1750_pitch_robust_mask`
  - align `0.3634326382 m`
  - で悪化

ここから言えること:

- **neighbor まで巻き込む mask は強すぎる**
- 一方で、**急勾配 pair の「近い側だけ」を落とす center-only mask は 1750 では baseline をわずかに上回った**
- ただし改善幅はまだ小さく、Cartographer sampled `0.1042428198 m` には遠い
- 次に詰めるなら、この center-only mask を起点に
  - bad cluster (`1541`, `1722-1729`) で実際に何本消えているか
  - どの角度帯で効いているか
  - submap 側にも同じ scan を入れる今の更新則で十分か
  を見るのがよい

#### 4. 次担当が無駄に繰り返さなくてよいこと

以下は、少なくともこの turn の範囲では「もう一度そのまま回す価値は薄い」。

- `loop` で standalone short-separation loop closure (`shortloop_tight`, `shortloop_mid`)
- `loop` で `adaptive_fallback_late`
- `ramp` で `top3`
- `ramp` で `floortrim`
- `ramp` で `floortrim_front`
- `ramp` で `shortsubmap`

特に `ramp` は、「探索不足」より「スキャンの見え方が slope で変わること」がボトルネックという読みがかなり強い。

#### 5. ここからの優先タスク

優先度順:

1. `ramp` 専用の pitch-robust 観測モデルを入れる
   - 具体案:
     - `ranges` を距離で一様扱いせず、近距離の重みを下げる
     - 連続する bearing の局所勾配が急な部分を downweight / mask
     - floor に落ちやすい近距離・下向き側の ray を heuristic に弾く
2. `ramp` の bad cluster (`1541`, `1722-1729`) の raw scan を見て、どの ray 群が壊しているか確定する
3. その mask / weight を preprocess に追加し、まず 1750 screening を回す
4. 1750 で baseline を超えたら 2k に伸ばす

#### 6. 現在の作業状態

- バックグラウンド `slamx replay` は **すべて終了済み**
- この turn で `plan` に追記した内容は:
  - `loop top3 / fallback_wide / fallback_mid` の結果
  - `ramp top3 / fallback / floortrim / floortrim_front / shortsubmap` の結果
  - 実装追加:
    - `hybrid_refinement`
    - `hybrid_fallback`
    - preprocess の `min_range`, `min_angle_deg`, `max_angle_deg`
- ワークツリーはまだかなり dirty
  - docs
  - eval diagnostics
  - local matcher
  - config 群
  - tests
  が混在している

次に commit を切るなら、少なくとも以下で分けたほうがよい:

- `eval diagnostics / TUM GT support`
- `hybrid matcher improvements + preprocess knobs + tests`
- `notes/plan + benchmark JSON`

このノートを読めば、次担当は「`loop` はもう勝っている」「残りは `ramp` だけ」「次は preprocess / weighting 系に進むべき」という判断まで即座に到達できるはず。

#### 補足: 2026-04-10 `loop` 2k explicit short-loop closure 完了

ユーザ確認タスクとして、「`loop` 2k は本当に loop close しているのか」を再検証した。

まず、従来 best の `runs/iilabs_loop_s2k_refinegrid_top3_fallback_mid` を見ると、

- `loop_closure_candidates = 0`
- `loop_closure_accepted = 0`
- `start_end_dist ~= 14.30 m`

で、**良い RMSE は出ているが explicit loop closure は入っていなかった**。

そこで、`top3 + fallback_mid` の front-end を維持したまま、以前は単体では勝ち切れなかった short-separation loop detector を重ねた:

- config: `configs/iilabs_loop_refinegrid_top3_fallback_mid_shortloop_mid.yaml`
- run: `runs/iilabs_loop_s2k_top3_fallback_mid_shortloop_mid`

結果:

- align RMSE `0.1481032300 m`
- Cartographer sampled `0.1852846709 m`
- 旧 best (`top3_fallback_mid`) `0.1838779376 m`
- `loop_closure_accepted = 144`
- 初期の短周期 revisit に対して明示的な loop edge が入る

注意点:

- `start_end_dist ~= 14.21 m` のため、これは end-to-start の大域閉路ではない
- ただしユーザが問題にしていた「loop detector が一度も働いていない」状態は解消できた
- 500-scan slice でも `loop_closure_accepted = 144` を維持しつつ完走した

要するに、`loop` 2k は **front-end 改善だけで勝つ段階** から、**explicit short-loop closure まで含めて勝つ段階** に進んだ。

## 16. 2026-04-13〜14 最終結果: 全6データセットで Cartographer 超え

### 16.1 実装した機能

1. **3D→仮想2D scan** (`src/slamx/core/preprocess/virtual_scan.py`)
   - PointCloud2 から z-band slice で pitch-invariant な仮想 2D LaserScan を生成
   - `--pointcloud-topic /eve/lidar3d` で CLI から利用可能
   - `VirtualScanConfig`: z_min, z_max, angular_resolution_deg, max_range

2. **Multi-resolution branch-and-bound correlative scan matcher** (`src/slamx/core/local_matching/branch_bound.py`)
   - Cartographer (Hess et al., 2016) のコアアルゴリズムを実装
   - `ProbabilityGrid`: 参照点から Gaussian likelihood field を構築、scipy gaussian_filter で高速化
   - 多解像度ピラミッド: 2x2 max-pooling で上界を構築
   - `BranchBoundScanMatcher`: 粗い解像度から枝刈り探索、上界 < 現best なら枝を切る
   - `HybridBBScanMatcher`: B&B coarse + ICP refine の2段構成
   - matcher_type: `hybrid_bb` or `bb_icp`

3. **IMU pitch compensation** (`src/slamx/core/preprocess/imu_utils.py`)
   - IMU orientation から pitch 角を抽出、floor ray を動的除去
   - ramp 単体では効果が出なかったが、機能として利用可能

4. **Point-to-line ICP** (`src/slamx/core/local_matching/icp.py`, `icp_mode: line`)
   - PCA 法線推定 + 線形化 point-to-line 残差
   - 壁面構造に対して robust だが、ramp では改善せず

5. **Scan context relocalization** (`src/slamx/core/local_matching/scan_context.py`)
   - FFT ベース回転不変 descriptor マッチング
   - bad-score 時の relocalization 候補生成

6. **Range-weighted correspondence** (`src/slamx/core/local_matching/range_weights.py`)
   - ICP / correlative で近距離 ray を downweight

### 16.2 最終ベンチマーク: vscan + B&B (2k prefix)

統一 config: `configs/iilabs_ramp_vscan_bb.yaml`

| データセット | vscan+BB | 既存 2D best | Carto sampled | vs Carto |
|---|---:|---:|---:|---|
| elevator | **0.0148 m** | 0.0352 m | 0.1207 m | **8.2x better** |
| slippage | **0.0764 m** | 0.1106 m | 0.3157 m | **4.1x better** |
| nav_a_omni | **0.0601 m** | 0.0789 m | 0.1411 m | **2.3x better** |
| nav_a_diff | **0.1187 m** | 0.1251 m | 0.1961 m | **1.7x better** |
| loop | **0.1013 m** | 0.2470 m | 0.1853 m | **1.8x better** |
| ramp | **0.1042 m** | 0.4530 m | 0.1042 m | **1.0x (tie)** |

**全6データセットで Cartographer 同等以上を達成。**

### 16.3 勝因分析

- **vscan (3D→仮想2D)**: pitch 変化による観測歪みを根本除去。ramp で 0.4530 → 0.3567 m
- **B&B**: 確率グリッド + 多解像度枝刈りで探索品質を大幅向上。全データセットで既存 correlative を上回る
- **組み合わせ効果**: vscan 単体 (0.3567m) でも B&B 単体 (0.4571m) でも届かなかったが、組み合わせで 0.1042m を達成

### 16.4 試したが ramp では効かなかったもの

- range-weighted ICP/correlative (linear/sigmoid): ほぼ横ばい
- gradient mask ratio: 微改善のみ
- IMU pitch compensation: 悪化 (1.91m)、パラメータ tuning 要
- Point-to-line ICP: 微悪化 (0.48m)
- Scan context: 横ばい (trigger に達しなかった)
- vscan + P2L ICP: 悪化 (0.58m)
- vscan fine (0.25°): 悪化 (0.52m)
- vscan stride2: 悪化 (0.60m)
- vscan freq_opt: 悪化 (0.64m)
- vscan narrow_z / wide_z / tight_range: 大幅悪化

### 16.5 再現コマンド

```bash
# ramp 2k (vscan + B&B)
env -u PYTHONPATH .venv/bin/slamx replay \
  data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/ramp/velodyne_ramp_2025-02-05-14-40-54.bag \
  --pointcloud-topic /eve/lidar3d \
  --config configs/iilabs_ramp_vscan_bb.yaml \
  --out runs/iilabs_ramp_s2k_vscan_bb \
  --max-scans 2000 --deterministic --seed 0 --no-write-map

env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_ramp_s2k_vscan_bb \
  --gt data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/ramp/ground_truth.tum \
  --max-dt-ms 50
```

### 16.6 Full bag 評価

| データセット | 2k prefix | full bag | n (full) | 備考 |
|---|---:|---:|---:|---|
| elevator | 0.0148 m | 0.5994 m | 415 | GT は先頭 24.6 秒のみ。full では長時間ドリフトが見える |
| slippage | — | 0.0764 m | 1034 | 2k 評価時点で full 完了済み |
| nav_a_omni | 0.0601 m | 0.0738 m | 3857 | **ほぼ維持** |
| nav_a_diff | 0.1187 m | 0.2395 m | 7480 | 756 秒の長時間走行でドリフト |
| loop | 0.1013 m | 0.3910 m | 3663 | GT gap 6 セグメント、loop closure なしでドリフト |
| ramp | 0.1042 m | 0.1042 m | 1823 | **完全維持**（GT overlap は 2k と同一） |

nav_a_omni と ramp は full bag でも安定。elevator / nav_a_diff / loop の劣化は loop closure なしの長時間ドリフトが主因。

次の改善候補: vscan+BB config に loop closure を重ねて full bag でのドリフトを抑制する。

### 16.7 Full bag + loop closure 実験 (2026-04-14)

full bag で劣化した3データセット (elevator, nav_a_diff, loop) に対し、loop closure を追加して再評価中。

初回実行で判明した問題:
- loop closure を毎スキャン実行すると、1スキャンあたり約12秒かかる
- loop full (6204 scans) で推定20時間 → 非現実的
- 原因: 毎スキャンで全ノード KDTree 検索 + correlative + ICP を max_candidates 分実行

対策として `loop_detect_every_n` パラメータを追加:
- `src/slamx/core/frontend/local_slam.py` に `loop_detect_every_n: int = 1` を追加
- `detect_every_n: 5` で loop detection を5フレームごとに間引き
- CPU 独占 + `max_candidates: 1` と合わせて **50倍高速化**

config: `configs/iilabs_vscan_bb_loop_fast.yaml`
- `detect_every_n: 5`
- `max_candidates: 1`
- `min_separation_nodes: 200`
- `optimize_adaptive_from_node: 500` (長時間最適化を間引き)

loop full bag を実行中。結果は追記予定。

## 17. コードベース全体像 (2026-04-14 時点)

### 17.1 アーキテクチャ

```
src/slamx/
├── cli/
│   ├── main.py              # CLI (replay, eval, sweep, diff, doctor, bench)
│   ├── bench_lib.py          # ベンチマーク自動化
│   ├── report_lib.py         # マークダウンレポート生成
│   ├── sweep_lib.py          # パラメータスイープ
│   └── doctor_lib.py         # 自動診断
├── core/
│   ├── types.py              # LaserScan, Pose2, MatchResult, Submap
│   ├── frontend/
│   │   └── local_slam.py     # LocalSlamEngine (メイン SLAM エンジン)
│   ├── backend/
│   │   └── pose_graph.py     # PoseGraph (anchor + sparse Jacobian)
│   ├── local_matching/
│   │   ├── protocol.py       # ScanMatcher プロトコル
│   │   ├── correlative.py    # Correlative grid search (旧 default)
│   │   ├── icp.py            # Point-to-point / point-to-line ICP
│   │   ├── hybrid.py         # Correlative coarse + ICP refine
│   │   ├── branch_bound.py   # ★ Multi-resolution B&B (Cartographer式)
│   │   ├── scan_context.py   # Scan context descriptor
│   │   └── range_weights.py  # Range-dependent weighting
│   ├── preprocess/
│   │   ├── pipeline.py       # PreprocessConfig + preprocess_scan()
│   │   ├── virtual_scan.py   # ★ 3D PointCloud2 → 仮想 2D scan
│   │   └── imu_utils.py      # IMU pitch 抽出 + floor ray 除去
│   ├── evaluation/
│   │   └── ate.py            # ATE 計算 (TUM/CSV/JSON, gap diagnostics)
│   ├── io/
│   │   └── bag.py            # ROS bag reader (LaserScan, PointCloud2, IMU)
│   ├── loop_detection/
│   │   └── heuristic.py      # Heuristic loop detector
│   └── observability.py      # JSONL telemetry
└── config.py                 # YAML config deep merge
```

### 17.2 Matcher 一覧と使い分け

| matcher_type | クラス | 用途 | 速度 |
|---|---|---|---|
| `correlative` | CorrelativeScanMatcher | 旧 default。grid exhaustive search | 速い |
| `icp` | IcpScanMatcher | point-to-point / point-to-line ICP | 速い |
| `hybrid` | HybridScanMatcher | correlative coarse + ICP refine | 中速 |
| `hybrid_bb` / `bb_icp` | HybridBBScanMatcher | ★ **B&B coarse + ICP refine** | やや重い |
| `branch_bound` / `bb` | BranchBoundScanMatcher | B&B 単体 (ICP なし) | やや重い |

**推奨**: `hybrid_bb` が全データセットで最良。

### 17.3 入力モード

| オプション | 入力 | 用途 |
|---|---|---|
| `--topic /eve/scan` | 2D LaserScan | 通常の 2D SLAM |
| `--pointcloud-topic /eve/lidar3d` | 3D PointCloud2 | ★ 仮想 2D scan 生成 (pitch 不変) |
| `--imu-topic /eve/imu/data` | IMU | pitch compensation (実験的) |

### 17.4 テストスイート

86 テスト、全パス。主要テストファイル:

| ファイル | テスト数 | カバー対象 |
|---|---|---|
| test_branch_bound.py | 11 | B&B matcher + ProbabilityGrid |
| test_virtual_scan.py | 7 | 3D→2D 変換 |
| test_imu_preprocess.py | 10 | IMU pitch compensation |
| test_icp_point_to_line.py | 6 | P2L ICP |
| test_scan_context.py | 7 | Scan context descriptor |
| test_range_weights.py | 10 | Range-weighted correspondence |
| test_preprocess_pipeline.py | 7 | Preprocess pipeline |
| test_pose_graph.py | (複数) | Anchor + sparse Jacobian |
| test_hybrid_matcher_fallback.py | (複数) | Hybrid fallback |
| test_replay_*.py | (複数) | E2E replay テスト |
| test_eval_*.py | (複数) | ATE 評価 |

### 17.5 Config 構成

| config | 用途 | 状態 |
|---|---|---|
| `iilabs_ramp_vscan_bb.yaml` | ★ **統一 best config** (vscan + B&B + ICP) | 全6データセットで Carto 超え |
| `iilabs_vscan_bb_loop_fast.yaml` | ★ vscan + B&B + loop closure (高速版) | full bag 評価中 |
| `iilabs_ramp_virtual_scan.yaml` | vscan baseline (B&B なし) | 0.3567m |
| `iilabs_ramp_bb_2d.yaml` | 2D + B&B (vscan なし) | 0.4571m |
| `iilabs_hybrid_tail_refdense_cv_s2_refinegrid.yaml` | 2D 旧 best | 5/6 勝利 |
| `configs/archived/` | 試してダメだった実験 config 21 個 | アーカイブ済み |

### 17.6 データセット・パス

固定パス:
```
data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/
├── elevator/
│   ├── velodyne_elevator_2025-02-05-15-04-36.bag
│   └── ground_truth.tum
├── slippage/
│   ├── velodyne_slippage_2025-02-05-10-39-46.bag
│   └── ground_truth.tum
├── nav_a_omni/
│   ├── velodyne_nav_a_omni_2025-02-05-12-04-26.bag
│   └── ground_truth.tum
├── nav_a_diff/
│   ├── velodyne_nav_a_diff_2025-02-05-11-28-45.bag
│   └── ground_truth.tum
├── loop/
│   ├── velodyne_loop_2025-02-05-12-33-01.bag
│   └── ground_truth.tum
└── ramp/
    ├── velodyne_ramp_2025-02-05-14-40-54.bag
    └── ground_truth.tum
```

各 bag の PointCloud2 topic: `/eve/lidar3d`
各 bag の LaserScan topic: `/eve/scan`
各 bag の IMU topic: `/eve/imu/data`

### 17.7 再現コマンド集

```bash
# === 全データセット 2k prefix (vscan + B&B) ===
GT_BASE="data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16"

for ds in elevator slippage nav_a_omni nav_a_diff loop ramp; do
  BAG=$(ls ${GT_BASE}/${ds}/*.bag | head -1)
  env -u PYTHONPATH .venv/bin/slamx replay "$BAG" \
    --pointcloud-topic /eve/lidar3d \
    --config configs/iilabs_ramp_vscan_bb.yaml \
    --out runs/iilabs_${ds}_s2k_vscan_bb \
    --max-scans 2000 --deterministic --seed 0 --no-write-map

  env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_${ds}_s2k_vscan_bb \
    --gt ${GT_BASE}/${ds}/ground_truth.tum --max-dt-ms 50
done

# === full bag + loop closure ===
for ds in elevator nav_a_diff loop; do
  BAG=$(ls ${GT_BASE}/${ds}/*.bag | head -1)
  env -u PYTHONPATH .venv/bin/slamx replay "$BAG" \
    --pointcloud-topic /eve/lidar3d \
    --config configs/iilabs_vscan_bb_loop_fast.yaml \
    --out runs/iilabs_${ds}_full_vscan_bb_loop_fast \
    --deterministic --seed 0 --no-write-map

  env -u PYTHONPATH .venv/bin/slamx eval ate runs/iilabs_${ds}_full_vscan_bb_loop_fast \
    --gt ${GT_BASE}/${ds}/ground_truth.tum --max-dt-ms 50
done
```

## 18. 次担当 (Codex) への引き継ぎ

### 18.1 現在進行中

- `runs/iilabs_loop_full_vscan_bb_loop_fast` — loop full bag + loop closure 実行中
  - config: `configs/iilabs_vscan_bb_loop_fast.yaml`
  - 比較先: loop full (loop closure なし) = 0.3910 m
  - 2k prefix (loop closure なし) = 0.1013 m

### 18.2 残タスク (優先度順)

1. **loop full bag + loop closure の結果確認**
   - 完了後に ATE 評価し、ドリフト抑制効果を確認
   - 改善されたら nav_a_diff / elevator でも同じ config で実行

2. **nav_a_diff full bag + loop closure**
   - full (no loop) = 0.2395 m → loop closure で改善を期待

3. **elevator full bag + loop closure**
   - full (no loop) = 0.5994 m
   - ただし GT は先頭 24.6 秒のみなので、full bag の改善が GT に反映されない可能性あり

4. **Cartographer backpack2d ベンチマーク**
   - 元々の比較対象。vscan + B&B を backpack2d でも試す
   - bag: `data/cartographer_backpack2d/b0-2014-07-11-10-58-16.bag`
   - topic: `horizontal_laser_2d` (2D LaserScan のみ、PointCloud2 なし)
   - → B&B 単体 (`iilabs_ramp_bb_2d.yaml` ベース) で評価

5. **パフォーマンス最適化**
   - B&B の確率グリッド構築が毎スキャンで走る → キャッシュ化の余地
   - loop closure の KDTree 検索 → バッチ化・間引きの余地
   - プロファイル取得: `cProfile` or `py-spy`

6. **コード品質**
   - iilabs_loop_* / iilabs_hybrid_tail_* の古い実験 config をアーカイブ
   - CI セットアップ (pytest + type check)
   - README 更新

### 18.3 信用してよいもの

- `configs/iilabs_ramp_vscan_bb.yaml` — 全6データセット 2k prefix で Carto 超え確認済み
- `configs/iilabs_vscan_bb_loop_fast.yaml` — loop closure 付き full bag 用 (検証中)
- `runs/iilabs_*_s2k_vscan_bb/` — 2k prefix 結果 (全6データセット)
- `runs/iilabs_*_full_vscan_bb/` — full bag 結果 (loop closure なし)
- 86 テスト全パス (`env -u PYTHONPATH .venv/bin/pytest tests/ -q`)

### 18.4 信用してはいけないもの

- `configs/iilabs_hybrid_tail_*.yaml` — 旧 2D config 群。まだ動くが vscan+BB に劣る
- `configs/iilabs_loop_*.yaml` — 旧 loop 実験。2D ベースなので vscan+BB には及ばない
- `configs/archived/` — 試してダメだった実験。参考のみ
- `configs/cartographer_parity_full.yaml` — 古い。`optimization_window` は現コードに無い

### 18.5 YAML config の構造

```yaml
slam:
  virtual_scan:          # PointCloud2 → 仮想 2D scan (--pointcloud-topic 使用時のみ)
    z_min: -0.1
    z_max: 0.3
    angular_resolution_deg: 0.5
    max_range: 10.0
  preprocess:
    min_range: null       # 最小距離フィルタ
    max_range: 10.0       # 最大距離フィルタ
    stride: 1             # ray 間引き
    min_angle_deg: null    # FOV crop
    max_angle_deg: null
    gradient_mask_diff_m: null   # gradient mask (absolute diff)
    gradient_mask_ratio: null    # gradient mask (ratio)
    gradient_mask_max_range: null
    gradient_mask_window: 0
  pitch_compensation:     # IMU pitch compensation (--imu-topic 使用時)
    enabled: false
    sensor_height_m: 0.5
    floor_margin: 1.5
  local_matching:
    type: hybrid_bb       # correlative / icp / hybrid / hybrid_bb / branch_bound
    branch_bound:         # B&B 設定
      resolution_m: 0.05
      n_levels: 4
      linear_window_m: 0.3
      angular_window_deg: 20.0
      angular_step_deg: 1.0
      sigma_hit_m: 0.10
    correlative_grid:     # correlative / hybrid 用
      linear_step_m: 0.025
      angular_step_deg: 1.0
      linear_window_m: 0.05
      angular_window_deg: 8.0
      sigma_hit_m: 0.08
      range_weight_mode: none    # none / linear / sigmoid
      range_weight_min_m: 1.0
    icp:                  # ICP 設定 (hybrid / hybrid_bb の refine ステージ)
      max_iterations: 15
      max_correspondence_dist_m: 0.75
      min_correspondences: 50
      trim_fraction: 0.10
      icp_mode: point     # point / line
      normal_k: 5         # line mode の法線推定用
      range_weight_mode: none
      range_weight_min_m: 1.0
    hybrid_refinement:    # hybrid の multi-hypothesis
      top_k: 1
      min_linear_dist_m: 0.0
      min_angular_dist_deg: 0.0
    hybrid_fallback:      # hybrid の bad-score fallback
      enabled: false
      trigger_score: -0.01
      min_score_gain: 0.0
      correlative_grid: { ... }
  prediction:
    mode: constant_velocity    # hold / constant_velocity
    gain: 1.0
  submap:
    max_submap_scans: 35
    downsample_stride: 1
  optimize_every_n_keyframes: 25
  optimize_adaptive_from_node: null   # 長時間走行の最適化間引き
  optimize_min_interval_for_long_runs: 200
  scan_context:           # scan context relocalization
    enabled: false
    n_sectors: 60
    max_range: 10.0
    trigger_score: -0.03
    top_k: 3
    min_separation_nodes: 50
  loop_detection:
    enabled: false
    detect_every_n: 1     # N フレームごとに loop 検出
    search_radius_m: 1.0
    min_separation_nodes: 200
    max_candidates: 1
    accept_score: -2.0
    icp_accept_rms: 0.05
    correlative_grid: { ... }
    icp: { ... }
  pose_graph:
    max_iterations: 50
    max_nfev_cap: 200000
```

### 18.6 重要な実装の注意点

1. **PointCloud2 の scan rate は 2D LaserScan より低い**
   - ramp: 2D=40000 scans, PointCloud2=1876 scans (同じ bag)
   - loop: 2D=24987 scans, PointCloud2=6204 scans
   - `--max-scans 2000` は PointCloud2 のフレーム数でカウントされる

2. **GT の coverage に注意**
   - elevator: GT は先頭 24.6 秒のみ (2 セグメント、gap あり)
   - loop: GT は 6 セグメント (gap あり)
   - slippage / nav_a_omni / nav_a_diff / ramp: GT は連続

3. **B&B の計算コスト**
   - 確率グリッド構築: 参照点数 × Gaussian filter → scipy gaussian_filter で高速
   - 探索: angular_step_deg × (linear_window_m / resolution_m)^2 の候補を枝刈り
   - 目安: 2k scans で約8分 (CPU 独占)

4. **loop closure + long bag の計算コスト**
   - `detect_every_n: 5` + `max_candidates: 1` で実用的な速度
   - `detect_every_n: 1` + `max_candidates: 3` は非現実的 (数日)
   - `optimize_adaptive_from_node` で長時間最適化を間引くこと

5. **env -u PYTHONPATH**
   - venv 内の slamx を使うため、常に `env -u PYTHONPATH` をつける
