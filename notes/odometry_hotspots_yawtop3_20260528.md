# Odometry Hotspots: full no-loop yawtop3 (2026-05-28)

Run: `runs/iilabs_elevator_full_vscan_bb_yawtop3_20260528`

生成コマンド:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/slamx cloud-analyze \
  runs/iilabs_elevator_full_vscan_bb_yawtop3_20260528 \
  --markdown --scan-match-hotspots 20
```

## Scan-Match Hotspots (top 20 by pose_jump)

| node | pose_jump_m | score    | pred_dx_m | pred_dyaw_rad | icp_rms | best_idx | n_ref | score_gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3771 | 0.275 | -0.0342 | 0.254 | 0.021 | 0.153 | 0 | 3 | 0.000 |
| 3941 | 0.228 | -0.0049 | 0.106 | 0.122 | 0.005 | 0 | 3 | 0.000 |
| 1032 | 0.217 | -0.0027 | 0.112 | 0.105 | 0.003 | 0 | 3 | 0.001 |
| 1176 | 0.203 | -0.0024 | 0.081 | 0.122 | 0.002 | 1 | 3 | 0.000 |
|  940 | 0.202 | -0.0019 | 0.080 | 0.122 | 0.002 | 0 | 3 | 0.000 |
| 1094 | 0.201 | -0.0022 | 0.079 | 0.122 | 0.002 | 2 | 3 | 0.000 |
|  936 | 0.201 | -0.0026 | 0.079 | 0.122 | 0.003 | 0 | 3 | 0.000 |
|  937 | 0.201 | -0.0020 | 0.079 | 0.122 | 0.002 | 2 | 3 | 0.000 |
| 3942 | 0.195 | -0.0051 | 0.092 | 0.103 | 0.005 | 1 | 3 | 0.001 |
|  938 | 0.184 | -0.0020 | 0.079 | 0.105 | 0.002 | 0 | 3 | 0.000 |
|  939 | 0.184 | -0.0019 | 0.079 | 0.105 | 0.002 | 2 | 3 | 0.000 |
| 3969 | 0.180 | -0.0087 | 0.091 | 0.088 | 0.009 | 1 | 3 | 0.001 |
| 3772 | 0.177 | -0.0115 | 0.177 | 0.000 | 0.012 | 0 | 3 | 0.001 |
| 4076 | 0.177 | -0.0049 | 0.123 | 0.054 | 0.005 | 2 | 3 | 0.005 |
|  941 | 0.176 | -0.0026 | 0.091 | 0.085 | 0.003 | 0 | 3 | 0.000 |
| 3972 | 0.175 | -0.0079 | 0.107 | 0.067 | 0.008 | 1 | 3 | 0.003 |
| 1169 | 0.171 | -0.0029 | 0.083 | 0.088 | 0.003 | 1 | 3 | 0.000 |
| 3943 | 0.168 | -0.0094 | 0.082 | 0.086 | 0.012 | 2 | 3 | 0.000 |
| 1095 | 0.166 | -0.0020 | 0.079 | 0.087 | 0.002 | 2 | 3 | 0.000 |
| 1177 | 0.163 | -0.0026 | 0.074 | 0.088 | 0.003 | 1 | 3 | 0.000 |

凡例:
- `pose_jump_m`: 前ノードからのポーズ並進ジャンプ。
- `score`: 採択 refined candidate のスコア。
- `pred_dx_m` / `pred_dyaw_rad`: keyframe pose と prediction の差。motion prior の誤差。
- `icp_rms`: 採択 candidate の ICP 最終 RMS。
- `best_idx`: yaw 候補 3 つ (`[0, +δ, -δ]`, δ=0.0349 rad=2deg) のうち選ばれた index。
- `n_ref`: refined candidate 数 (常に 3 = yaw top-3)。
- `score_gap`: 採択候補と 2 位候補のスコア差。0 はほぼ同点。

## ノードクラスタ

ホットスポット 20 件は実質 3 つの局所範囲に偏る:

1. ノード 936–941 (+ 1094–1095, 1169–1177, 1032 を含む 900–1180 帯) ── yaw 旋回期。
2. ノード 3771–3772 ── 短い大ジャンプ。
3. ノード 3941–3972 + 4076 ── 別の yaw 旋回期。

## 観察

- **クラスタ 1 / 3 の共通特性 (yaw 旋回中)**
  - `pred_dyaw_rad ≈ 0.085 – 0.122 rad (5–7 deg)` で **prediction が yaw を取りこぼしている**。
  - yaw 候補は `[0, ±2deg]` の 3 通りしかなく、**実 yaw 差を coarse 探索でカバーできていない**。
  - それでも `icp_rms` は 0.002–0.009 m と小さく、**ICP が後付けで yaw を吸収している**。
  - `score_gap ≈ 0` で 3 候補は実質同点 → `best_idx` が 0/1/2 でフラフラする (936=0, 937=2, 940=0, 941=0, 1094=2 等)。
  - つまり「coarse の選択は意味を成していない」「精度は ICP に依存」「pose_jump はその副作用」。
  - 失敗モード: **motion prior failure (旋回中の yaw 予測弱)**。Direction C 寄り。

- **ノード 3771 (single big jump)**
  - `pred_dx_m = 0.254 m` で並進 prediction が大幅にずれている。
  - `icp_rms = 0.153 m` も他より一桁高く、**ICP も収束できていない**。
  - 直後の 3772 も `pred_dx_m = 0.177 m`, `icp_rms = 0.012` で取り戻している。
  - 失敗モード: **prediction / matcher 両方が一瞬崩れた局所イベント**。観測 (人の通過 / 反射) か prediction の不連続が疑わしい。Direction B (motion plausibility filter) または Direction A (candidate ambiguity gate) の対象になりにくく、原因切り分けには raw cloud / submap entropy のさらなる調査が要る。

## 結論 (Task 1 短評)

- 「pose_jump 上位 20 ノード」は 8 割が `pred_dyaw_rad >> coarse yaw step` の旋回期に固まっており、**candidate ambiguity というより motion prior (yaw) の不足**が支配的。
- candidate を増やしても (`score_gap=0` のため) どの候補も同点近傍で、ATE を犠牲にして局所平滑を稼ぐ過去の B&B 拡張結果 (plan.md `True B&B Candidate Expansion`) と整合する。
- 単独イベント 3771 だけは旋回起因ではなく、別系統 (観測異常か prediction 不連続) の局所事故と見るのが妥当。

## 推奨される次手 (plan.md Direction との対応)

- 最有望: **Direction C** ── 旋回中の yaw prediction 強化。具体には IMU 不使用の現状で `prediction_delta_yaw_rad p90` を縮めるレシピを試す (例: 直近 k ノードの yaw 速度推定を改善 / yaw 探索 step を旋回期だけ広げる)。
- 並行: **Direction A** の狭義版 ── `score_gap=0` のときに `prediction_delta_yaw_rad` の小さい候補を採るルールを追加。今のところ候補スコア自体に差がないため、副作用が小さい狭い変更で済む可能性が高い。
- 候補数を増やす広い変更 (`candidate_limit=10`, gain tuning) は過去に full-run ATE を悪化させた実績があるため見送り。
- ノード 3771 は単独現象として別途精査 (生スキャン / submap entropy / 直前 pose) し、汎用ルールではなく原因記述優先。

## メモ

- 本テーブルは新フラグ `cloud-analyze --scan-match-hotspots N` (今回 PR で追加) の出力で再生成可能。
- plan.md Task 1 「指定 8 ノード (3771, 3941, 1032, 1176, 940, 1094, 936, 937)」はすべて top 8 に入っている。
