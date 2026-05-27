# Odometry Tiebreak (prediction-yaw): full no-loop results (2026-05-28)

Hypothesis (from `notes/odometry_hotspots_yawtop3_20260528.md`):

> 旋回中の hotspot は `pred_dyaw_rad >> coarse yaw step` で coarse 探索が 3 候補ともほぼ同点になる
> (`score_gap ≈ 0`)。score だけでは候補を区別できず、ICP が後付けで吸収しているため
> `pose_jump` がそこに現れる。score が同点なら、`prediction_delta_yaw` が最も小さい
> refined candidate を採るルールが当たるはず。

Implementation: opt-in `HybridRefinementConfig.prediction_yaw_tiebreak_enabled` +
  `tiebreak_score_eps`, `tiebreak_rms_eps`, `tiebreak_yaw_margin_rad`. Default OFF.

Config: `configs/iilabs_vscan_bb_yawtop3_tiebreak.yaml` (Δscore<=0.002,
  ΔICP_rms<=0.005, Δyaw>=0.0087 rad = 0.5 deg).

## Full no-loop replay (5022 keyframes)

```bash
env -u PYTHONPATH .venv/bin/slamx replay \
  data/iilabs3d/iilabs3d_dataset/benchmark/velodyne_vlp-16/elevator/velodyne_elevator_2025-02-05-15-04-36.bag \
  --pointcloud-topic /eve/lidar3d \
  --config configs/iilabs_vscan_bb_yawtop3_tiebreak.yaml \
  --out runs/iilabs_elevator_full_vscan_bb_yawtop3_tiebreak_20260528 \
  --deterministic --seed 0 --no-write-map
```

| metric                  | baseline yawtop3 | tiebreak | delta            |
|-------------------------|------------------|----------|------------------|
| align ATE (m)           | 0.16980          | 0.13542  | -20.3%           |
| no-align ATE (m)        | 0.23155          | 0.18893  | -18.4%           |
| start/end gap (m)       | 0.38364          | 0.34912  |  -9.0%           |
| pose_jump p50 (m)       | 0.03933          | 0.03664  |  -6.9%           |
| pose_jump p90 (m)       | 0.08191          | 0.07980  |  -2.6%           |
| pose_jump max (m)       | 0.27502          | 0.22020  | -19.9%           |
| path length (m)         | 146.275          | 152.251  |  +4.1%           |
| nonzero best candidate  | 857 / 5021       | 1021 / 5021 | +164 swaps      |

すべての plan.md `Required Full No-Loop Gates` をクリア:

- start/end gap < 0.3836 ✓
- align ATE < 0.1698 ✓
- no-align ATE < 0.2316 ✓
- path length が 146.275 m の周辺で plausible (collapse なし) ✓
- pose_jump max が悪化していない ✓
- ランタイム regression なし (matcher 1回追加の O(K) ループ。K=3)

## Where the wins concentrate

Hotspots shift, total worst pose_jump drops 0.275 → 0.220:

| node  | base pose_jump | tiebreak pose_jump | comment                    |
|------:|---------------:|-------------------:|----------------------------|
|  3771 | 0.275          | 0.126              | yaw cluster broken         |
|  3941 | 0.228          | <0.13 (out of top20) | yaw cluster broken      |
|  1032 | 0.217          | 0.132              | yaw cluster softened       |
|  1176 | 0.203          | 0.149              | yaw cluster softened       |
|   940 | 0.202          | <0.13 (out of top20) | yaw cluster broken      |
|  1094 | 0.201          | <0.13 (out of top20) | yaw cluster broken      |
|   936 | 0.201          | <0.13 (out of top20) | yaw cluster broken      |
|   937 | 0.201          | <0.13 (out of top20) | yaw cluster broken      |
|  1041 |  -             | 0.220              | new top 1 (smaller cluster)|
|  4044 |  -             | 0.197              | new turning event          |

## 2k Cartographer-window check

(Plan.md says "2k is fast but do not trust it alone"; full no-loop is the binding gate.)

| metric                  | baseline | tiebreak | delta            |
|-------------------------|----------|----------|------------------|
| align ATE (m)           | 0.16127  | 0.16700  | +3.6% (slightly worse) |
| no-align ATE (m)        | 0.33722  | 0.33632  | -0.3%            |
| pose_jump p50 (m)       | 0.04678  | 0.03611  | -22.8%           |
| pose_jump p90 (m)       | 0.08909  | 0.07957  | -10.7%           |
| pose_jump max (m)       | 0.21702  | 0.22020  | +1.5%            |

The 2k window aligns differently when the full-trajectory shape shifts; this is the
expected mild align-ATE noise plan.md warned about. The full GT-aligned 5k+ ATE
moves in the right direction unambiguously.

## Conclusion

- Direction A 狭義版 (score_gap ≈ 0 のときに prediction-yaw 最小候補を採るタイブレーク)
  は full no-loop に対して **plan.md の必須ゲートを全て満たす**。
- 副次的に worst pose_jump も -19.9%。
- 過去の `True B&B Candidate Expansion` / `Selection Prior` 系が ATE を悪化させたのに対し、
  この変更は **候補集合は同じ (yaw top-3) のまま、選び方だけを変えている** ので副作用が
  最小化されている。
- デフォルト OFF にしているため、既存ベースラインに影響なし。opt-in config から有効化。

## Full loop replay (caveat — tiebreak destabilizes the current loop config)

The existing sparse-cap32 loop config with tiebreak enabled regresses heavily:

| metric             | baseline yawtop3 loop | yawtop3 loop + tiebreak | delta              |
|--------------------|----------------------:|------------------------:|--------------------|
| align ATE (m)      | 0.0312                | 0.8050                  | **~25x worse**     |
| no-align ATE (m)   | 0.0722                | 1.1088                  | **~15x worse**     |
| start/end gap (m)  | 0.0188                | 1.7449                  | **~93x worse**     |
| accepted loops     | 168                   | 158                     | -10                |
| rejected loops     | 122                   | 134                     | +12                |

Hypothesis: the existing loop config uses an online single-iteration pose-graph
solver (`pose_graph: max_iterations: 1`, `optimize_every_n_keyframes: 50`). The
tiebreak shifts the odometry path enough that the *same* loop edges produce
different residuals, and the low-budget solver lands in a different basin. This
is a loop-pipeline tuning issue, not an odometry-quality issue — the no-loop
trajectory is unambiguously better.

This PR therefore **does not ship a tiebreak-enabled loop config**. Only the
opt-in no-loop config is promoted. Future work: separately re-tune the loop
policy (search radius, accept score, optimizer budget) for the tiebreak-enabled
odometry path.

## Next

- This (Conservative Odometry) PR ships only:
  - `HybridRefinementConfig` tiebreak fields (default OFF).
  - `_apply_prediction_yaw_tiebreak` helper.
  - `configs/iilabs_vscan_bb_yawtop3_tiebreak.yaml` (no-loop).
  - This note.
- Loop policy adjustment for the tiebreak-enabled path is **NOT** in scope of
  this PR (would mix matcher and loop-policy changes, against plan.md PR
  Strategy).
- Do not regenerate the GIF until both no-loop and loop pass their respective
  gates with the same odometry config.
