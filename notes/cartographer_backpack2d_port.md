## 結論

- **できること**: Cartographer の公開設定（Apache-2.0）に書かれた **数値**を起点に、「どこから持ってきたか」を URL 付きで説明できる `configs/cartographer_backpack2d_port.yaml` を用意した。**slamx と Cartographer はアルゴリズムが同一ではない**ため「同等実装」ではなく **参照ポート（パクり＝出典明示のトレーサビリティ）** として扱う。
- **できないこと**: Ceres 格子マッチ・姿勢グラフのハブロス等は slamx にそのまま無い。**対応は「一番近いキーに数値を寄せる」**に留まる。

## 確認済み事実（出典）

| Cartographer（Lua） | 値（概要） | slamx での対応先 |
|---------------------|------------|------------------|
| `trajectory_builder_2d.lua` の `max_range` | `30.` m | `slam.preprocess.max_range` |
| 同 `voxel_filter_size` | `0.025` m | **直接なし** → `slam.preprocess.stride` で間引き（コメントに記載） |
| 同 `real_time_correlative_scan_matcher.linear_search_window` | `0.1` m | `slam.local_matching.correlative_grid.linear_window_m` |
| 同 `angular_search_window` | `math.rad(20.)` | `slam.local_matching.correlative_grid.angular_window_deg` = `20` |
| `submaps.grid_options_2d.resolution` | `0.05` m | `map.occupancy_grid.resolution_m` |
| `pose_graph.lua` の `optimize_every_n_nodes` | `90` | `slam.optimize_every_n_keyframes`（**意味は近いが同一ではない**：slamx はキーフレーム単位） |
| `backpack_2d.lua` の `num_accumulated_range_data` | `10`（echo 合成） | **直接なし**（MultiEcho は slamx 側で先頭エコー化済み） |

- `backpack_2d.lua`: https://raw.githubusercontent.com/cartographer-project/cartographer_ros/master/cartographer_ros/configuration_files/backpack_2d.lua  
- `trajectory_builder_2d.lua`: https://raw.githubusercontent.com/cartographer-project/cartographer/master/configuration_files/trajectory_builder_2d.lua  
- `pose_graph.lua`: https://raw.githubusercontent.com/cartographer-project/cartographer/master/configuration_files/pose_graph.lua  

## 未確認 / 要確認

- **ローカル SLAM**: Cartographer は既定で `use_online_correlative_scan_matching = false` かつ `ceres_scan_matcher` 中心。slamx は **格子相関（correlative）** をポート用に使用。**数値は CSM ブロック由来**で Ceres の重み（`translation_weight` 等）とは別物。
- **サブマップ**: Cartographer `submaps.num_range_data = 90` と slamx `max_submap_scans` は構造が違うため、ポート YAML では **中間的な窓長**にしている（メモのみ；最適値は未調査）。

## 次アクション

- 短区間:  
  `slamx replay ... --config configs/cartographer_backpack2d_port.yaml --max-scans 300 ...`
- 長〜全区間:  
  `configs/cartographer_backpack2d_port_full.yaml`（ポート數値 + 長系列用 BA ガード）

絶対パス（ワークスペース）: `lidar_slam_2d/notes/cartographer_backpack2d_port.md`
