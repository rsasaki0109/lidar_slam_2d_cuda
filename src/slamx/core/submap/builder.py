from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slamx.core.types import LaserScan, Pose2, Submap, transform_points_xy


@dataclass
class SubmapBuilderConfig:
    max_submap_scans: int = 30
    downsample_stride: int = 2


@dataclass
class SubmapBuilder:
    cfg: SubmapBuilderConfig = field(default_factory=SubmapBuilderConfig)
    _next_id: int = 0

    def build(
        self,
        *,
        submap_node_id: int,
        origin: Pose2,
        scans_map: list[tuple[Pose2, LaserScan]],
    ) -> Submap:
        hits: list[np.ndarray] = []
        for pose, scan in scans_map:
            pts_s = scan.points_xy()
            if pts_s.size == 0:
                continue
            if self.cfg.downsample_stride > 1:
                pts_s = pts_s[:: self.cfg.downsample_stride]
            T = pose.as_se2()
            hits.append(transform_points_xy(T, pts_s))
        if not hits:
            xy = np.zeros((0, 2), dtype=np.float64)
        else:
            xy = np.concatenate(hits, axis=0)
        sm = Submap(id=self._next_id, node_id=submap_node_id, origin=origin, hit_xy_map=xy)
        self._next_id += 1
        return sm
