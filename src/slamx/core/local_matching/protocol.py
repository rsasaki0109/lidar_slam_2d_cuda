from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from slamx.core.types import LaserScan, MatchResult, Pose2


@runtime_checkable
class ScanMatcher(Protocol):
    def match(
        self,
        *,
        scan: LaserScan,
        prediction_map: Pose2,
        ref_points_xy_map: np.ndarray,
    ) -> MatchResult: ...
