from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamx.core.types import Pose2


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from quaternion (assumes ENU, z-up)."""
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@dataclass(frozen=True)
class TFEdge2D:
    parent: str
    child: str
    t: Pose2  # parent_T_child


class TFBuffer2D:
    """Very small TF buffer for 2D (x,y,yaw). Stores latest transforms."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], Pose2] = {}

    def set_transform(self, parent: str, child: str, t: Pose2) -> None:
        self._edges[(parent, child)] = t

    def can_transform(self, parent: str, child: str) -> bool:
        return self.lookup(parent, child) is not None

    def lookup(self, parent: str, child: str, *, max_depth: int = 6) -> Pose2 | None:
        if parent == child:
            return Pose2(0.0, 0.0, 0.0)
        direct = self._edges.get((parent, child))
        if direct is not None:
            return direct

        # BFS over directed edges, also allow using inverse edges
        from collections import deque

        q = deque([(parent, Pose2(0.0, 0.0, 0.0), 0)])
        visited = {parent}

        while q:
            cur_frame, cur_T, depth = q.popleft()
            if depth >= max_depth:
                continue
            # outgoing
            for (p, c), T_pc in list(self._edges.items()):
                if p == cur_frame and c not in visited:
                    nxt_T = cur_T.compose(T_pc)
                    if c == child:
                        return nxt_T
                    visited.add(c)
                    q.append((c, nxt_T, depth + 1))
                if c == cur_frame and p not in visited:
                    # use inverse edge
                    T_cp = T_pc.inverse()
                    nxt_T = cur_T.compose(T_cp)
                    if p == child:
                        return nxt_T
                    visited.add(p)
                    q.append((p, nxt_T, depth + 1))
        return None

