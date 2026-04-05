from __future__ import annotations

import json
from pathlib import Path

from slamx.core.types import Pose2


def save_trajectory_json(path: Path, poses: list[Pose2], stamps_ns: list[int | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"i": i, "stamp_ns": s, "x": p.x, "y": p.y, "theta": p.theta}
        for i, (p, s) in enumerate(zip(poses, stamps_ns, strict=False))
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
