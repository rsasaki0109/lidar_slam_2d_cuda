from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from slamx.core.types import LaserScan


def iter_scans_jsonl(path: Path) -> Iterator[LaserScan]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            yield _scan_from_obj(obj)


def _scan_from_obj(obj: dict) -> LaserScan:
    required = ("angle_min", "angle_max", "angle_increment", "ranges")
    for k in required:
        if k not in obj:
            raise KeyError(f"scan missing {k}")
    return LaserScan(
        stamp_ns=obj.get("stamp_ns"),
        frame_id=str(obj.get("frame_id", "laser")),
        angle_min=float(obj["angle_min"]),
        angle_max=float(obj["angle_max"]),
        angle_increment=float(obj["angle_increment"]),
        ranges=np.asarray(obj["ranges"], dtype=np.float64),
        range_min=float(obj.get("range_min", 0.0)),
        range_max=float(obj.get("range_max", 1e9)),
    )
