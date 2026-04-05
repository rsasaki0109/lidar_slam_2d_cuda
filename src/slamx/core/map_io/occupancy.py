from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def save_pgm(path: Path, img_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img_u8.ndim != 2:
        raise ValueError("PGM expects 2D uint8 image")
    h, w = img_u8.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img_u8.tobytes(order="C"))


def save_occupancy_yaml(
    path: Path,
    *,
    image: str,
    resolution: float,
    origin: list[float],
    negate: int = 0,
    occupied_thresh: float = 0.65,
    free_thresh: float = 0.196,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "image": image,
        "resolution": float(resolution),
        "origin": origin,
        "negate": int(negate),
        "occupied_thresh": float(occupied_thresh),
        "free_thresh": float(free_thresh),
    }
    if extra:
        data.update(extra)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

