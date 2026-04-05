from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeterminismContext:
    enabled: bool
    seed: int


def apply_determinism(ctx: DeterminismContext) -> None:
    if not ctx.enabled:
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    np.random.seed(ctx.seed)
