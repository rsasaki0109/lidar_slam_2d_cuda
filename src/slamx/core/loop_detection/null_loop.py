from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from slamx.core.types import Pose2


@dataclass(frozen=True)
class LoopCandidate:
    i: int
    j: int
    score: float


@runtime_checkable
class LoopDetector(Protocol):
    def detect(self, *, poses: list[Pose2], stamp_ns: int | None) -> list[LoopCandidate]: ...


class NullLoopDetector:
    def detect(self, *, poses: list[Pose2], stamp_ns: int | None) -> list[LoopCandidate]:
        return []
