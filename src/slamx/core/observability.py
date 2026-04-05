from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if is_dataclass(x):
        return _json_safe(asdict(x))
    if hasattr(x, "tolist"):
        return x.tolist()
    return repr(x)


class JsonlTelemetry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        rec = {"type": event_type, "schema_version": SCHEMA_VERSION, **_json_safe(payload)}
        line = json.dumps(rec, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
