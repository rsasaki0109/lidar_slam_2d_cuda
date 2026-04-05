from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    sha256: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(spec: DownloadSpec, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / spec.name
    if dst.exists() and spec.sha256:
        if _sha256(dst).lower() == spec.sha256.lower():
            return dst

    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(spec.url) as r, tmp.open("wb") as f:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            f.write(b)
    tmp.replace(dst)

    if spec.sha256:
        got = _sha256(dst)
        if got.lower() != spec.sha256.lower():
            raise ValueError(f"sha256 mismatch for {dst}: got {got} expected {spec.sha256}")
    return dst


def cartographer_backpack2d_bag(bag_name: str) -> DownloadSpec:
    base = "https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/"
    return DownloadSpec(name=bag_name, url=base + bag_name, sha256=None)

