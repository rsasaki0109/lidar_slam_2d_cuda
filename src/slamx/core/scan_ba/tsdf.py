from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tsdf2DConfig:
    resolution_m: float = 0.05
    origin_x_m: float = -20.0
    origin_y_m: float = -20.0
    size_x_m: float = 40.0
    size_y_m: float = 40.0
    truncation_m: float = 0.4


@dataclass
class Tsdf2D:
    """2D Truncated Signed Distance Field with bilinear sampling.

    Convention: phi > 0 on the free-space side of the surface, phi < 0 inside,
    phi == 0 on the surface. `weight` == 0 marks unobserved cells.
    """

    cfg: Tsdf2DConfig
    phi: np.ndarray  # (H, W) float32
    weight: np.ndarray  # (H, W) float32

    def __post_init__(self) -> None:
        if self.phi.shape != self.weight.shape:
            raise ValueError("phi and weight must share shape")
        if self.phi.ndim != 2:
            raise ValueError("phi must be 2D")
        self.phi = np.ascontiguousarray(self.phi, dtype=np.float32)
        self.weight = np.ascontiguousarray(self.weight, dtype=np.float32)

    @classmethod
    def zeros(cls, cfg: Tsdf2DConfig) -> Tsdf2D:
        h = int(np.ceil(cfg.size_y_m / cfg.resolution_m))
        w = int(np.ceil(cfg.size_x_m / cfg.resolution_m))
        return cls(cfg=cfg, phi=np.zeros((h, w), dtype=np.float32), weight=np.zeros((h, w), dtype=np.float32))

    @property
    def height(self) -> int:
        return int(self.phi.shape[0])

    @property
    def width(self) -> int:
        return int(self.phi.shape[1])

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinearly sample phi and its gradient at world-frame points.

        Returns (phi_val[N], grad[N,2], valid_mask[N]).
        valid_mask is True iff all four corner cells lie in bounds and have weight > 0.
        """
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be (N, 2)")
        n = xy.shape[0]
        if n == 0:
            return (
                np.zeros((0,), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=bool),
            )

        res = float(self.cfg.resolution_m)
        ox, oy = float(self.cfg.origin_x_m), float(self.cfg.origin_y_m)
        h, w = self.height, self.width

        # continuous grid coords; values are stored at cell centers, so shift by 0.5
        gx = (xy[:, 0] - ox) / res - 0.5
        gy = (xy[:, 1] - oy) / res - 0.5
        ix0 = np.floor(gx).astype(np.int64)
        iy0 = np.floor(gy).astype(np.int64)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        in_bounds = (ix0 >= 0) & (ix1 < w) & (iy0 >= 0) & (iy1 < h)

        # clamp for safe indexing; we'll mask out invalid entries afterwards
        ix0c = np.clip(ix0, 0, w - 1)
        ix1c = np.clip(ix1, 0, w - 1)
        iy0c = np.clip(iy0, 0, h - 1)
        iy1c = np.clip(iy1, 0, h - 1)

        phi = self.phi
        wgt = self.weight
        p00 = phi[iy0c, ix0c].astype(np.float64)
        p10 = phi[iy0c, ix1c].astype(np.float64)
        p01 = phi[iy1c, ix0c].astype(np.float64)
        p11 = phi[iy1c, ix1c].astype(np.float64)
        w00 = wgt[iy0c, ix0c]
        w10 = wgt[iy0c, ix1c]
        w01 = wgt[iy1c, ix0c]
        w11 = wgt[iy1c, ix1c]

        fx = (gx - ix0).astype(np.float64)
        fy = (gy - iy0).astype(np.float64)
        one_fx = 1.0 - fx
        one_fy = 1.0 - fy

        phi_val = one_fx * one_fy * p00 + fx * one_fy * p10 + one_fx * fy * p01 + fx * fy * p11
        # Gradient of bilinear interpolation; df/dfx * dfx/dx = (...) / res
        dphi_dx = ((-one_fy) * p00 + one_fy * p10 + (-fy) * p01 + fy * p11) / res
        dphi_dy = ((-one_fx) * p00 + (-fx) * p10 + one_fx * p01 + fx * p11) / res
        grad = np.column_stack((dphi_dx, dphi_dy))

        valid = in_bounds & (w00 > 0) & (w10 > 0) & (w01 > 0) & (w11 > 0)
        return phi_val, grad, valid


def build_tsdf_from_signed_distance(
    cfg: Tsdf2DConfig,
    sdf_fn,
    *,
    weight_mask_fn=None,
) -> Tsdf2D:
    """Bake a TSDF from an analytical signed-distance function.

    sdf_fn(x, y) -> phi; truncated to [-truncation, +truncation].
    weight_mask_fn(x, y) -> bool array; defaults to True everywhere within truncation band.
    """
    tsdf = Tsdf2D.zeros(cfg)
    res = float(cfg.resolution_m)
    ox, oy = float(cfg.origin_x_m), float(cfg.origin_y_m)
    h, w = tsdf.height, tsdf.width
    xs = ox + (np.arange(w, dtype=np.float64) + 0.5) * res
    ys = oy + (np.arange(h, dtype=np.float64) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    raw = sdf_fn(xx, yy)
    trunc = float(cfg.truncation_m)
    phi = np.clip(raw, -trunc, trunc).astype(np.float32)
    if weight_mask_fn is None:
        weight = (np.abs(raw) <= trunc).astype(np.float32)
    else:
        weight = weight_mask_fn(xx, yy).astype(np.float32)
    tsdf.phi[:] = phi
    tsdf.weight[:] = weight
    return tsdf
