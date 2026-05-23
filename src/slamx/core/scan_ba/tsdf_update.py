from __future__ import annotations

import numpy as np

from slamx.core.scan_ba.tsdf import Tsdf2D
from slamx.core.types import Pose2


def update_tsdf_from_scan(
    tsdf: Tsdf2D,
    *,
    pose_map: Pose2,
    points_sensor: np.ndarray,
    truncation_m: float | None = None,
    weight_inc: float = 1.0,
    weight_max: float = 100.0,
) -> int:
    """Volumetric weighted-average TSDF update from one scan.

    For each hit point in the sensor frame:
      - Transform to map frame with pose_map.
      - For voxels within `truncation_m` of the hit, compute signed distance
        along the ray (positive on sensor side of the surface).
      - Update phi and weight with KinectFusion-style weighted average.

    Returns the number of voxels touched.
    """
    if points_sensor.size == 0:
        return 0
    trunc = float(truncation_m if truncation_m is not None else tsdf.cfg.truncation_m)
    res = float(tsdf.cfg.resolution_m)
    ox = float(tsdf.cfg.origin_x_m)
    oy = float(tsdf.cfg.origin_y_m)
    h = tsdf.height
    w = tsdf.width

    c, s = float(np.cos(pose_map.theta)), float(np.sin(pose_map.theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = np.array([pose_map.x, pose_map.y], dtype=np.float64)
    hits = points_sensor @ R.T + t  # (N, 2)
    sensor = t

    d = hits - sensor  # (N, 2)
    dn = np.linalg.norm(d, axis=1)
    valid_dir = dn > 1e-6
    d_unit = np.zeros_like(d)
    d_unit[valid_dir] = d[valid_dir] / dn[valid_dir, None]

    rc = int(np.ceil(trunc / res))
    di = np.arange(-rc, rc + 1, dtype=np.int64)
    dj = np.arange(-rc, rc + 1, dtype=np.int64)
    ddi, ddj = np.meshgrid(di, dj, indexing="ij")
    ddi_f = ddi.ravel()
    ddj_f = ddj.ravel()

    hi = np.floor((hits[:, 0] - ox) / res - 0.5).astype(np.int64)
    hj = np.floor((hits[:, 1] - oy) / res - 0.5).astype(np.int64)

    Hi = hi[:, None] + ddi_f[None, :]  # (N, M)
    Hj = hj[:, None] + ddj_f[None, :]
    in_bounds = (Hi >= 0) & (Hi < w) & (Hj >= 0) & (Hj < h)

    vx = ox + (Hi + 0.5) * res
    vy = oy + (Hj + 0.5) * res

    dx_to_hit = hits[:, 0:1] - vx  # (N, M)
    dy_to_hit = hits[:, 1:2] - vy
    s_dist = dx_to_hit * d_unit[:, 0:1] + dy_to_hit * d_unit[:, 1:2]
    dist = np.hypot(dx_to_hit, dy_to_hit)

    mask = in_bounds & (dist <= trunc) & valid_dir[:, None]
    s_dist = np.clip(s_dist, -trunc, trunc)

    flat_idx = Hj * w + Hi
    sel_idx = flat_idx[mask].astype(np.int64)
    sel_dist = s_dist[mask].astype(np.float64)
    if sel_idx.size == 0:
        return 0

    flat_phi = tsdf.phi.ravel().astype(np.float64)
    flat_w = tsdf.weight.ravel().astype(np.float64)

    sum_w = np.zeros_like(flat_w)
    sum_wd = np.zeros_like(flat_phi)
    w_inc = float(weight_inc)
    np.add.at(sum_w, sel_idx, w_inc)
    np.add.at(sum_wd, sel_idx, w_inc * sel_dist)

    touched = sum_w > 0
    new_w = np.minimum(flat_w + sum_w, float(weight_max))
    new_phi = flat_phi.copy()
    denom = flat_w + sum_w
    new_phi[touched] = (
        flat_phi[touched] * flat_w[touched] + sum_wd[touched]
    ) / denom[touched]

    tsdf.phi[:] = new_phi.astype(np.float32).reshape(tsdf.phi.shape)
    tsdf.weight[:] = new_w.astype(np.float32).reshape(tsdf.weight.shape)
    return int(touched.sum())
