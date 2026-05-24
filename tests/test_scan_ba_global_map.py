from __future__ import annotations

import numpy as np

from slamx.core.scan_ba.global_map import GlobalTsdfMap
from slamx.core.scan_ba.tsdf import Tsdf2DConfig
from slamx.core.types import Pose2

from tests.test_scan_ba_align import _raycast_scan


def _cfg() -> Tsdf2DConfig:
    return Tsdf2DConfig(
        resolution_m=0.05, origin_x_m=-2.0, origin_y_m=-2.0, size_x_m=12.0, size_y_m=12.0, truncation_m=0.5
    )


def _gt():
    return [Pose2(2.0 + 0.25 * i, 1.5 + 0.15 * i, 0.05 * i) for i in range(5)]


def test_integrate_matches_rebuild():
    """Folding scans one-by-one online gives the same map as a batch rebuild from the
    same poses -- rebuild is just a clear + replay, so the two must be bit-identical."""
    gt = _gt()
    scans = [_raycast_scan(p) for p in gt]

    inc = GlobalTsdfMap(_cfg())
    for p, sc in zip(gt, scans):
        inc.integrate(p, sc)

    batch = GlobalTsdfMap(_cfg())
    batch.rebuild(gt, scans)

    np.testing.assert_array_equal(inc.tsdf.phi, batch.tsdf.phi)
    np.testing.assert_array_equal(inc.tsdf.weight, batch.tsdf.weight)
    assert np.any(inc.tsdf.weight > 0)


def test_rebuild_after_correction_drops_drift():
    """A map folded at drifted poses then rebuilt at the corrected poses must equal a
    map that only ever saw the corrected poses -- i.e. rebuild fully discards the
    pre-correction (drifted) accumulation. This is the loop-closure consistency the
    engine relies on after a pose-graph solve."""
    gt = _gt()
    scans = [_raycast_scan(p) for p in gt]
    drifted = [Pose2(p.x + 0.4, p.y - 0.3, p.theta + 0.1) for p in gt]

    m = GlobalTsdfMap(_cfg())
    for p, sc in zip(drifted, scans):
        m.integrate(p, sc)
    assert np.any(m.tsdf.weight > 0)
    m.rebuild(gt, scans)  # loop closure corrected the trajectory

    ref = GlobalTsdfMap(_cfg())
    ref.rebuild(gt, scans)
    np.testing.assert_array_equal(m.tsdf.phi, ref.tsdf.phi)
    np.testing.assert_array_equal(m.tsdf.weight, ref.tsdf.weight)


def test_occupancy_render_classes():
    """The occupancy render must produce all three ROS classes (free/occupied/unknown)
    and mark unobserved cells unknown."""
    gt = _gt()
    scans = [_raycast_scan(p) for p in gt]
    m = GlobalTsdfMap(_cfg())
    m.rebuild(gt, scans)

    img = m.to_occupancy_u8()
    vals = set(np.unique(img).tolist())
    assert vals <= {0, 205, 254}
    assert 0 in vals and 254 in vals and 205 in vals  # occupied, free, unknown all present
    # cells never observed (weight 0) stay unknown
    n_unknown_img = int((img == 205).sum())
    n_unobserved = int((m.tsdf.weight == 0).sum())
    assert n_unknown_img == n_unobserved


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
