from __future__ import annotations

import math


def quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract pitch angle (radians) from quaternion. Positive = nose up.

    Uses the standard aerospace ZYX (yaw-pitch-roll) decomposition:
        pitch = arcsin(2*(qw*qy - qz*qx))
    """
    sinp = 2.0 * (qw * qy - qz * qx)
    return math.asin(max(-1.0, min(1.0, sinp)))


def pitch_adjusted_min_range(
    pitch_rad: float,
    sensor_height_m: float,
    bearing_rad: float,
) -> float:
    """Compute minimum valid range for a given bearing considering pitch.

    On a ramp, the laser plane tilts by *pitch_rad*.  Rays pointing toward
    the floor will hit at approximately:

        range = sensor_height / sin(abs(pitch))

    For a 2D scanner the elevation angle of each ray is approximately zero
    in the scanner frame, so the effective tilt of a ray toward the ground
    depends primarily on the pitch itself.  A ``bearing_rad`` dependency is
    intentionally kept minimal (reserved for future 3D scanners) -- here we
    use the absolute pitch directly.

    Returns the range below which a ray is likely hitting the floor.  If
    the geometry makes the floor unreachable (e.g. pitch ~ 0), returns 0.0
    so no rays are filtered.
    """
    sin_abs = abs(math.sin(pitch_rad))
    if sin_abs < 1e-6:
        return 0.0
    return sensor_height_m / sin_abs
