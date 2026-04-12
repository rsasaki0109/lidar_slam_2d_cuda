from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from slamx.core.types import LaserScan


@dataclass
class ImuSample:
    stamp_ns: int
    orientation_quat: tuple[float, float, float, float]  # (x, y, z, w)
    angular_velocity: tuple[float, float, float]  # (x, y, z)


@dataclass
class PointCloud3D:
    """A 3D point cloud extracted from a PointCloud2 message."""
    stamp_ns: int
    frame_id: str
    points_xyz: np.ndarray  # (N, 3)


def _get_anyreader() -> object:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError("Bag IO requires optional dependency: pip install 'slamx[rosbag]'") from e
    return AnyReader


def list_laserscan_topics(path: Path) -> list[tuple[str, str]]:
    """Return [(topic, msgtype)] for 2D scan topics in a bag (.db3 or .bag).

    Includes sensor_msgs/(msg/)LaserScan and MultiEchoLaserScan.
    """
    AnyReader = _get_anyreader()
    with AnyReader([path]) as reader:
        conns = list(getattr(reader, "connections"))
    def is_2d_scan(msgtype: str) -> bool:
        mt = msgtype.strip()
        last = mt.split("/")[-1] if mt else ""
        return last in {"LaserScan", "MultiEchoLaserScan"}

    scan_conns = [c for c in conns if is_2d_scan(str(getattr(c, "msgtype", "")))]
    out = sorted({(c.topic, c.msgtype) for c in scan_conns})
    return list(out)


def list_topics_by_suffix(path: Path, *, msgtype_suffix: str) -> list[tuple[str, str]]:
    """Return [(topic, msgtype)] for connections whose msgtype endswith suffix."""
    AnyReader = _get_anyreader()
    with AnyReader([path]) as reader:
        conns = list(getattr(reader, "connections"))
    out = sorted(
        {
            (c.topic, c.msgtype)
            for c in conns
            if str(getattr(c, "msgtype", "")).endswith(msgtype_suffix)
        }
    )
    return list(out)


def iter_scans_bag(
    path: Path,
    *,
    topic: str | None = None,
) -> Iterator[LaserScan]:
    """Iterate LaserScan from either ROS1 .bag or ROS2 .db3 using AnyReader."""
    AnyReader = _get_anyreader()
    with AnyReader([path]) as reader:
        conns = list(getattr(reader, "connections"))
        def is_2d_scan(msgtype: str) -> bool:
            mt = msgtype.strip()
            last = mt.split("/")[-1] if mt else ""
            return last in {"LaserScan", "MultiEchoLaserScan"}
        scan_conns = [
            c
            for c in conns
            if is_2d_scan(str(getattr(c, "msgtype", "")))
        ]
        if not scan_conns:
            raise ValueError("No LaserScan connections found in bag.")

        if topic is None:
            preferred = [c for c in scan_conns if c.topic in {"/scan", "scan"}]
            use = preferred[0] if preferred else scan_conns[0]
        else:
            match = [c for c in scan_conns if c.topic == topic]
            if not match:
                topics = sorted({c.topic for c in scan_conns})
                raise ValueError(f"LaserScan topic not found: {topic}. Available: {topics}")
            use = match[0]

        for _conn, ts, raw in reader.messages(connections=[use]):
            msg = reader.deserialize(raw, use.msgtype)
            mt_last = str(use.msgtype).split("/")[-1]
            stamp_ns = None
            if getattr(msg, "header", None) is not None and getattr(msg.header, "stamp", None) is not None:
                st = msg.header.stamp
                sec = int(getattr(st, "sec", getattr(st, "secs", 0)))
                nsec = int(getattr(st, "nanosec", getattr(st, "nsecs", 0)))
                stamp_ns = sec * 1_000_000_000 + nsec
            else:
                stamp_ns = int(ts)

            frame_id = "laser"
            if getattr(msg, "header", None) is not None:
                frame_id = str(getattr(msg.header, "frame_id", "")) or "laser"

            ranges = msg.ranges
            if mt_last == "MultiEchoLaserScan":
                # sensor_msgs/MultiEchoLaserScan: ranges is LaserEcho[]
                flat = []
                for le in ranges:
                    echoes = getattr(le, "echoes", None)
                    if echoes is None or len(echoes) == 0:
                        flat.append(float("nan"))
                    else:
                        flat.append(float(echoes[0]))
                ranges = flat

            yield LaserScan(
                stamp_ns=stamp_ns,
                frame_id=frame_id,
                angle_min=float(msg.angle_min),
                angle_max=float(msg.angle_max),
                angle_increment=float(msg.angle_increment),
                ranges=list(ranges),
                range_min=float(msg.range_min),
                range_max=float(msg.range_max),
            )


def iter_scans_db3(
    path: Path,
    *,
    topic: str | None = None,
) -> Iterator[LaserScan]:
    """Iterate ROS 2 bag2 (.db3) LaserScan messages."""
    if path.suffix.lower() != ".db3":
        raise ValueError(f"Not a .db3: {path}")
    yield from iter_scans_bag(path, topic=topic)


def iter_scans_bag1(
    path: Path,
    *,
    topic: str | None = None,
) -> Iterator[LaserScan]:
    """Iterate ROS 1 bag (.bag) LaserScan messages."""
    if path.suffix.lower() != ".bag":
        raise ValueError(f"Not a .bag: {path}")
    yield from iter_scans_bag(path, topic=topic)


def _extract_xyz_from_pointcloud2(msg: object) -> np.ndarray:
    """Extract (N, 3) xyz array from a deserialized PointCloud2 message.

    Handles both big- and little-endian data via the ``is_bigendian`` flag and
    uses the field offsets/datatypes advertised in the message.
    """
    # PointCloud2 datatype enum -> struct format char (ROS convention)
    _DTYPE_TO_FMT = {
        1: "b",  # INT8
        2: "B",  # UINT8
        3: "h",  # INT16
        4: "H",  # UINT16
        5: "i",  # INT32
        6: "I",  # UINT32
        7: "f",  # FLOAT32
        8: "d",  # FLOAT64
    }

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float64)

    point_step = int(msg.point_step)
    width = int(msg.width)
    height = int(msg.height)
    n_points = width * height
    data = bytes(msg.data)
    endian = ">" if msg.is_bigendian else "<"

    xyz_info = []
    for name in ("x", "y", "z"):
        f = fields[name]
        fmt_char = _DTYPE_TO_FMT.get(int(f.datatype), "f")
        xyz_info.append((int(f.offset), fmt_char))

    points = np.empty((n_points, 3), dtype=np.float64)
    for i in range(n_points):
        base = i * point_step
        for j, (offset, fmt_char) in enumerate(xyz_info):
            val = struct.unpack_from(endian + fmt_char, data, base + offset)[0]
            points[i, j] = val

    # Filter out NaN / inf points
    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def iter_pointcloud2_bag(
    path: Path,
    *,
    topic: str | None = None,
) -> Iterator[PointCloud3D]:
    """Iterate PointCloud2 messages from a ROS bag (.db3 or .bag), yielding PointCloud3D."""
    AnyReader = _get_anyreader()
    with AnyReader([path]) as reader:
        conns = list(getattr(reader, "connections"))

        def is_pc2(msgtype: str) -> bool:
            mt = msgtype.strip()
            last = mt.split("/")[-1] if mt else ""
            return last == "PointCloud2"

        pc2_conns = [c for c in conns if is_pc2(str(getattr(c, "msgtype", "")))]
        if not pc2_conns:
            raise ValueError("No PointCloud2 connections found in bag.")

        if topic is None:
            use = pc2_conns[0]
        else:
            match = [c for c in pc2_conns if c.topic == topic]
            if not match:
                topics = sorted({c.topic for c in pc2_conns})
                raise ValueError(
                    f"PointCloud2 topic not found: {topic}. Available: {topics}"
                )
            use = match[0]

        for _conn, ts, raw in reader.messages(connections=[use]):
            msg = reader.deserialize(raw, use.msgtype)

            stamp_ns = int(ts)
            if (
                getattr(msg, "header", None) is not None
                and getattr(msg.header, "stamp", None) is not None
            ):
                st = msg.header.stamp
                sec = int(getattr(st, "sec", getattr(st, "secs", 0)))
                nsec = int(getattr(st, "nanosec", getattr(st, "nsecs", 0)))
                stamp_ns = sec * 1_000_000_000 + nsec

            frame_id = "lidar"
            if getattr(msg, "header", None) is not None:
                frame_id = str(getattr(msg.header, "frame_id", "")) or "lidar"

            points_xyz = _extract_xyz_from_pointcloud2(msg)
            yield PointCloud3D(
                stamp_ns=stamp_ns,
                frame_id=frame_id,
                points_xyz=points_xyz,
            )


def iter_imu_bag(
    path: Path,
    *,
    topic: str = "/imu",
) -> Iterator[ImuSample]:
    """Iterate IMU messages from a bag file (.db3 or .bag)."""
    AnyReader = _get_anyreader()
    with AnyReader([path]) as reader:
        conns = list(getattr(reader, "connections"))

        def is_imu(msgtype: str) -> bool:
            mt = msgtype.strip()
            last = mt.split("/")[-1] if mt else ""
            return last == "Imu"

        imu_conns = [c for c in conns if is_imu(str(getattr(c, "msgtype", "")))]
        if not imu_conns:
            raise ValueError("No Imu connections found in bag.")

        match = [c for c in imu_conns if c.topic == topic]
        if not match:
            topics = sorted({c.topic for c in imu_conns})
            raise ValueError(
                f"Imu topic not found: {topic}. Available: {topics}"
            )
        use = match[0]

        for _conn, ts, raw in reader.messages(connections=[use]):
            msg = reader.deserialize(raw, use.msgtype)

            stamp_ns = int(ts)
            if (
                getattr(msg, "header", None) is not None
                and getattr(msg.header, "stamp", None) is not None
            ):
                st = msg.header.stamp
                sec = int(getattr(st, "sec", getattr(st, "secs", 0)))
                nsec = int(getattr(st, "nanosec", getattr(st, "nsecs", 0)))
                stamp_ns = sec * 1_000_000_000 + nsec

            ori = getattr(msg, "orientation", None)
            if ori is not None:
                qx = float(getattr(ori, "x", 0.0))
                qy = float(getattr(ori, "y", 0.0))
                qz = float(getattr(ori, "z", 0.0))
                qw = float(getattr(ori, "w", 1.0))
            else:
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            av = getattr(msg, "angular_velocity", None)
            if av is not None:
                avx = float(getattr(av, "x", 0.0))
                avy = float(getattr(av, "y", 0.0))
                avz = float(getattr(av, "z", 0.0))
            else:
                avx, avy, avz = 0.0, 0.0, 0.0

            yield ImuSample(
                stamp_ns=stamp_ns,
                orientation_quat=(qx, qy, qz, qw),
                angular_velocity=(avx, avy, avz),
            )
