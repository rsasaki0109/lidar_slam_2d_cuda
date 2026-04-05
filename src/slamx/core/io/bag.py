from __future__ import annotations

from pathlib import Path
from typing import Iterator

from slamx.core.types import LaserScan


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
