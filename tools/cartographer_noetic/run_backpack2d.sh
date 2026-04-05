#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <bag_path> <scan_topic> [out_record_bag]"
  exit 2
fi

BAG_PATH="$(realpath "$1")"
SCAN_TOPIC="$2"
OUT_RECORD="${3:-carto_recorded.bag}"
OUT_RECORD="$(realpath -m "$OUT_RECORD")"

IMG="slamx-cartographer-melodic:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[i] Building image $IMG"
docker build -t "$IMG" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

echo "[i] Running Cartographer with bag: $BAG_PATH"
echo "[i] Will record TF into: $OUT_RECORD"

WORKDIR="$(pwd)"

docker run --rm -i \
  --net=host \
  -v "$WORKDIR":"$WORKDIR" \
  -w "$WORKDIR" \
  -e "BAG_PATH=$BAG_PATH" \
  -e "SCAN_TOPIC=$SCAN_TOPIC" \
  -e "OUT_RECORD=$OUT_RECORD" \
  "$IMG" bash -lc '
    set -euo pipefail
    source /opt/ros/melodic/setup.bash
    roscore >/tmp/roscore.log 2>&1 &
    sleep 2
    rosbag record -O "$OUT_RECORD" /tf /tf_static __name:=slamx_tf_record >/tmp/record.log 2>&1 &
    RECORD_PID=$!
    sleep 1
    # Offline node exits when the bag is finished (demo_backpack_2d.launch keeps RViz alive otherwise).
    roslaunch cartographer_ros offline_backpack_2d.launch \
      bag_filenames:="$BAG_PATH" \
      no_rviz:=true \
      >/tmp/carto.log 2>&1 || true
    kill -INT "$RECORD_PID" 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true
    # rosbag sometimes leaves only *.bag.active unless the wrapper exits cleanly
    if [[ -f "${OUT_RECORD}.active" && ! -f "${OUT_RECORD}" ]]; then
      rosbag reindex "${OUT_RECORD}.active" 2>/dev/null || true
      [[ -f "${OUT_RECORD}.active" ]] && cp -f "${OUT_RECORD}.active" "${OUT_RECORD}"
    fi
    rm -f "${OUT_RECORD}.orig.active" 2>/dev/null || true
    echo "[i] Done. Logs: /tmp/carto.log /tmp/record.log"
  '

