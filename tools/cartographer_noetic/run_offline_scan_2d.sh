#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <bag_path> <scan_topic> <lua_config> [out_record_bag]"
  exit 2
fi

BAG_PATH="$(realpath "$1")"
SCAN_TOPIC="$2"
LUA_CONFIG="$(realpath "$3")"
OUT_RECORD="${4:-carto_recorded.bag}"
OUT_RECORD="$(realpath -m "$OUT_RECORD")"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "[e] Bag not found: $BAG_PATH" >&2
  exit 1
fi

if [[ ! -f "$LUA_CONFIG" ]]; then
  echo "[e] Lua config not found: $LUA_CONFIG" >&2
  exit 1
fi

IMG="slamx-cartographer-melodic:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$(pwd)"
CONFIG_DIR="$(dirname "$LUA_CONFIG")"
CONFIG_BASENAME="$(basename "$LUA_CONFIG")"

echo "[i] Building image $IMG"
docker build -t "$IMG" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

echo "[i] Running Cartographer offline 2D"
echo "[i] Bag: $BAG_PATH"
echo "[i] Scan topic: $SCAN_TOPIC"
echo "[i] Lua: $LUA_CONFIG"
echo "[i] Will record TF into: $OUT_RECORD"

docker run --rm -i \
  --net=host \
  -v "$WORKDIR":"$WORKDIR" \
  -w "$WORKDIR" \
  -e "BAG_PATH=$BAG_PATH" \
  -e "SCAN_TOPIC=$SCAN_TOPIC" \
  -e "CONFIG_DIR=$CONFIG_DIR" \
  -e "CONFIG_BASENAME=$CONFIG_BASENAME" \
  -e "OUT_RECORD=$OUT_RECORD" \
  "$IMG" bash -lc '
    set -euo pipefail
    source /opt/ros/melodic/setup.bash
    roscore >/tmp/roscore.log 2>&1 &
    sleep 2
    rosparam set /use_sim_time true
    rosbag record -O "$OUT_RECORD" /tf /tf_static __name:=slamx_tf_record >/tmp/record.log 2>&1 &
    RECORD_PID=$!
    sleep 1
    rosrun cartographer_ros cartographer_offline_node \
      -configuration_directory "$CONFIG_DIR" \
      -configuration_basenames "$CONFIG_BASENAME" \
      -bag_filenames "$BAG_PATH" \
      scan:="$SCAN_TOPIC" \
      >/tmp/carto.log 2>&1 || true
    kill -INT "$RECORD_PID" 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true
    if [[ -f "${OUT_RECORD}.active" && ! -f "${OUT_RECORD}" ]]; then
      rosbag reindex "${OUT_RECORD}.active" 2>/dev/null || true
      [[ -f "${OUT_RECORD}.active" ]] && cp -f "${OUT_RECORD}.active" "${OUT_RECORD}"
    fi
    rm -f "${OUT_RECORD}.orig.active" 2>/dev/null || true
    echo "[i] Done. Logs: /tmp/carto.log /tmp/record.log"
  '
