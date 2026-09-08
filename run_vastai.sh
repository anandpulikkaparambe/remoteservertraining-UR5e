#!/bin/bash
# Headless launcher for a single training instance on a Vast.ai (or any headless
# Docker) host. Adapted from the original run_docker.sh, which assumed a local
# machine with a display (X11/DISPLAY, --net=host for GUI tooling) -- none of that
# exists on a remote GPU rental, so it's dropped here rather than reused as-is.
#
# NOT YET LIVE-VALIDATED on an actual Vast.ai instance -- built from reading this
# project's own launch files (ur.gazebo.launch.py's headless/namespace/gz_partition
# args, all pre-existing) and its Docker setup, not from a real end-to-end run. Treat
# the first run as a real test, not a known-working script -- see this project's own
# history of "looked right on paper" launch/handoff code needing live correction.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configurable via environment, all have defaults ---
ALGO="${ALGO:-sac}"                          # sac | ppo
INSTANCE_ID="${INSTANCE_ID:-0}"              # 0, 1, 2, ... one per parallel container
NAMESPACE="${NAMESPACE:-env${INSTANCE_ID}}"
GZ_PARTITION="${GZ_PARTITION:-${NAMESPACE}}"
TARGET_SOURCE="${TARGET_SOURCE:-ground_truth}"  # ground_truth | perception
TARGET_NOISE_STD_M="${TARGET_NOISE_STD_M:-0.0}"
RUN_YOLO="${RUN_YOLO:-false}"                # must be true if TARGET_SOURCE=perception
CPUS="${CPUS:-4}"                            # docker --cpus cap, see the parallel-instance
MEMORY="${MEMORY:-6g}"                       # sizing discussion this fork was built from
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/rl_logs}"

if [ "$TARGET_SOURCE" = "perception" ] && [ "$RUN_YOLO" != "true" ]; then
    echo "TARGET_SOURCE=perception requires RUN_YOLO=true (yolo_detector.py must be" \
         "running to publish /detected_object_pose) -- refusing to start." >&2
    exit 1
fi

echo "Building the Docker image..."
docker build -t ur3e_vastai "$SCRIPT_DIR"

mkdir -p "$CHECKPOINT_DIR"

echo "Starting instance '$INSTANCE_ID' (namespace=$NAMESPACE, gz_partition=$GZ_PARTITION," \
     "target_source=$TARGET_SOURCE, cpus=$CPUS, memory=$MEMORY)..."

docker run -d \
    --name "ur3e_vastai_${INSTANCE_ID}" \
    --cpus="$CPUS" \
    --memory="$MEMORY" \
    --env="UR3E_TARGET_SOURCE=${TARGET_SOURCE}" \
    --env="UR3E_TARGET_NOISE_STD_M=${TARGET_NOISE_STD_M}" \
    --volume="$CHECKPOINT_DIR:/ros2_ws/rl_logs" \
    ur3e_vastai \
    /ros2_ws/vastai_train_entrypoint.sh "$ALGO" "$NAMESPACE" "$GZ_PARTITION" "$RUN_YOLO"

echo "Started. Follow logs with: docker logs -f ur3e_vastai_${INSTANCE_ID}"
