#!/bin/bash
# Headless launcher for ONE Vast.ai (or any headless Docker host) instance, running
# NUM_ENVS Gazebo+MoveIt sub-environments CLUBBED into a single SAC/PPO policy via
# SB3's SubprocVecEnv (train_sac.py/train_sb3.py's existing --num-envs support) --
# NOT NUM_ENVS separate/independent training runs. If you want N unrelated runs
# instead (e.g. to compare seeds), run this script N times with different NUM_ENVS=1
# and CHECKPOINT_DIR values instead.
#
# NOT YET LIVE-VALIDATED on an actual Vast.ai instance -- built from reading this
# project's own launch files (ur.gazebo.launch.py's headless/namespace/gz_partition
# args, train_sac.py/train_sb3.py's --num-envs, all pre-existing) and its Docker
# setup, not from a real end-to-end run. Treat the first run as a real test, not a
# known-working script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configurable via environment, all have defaults ---
ALGO="${ALGO:-sac}"                          # sac | ppo
NUM_ENVS="${NUM_ENVS:-4}"                    # how many Gazebo instances feed the ONE policy
TARGET_SOURCE="${TARGET_SOURCE:-ground_truth}"  # ground_truth | perception
TARGET_NOISE_STD_M="${TARGET_NOISE_STD_M:-0.0}"
RUN_YOLO="${RUN_YOLO:-false}"                # must be true if TARGET_SOURCE=perception
RESUME="${RESUME:-}"                         # path (inside the container, under /ros2_ws/rl_logs) to a .zip checkpoint, optional
# Sized for the WHOLE club (all NUM_ENVS Gazebo instances + the one training process),
# not per-instance -- see this fork's originating conversation for the per-instance
# cost this is built from (~3 CPU cores / ~1.2GB RAM each, headless). Defaults below
# assume NUM_ENVS=4 on a ~20-vCPU host; adjust both together if you change NUM_ENVS.
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-10g}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/rl_logs}"

if [ "$TARGET_SOURCE" = "perception" ] && [ "$RUN_YOLO" != "true" ]; then
    echo "TARGET_SOURCE=perception requires RUN_YOLO=true (yolo_detector.py must be" \
         "running to publish /detected_object_pose) -- refusing to start." >&2
    exit 1
fi

echo "Building the Docker image..."
docker build -t ur3e_vastai "$SCRIPT_DIR"

mkdir -p "$CHECKPOINT_DIR"

echo "Starting ONE clubbed run: algo=$ALGO, num_envs=$NUM_ENVS, target_source=$TARGET_SOURCE," \
     "cpus=$CPUS, memory=$MEMORY..."

docker run -d \
    --name "ur3e_vastai_club" \
    --cpus="$CPUS" \
    --memory="$MEMORY" \
    --env="UR3E_TARGET_SOURCE=${TARGET_SOURCE}" \
    --env="UR3E_TARGET_NOISE_STD_M=${TARGET_NOISE_STD_M}" \
    --volume="$CHECKPOINT_DIR:/ros2_ws/rl_logs" \
    ur3e_vastai \
    /ros2_ws/vastai_train_entrypoint.sh "$ALGO" "$NUM_ENVS" "$RUN_YOLO" "$RESUME"

echo "Started. Follow logs with: docker logs -f ur3e_vastai_club"
