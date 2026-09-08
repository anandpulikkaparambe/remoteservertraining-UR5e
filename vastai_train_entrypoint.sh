#!/bin/bash
# Runs inside the container (see run_vastai.sh). Launches headless Gazebo+MoveIt for
# one instance, then starts training against it. NOT YET LIVE-VALIDATED -- the fixed
# 60s wait below is a placeholder, not a verified readiness check; if move_group isn't
# actually up by then, train_sac/train_ppo's first reset() will fail loudly rather than
# silently, so this should fail fast and visibly rather than hang.
set -e

ALGO="${1:-sac}"
NAMESPACE="${2:-env0}"
GZ_PARTITION="${3:-env0}"
RUN_YOLO="${4:-false}"

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

echo "Launching headless Gazebo+MoveIt (namespace=$NAMESPACE, gz_partition=$GZ_PARTITION, run_yolo=$RUN_YOLO)..."
ros2 launch ur_gazebo ur.gazebo.launch.py \
    namespace:="$NAMESPACE" \
    gz_partition:="$GZ_PARTITION" \
    headless:=true \
    launch_rviz:=false \
    run_yolo:="$RUN_YOLO" \
    > "/ros2_ws/rl_logs/gazebo_${NAMESPACE}.log" 2>&1 &

echo "Waiting 60s for the stack to come up (placeholder -- not a verified readiness poll)..."
sleep 60

echo "Starting training: $ALGO"
exec ros2 run ur3e_rl "train_${ALGO}" \
    > "/ros2_ws/rl_logs/train_${ALGO}_${NAMESPACE}.log" 2>&1
