#!/bin/bash
# Runs inside the container (see run_vastai.sh). Launches NUM_ENVS headless
# Gazebo+MoveIt stacks (namespaces '', env1, env2, ... -- matching train_sac.py's own
# convention: instance 0 is always the blank namespace), waits for them to come up,
# then starts ONE train_sac/train_ppo process with --num-envs=NUM_ENVS so all
# instances feed a single shared policy via SB3's SubprocVecEnv.
#
# NOT YET LIVE-VALIDATED -- the fixed per-instance wait below is a placeholder, not a
# verified readiness check. If move_group isn't actually up by the time training
# starts, the first reset() on that instance will fail loudly, not hang silently.
set -e

ALGO="${1:-sac}"
NUM_ENVS="${2:-4}"
RUN_YOLO="${3:-false}"
RESUME="${4:-}"

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

for i in $(seq 0 $((NUM_ENVS - 1))); do
    if [ "$i" -eq 0 ]; then
        NS=""
    else
        NS="env${i}"
    fi
    echo "Launching Gazebo+MoveIt instance $i (namespace='${NS}', gz_partition='${NS}')..."
    # `ros2 launch` rejects a bare `namespace:=` with nothing after the `=` as a
    # malformed argument -- can't pass NS unconditionally when it's empty (instance 0).
    # Omitting the args entirely for that case is equivalent: ur.gazebo.launch.py
    # already defaults both to "" (see its DeclareLaunchArgument calls).
    NS_ARGS=()
    if [ -n "$NS" ]; then
        NS_ARGS=(namespace:="$NS" gz_partition:="$NS")
    fi
    # `headless:=true` only drops the gzclient GUI window -- `ign gazebo -s` (server-only)
    # still initializes OGRE for the robot's wrist camera sensor (use_camera:=true, always
    # on regardless of run_yolo), which needs a real X/GLX context. On a true headless host
    # (no X server at all, e.g. any Vast.ai instance) that crashes gzserver outright:
    # "Ogre::RenderingAPIException: Couldn't open X display", exit code 134 (SIGABRT) --
    # confirmed live, not hypothetical. xvfb-run -a gives it a virtual display per instance
    # (auto-picks a free display number, so concurrent instances don't collide).
    xvfb-run -a -s "-screen 0 1280x1024x24" \
        ros2 launch ur_gazebo ur.gazebo.launch.py \
        "${NS_ARGS[@]}" \
        headless:=true \
        launch_rviz:=false \
        run_yolo:="$RUN_YOLO" \
        > "/ros2_ws/rl_logs/gazebo_instance${i}.log" 2>&1 &
    # Staggered, not simultaneous: NUM_ENVS Gazebo processes all starting at once on a
    # shared CPU budget is a heavier simultaneous spike than train_sac.py's own docs
    # assume (each instance normally comes up one at a time, by hand, per the
    # multi-instance workflow this project already had). 15s apart is a guess, not
    # measured -- tune if instances are still contending for startup CPU in practice.
    sleep 15
done

echo "All $NUM_ENVS instance(s) launched. Waiting 60s more for the slowest to settle" \
     "(placeholder -- not a verified readiness poll)..."
sleep 60

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume "$RESUME")
fi

echo "Starting clubbed training: $ALGO across $NUM_ENVS instance(s)"
exec ros2 run ur3e_rl "train_${ALGO}" --num-envs "$NUM_ENVS" "${RESUME_ARGS[@]}" \
    > "/ros2_ws/rl_logs/train_${ALGO}_club.log" 2>&1
