#!/bin/bash
# Memory watchdog for unattended training (2026-09-04). This project's Gazebo world has
# twice shown unbounded-looking memory growth this session (10-11GB within 20-83 minutes
# under some conditions) -- this script exists to protect a long unattended run against
# that, without needing anyone present to notice and react.
#
# Runs INSIDE the ur3e_train container. Polls free memory every 5 minutes. If available
# memory drops below CRITICAL_FREE_GB, it stops training+Gazebo cleanly (to reclaim memory
# immediately) and attempts exactly ONE automatic relaunch + resume from the latest
# checkpoint. If that relaunch itself fails to come up within its own timeout, the script
# stops trying (deliberately -- an unattended retry storm that never succeeds is worse than
# stopping once, cleanly, with the last checkpoint intact) and just keeps logging/monitoring
# in case a human comes back. Every action is timestamped in the log file so it's clear on
# return what happened and when.

set -u
LOG="/ros2_ws/rl_logs/memory_watchdog.log"
CRITICAL_FREE_GB=2
CHECK_INTERVAL_S=300      # 5 minutes
MAX_RUNTIME_S=$((8 * 3600))  # 8h safety margin over the planned 7h absence
RELAUNCH_WAIT_S=90

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

free_gb() {
  free -g | awk '/^Mem:/{print $7}'  # "available" column, in GB
}

latest_checkpoint() {
  ls -t /ros2_ws/rl_logs/ur3e_sac_model_*_steps.zip 2>/dev/null | head -1
}

attempt_recovery() {
  log "CRITICAL: available memory below ${CRITICAL_FREE_GB}GB -- stopping training and Gazebo to reclaim memory."
  pkill -9 -f 'ros2 run ur3e_rl train_sac' 2>/dev/null
  pkill -9 -f 'ros2 launch ur_gazebo' 2>/dev/null
  pkill -9 -f 'ign gazebo' 2>/dev/null
  pkill -9 -f move_group 2>/dev/null
  pkill -9 -f parameter_bridge 2>/dev/null
  pkill -9 -f robot_state_publisher 2>/dev/null
  sleep 5
  log "Stopped. Free memory now: $(free_gb)GB available."

  local ckpt
  ckpt=$(latest_checkpoint)
  if [ -z "$ckpt" ]; then
    log "No checkpoint found to resume from -- leaving everything stopped for manual recovery."
    return 1
  fi
  log "Attempting ONE automatic relaunch, will resume from: $ckpt"

  # ROS_LOCALHOST_ONLY + GZ_IP (2026-09-04): required so this relaunch survives network
  # loss the same way a manual restart now does -- without these, ROS2's own DDS discovery
  # and Gazebo's transport both fall back to depending on an active network interface,
  # which is exactly the failure mode this watchdog exists to protect against while
  # unattended. Omitting them here would silently reintroduce that bug during recovery.
  export ROS_LOCALHOST_ONLY=1
  export GZ_IP=127.0.0.1
  source /opt/ros/humble/setup.bash
  source /ros2_ws/install/setup.bash
  ros2 launch ur_gazebo ur.gazebo.launch.py headless:=true launch_rviz:=false \
    > /ros2_ws/rl_logs/watchdog_gazebo_relaunch.log 2>&1 &

  local waited=0
  while [ "$waited" -lt "$RELAUNCH_WAIT_S" ]; do
    sleep 8
    waited=$((waited + 8))
    n=$(ros2 control list_controllers 2>/dev/null | grep -o '\bactive\b' | wc -l)
    if [ "$n" -ge 3 ]; then
      log "Gazebo relaunch succeeded (controllers active after ${waited}s). Resuming training."
      nohup ros2 run ur3e_rl train_sac --resume "$ckpt" \
        > /ros2_ws/rl_logs/watchdog_train_resume.log 2>&1 &
      log "Training resumed (PID $!). Continuing to monitor memory."
      return 0
    fi
  done
  log "Relaunch did NOT come up within ${RELAUNCH_WAIT_S}s -- giving up on auto-recovery. \
Everything left stopped, last checkpoint ($ckpt) is safe. Needs manual restart on return."
  return 1
}

log "=== Memory watchdog started. Checking every ${CHECK_INTERVAL_S}s, critical threshold ${CRITICAL_FREE_GB}GB, max runtime $((MAX_RUNTIME_S / 3600))h. ==="
start_ts=$(date +%s)
recovered_once=0

while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if [ "$elapsed" -ge "$MAX_RUNTIME_S" ]; then
    log "Max runtime reached -- watchdog exiting (does not stop training, just stops monitoring)."
    break
  fi

  avail=$(free_gb)
  log "Check: ${avail}GB available."
  if [ "$avail" -lt "$CRITICAL_FREE_GB" ]; then
    if [ "$recovered_once" -eq 1 ]; then
      log "Already attempted one auto-recovery this run -- not retrying again automatically. \
Stopping training/Gazebo again to be safe, then just monitoring."
      pkill -9 -f 'ros2 run ur3e_rl train_sac' 2>/dev/null
      pkill -9 -f 'ros2 launch ur_gazebo' 2>/dev/null
      pkill -9 -f 'ign gazebo' 2>/dev/null
    else
      attempt_recovery
      recovered_once=1
    fi
  fi
  sleep "$CHECK_INTERVAL_S"
done
