#!/usr/bin/env python3
"""
Phase 2 RL Policy CLOSED-LOOP Dry-Run on Real Hardware.

Unlike real_phase2_dry_run.py (pure observation, zero motion, single repeated
data point), this version actually steps the real arm forward using the policy's
OWN output -- but never the raw output. Every arm delta is clamped to a fraction of
what the policy was trained to command, and every step is verified against real
safety-mode telemetry before the next one is attempted. This exists to see how the
policy's behavior evolves as its observation actually changes (impossible to observe
in the pure dry-run), while keeping the real risk of an untested, never-successful
policy bounded.

Built 2026-09-01 as the deliberate follow-up to real_phase2_dry_run.py, after that
script's single-step result showed the raw policy wanting to move ~5 of 6 joints
simultaneously at ~94-100% of its max trained delta (0.08 rad/step, ~46 deg/s if
executed raw) as its very first action from the current resting pose -- see
project_real_hardware_setup memory. That result is exactly why this script clamps
MUCH harder than the trained bound (see CLAMP_FRACTION below) rather than executing
the policy's real output.

Safety design, all deliberate and NOT to be loosened without a fresh justification:
- CLAMP_FRACTION shrinks every per-step delta to a small fraction of what the policy
  actually asked for, not the trained max -- the policy's raw ask is still logged in
  full for comparison, but only the shrunk value is ever sent.
- Each clamped delta is sent as its own single-point FollowJointTrajectory goal with
  a generous fixed duration (STEP_DURATION_S), not the sim's raw 0.1s cadence -- this
  keeps real executed speed far below the trained regime regardless of clamp math.
- The gripper is NEVER actuated by this script, in either phase -- the policy's
  gripper output is only logged.
- safety_mode is checked after every single step; the loop stops immediately (not at
  the next iteration) on anything other than NORMAL.
- A hard MAX_STEPS bound and a hard MAX_TOTAL_DELTA_RAD per joint (cumulative across
  the whole run) are both enforced independently of the step-by-step clamp, so a
  policy that keeps pushing the same direction every step still can't accumulate a
  large real move over many steps.
"""

import argparse
import math
import threading
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from ur_dashboard_msgs.msg import SafetyMode
from trajectory_msgs.msg import JointTrajectoryPoint
from stable_baselines3 import SAC

UR3E_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Real Cartesian positions (base_link frame), sampled live via TF 2026-09-01 at the
# Freedrive-taught real_pick_position/real_bin_position waypoints -- see
# real_phase2_dry_run.py's docstring for the same values and full rationale.
REAL_PICK_XYZ = np.array([0.802, -0.260, 0.237], dtype=np.float32)
REAL_BIN_XYZ = np.array([0.415, 0.608, 0.294], dtype=np.float32)

MAX_JOINT_DELTA_RAD = 0.08  # the policy's own trained action-space bound, for reference/logging only

# --- The actual safety knobs for this script ---
CLAMP_FRACTION = 0.125          # only 1/8 of the policy's own (already-bounded) request is ever sent
STEP_DURATION_S = 1.0           # generous fixed duration per tiny move -- keeps real speed far below training regime
MAX_STEPS = 10                  # hard cap on how many real moves this script will ever make in one run
MAX_TOTAL_DELTA_RAD = 0.15      # hard cap on cumulative per-joint motion across the whole run (~8.6deg)

# --legacy: pre-two-phase-extension (23-dim) checkpoint -- see real_phase2_dry_run.py's
# LEGACY_OBSERVATION_SPACE/LEGACY_ACTION_SPACE docstring for the full rationale (same
# numpy-pickle incompatibility, same custom_objects fix).
from gymnasium import spaces
LEGACY_OBSERVATION_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
LEGACY_ACTION_SPACE = spaces.Box(
    low=np.array([-MAX_JOINT_DELTA_RAD] * 6 + [-math.pi], dtype=np.float32),
    high=np.array([MAX_JOINT_DELTA_RAD] * 6 + [math.pi], dtype=np.float32),
    dtype=np.float32,
)


class Phase2ClosedLoopDryRunNode(Node):
    def __init__(self, checkpoint_path, phase, legacy=False):
        super().__init__('phase2_closed_loop_dry_run_node')

        self.phase = phase
        self.legacy = legacy
        self.current_joint_positions = np.zeros(6, dtype=np.float32)
        self.current_joint_velocities = np.zeros(6, dtype=np.float32)
        self.current_gripper_pos = 0.0
        self.safety_mode = None

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(JointState, '/robotiq_2f_gripper/joint_states', self._gripper_state_cb, 10)
        # io_and_status_controller publishes safety_mode as TRANSIENT_LOCAL (latched --
        # only republished on an actual state change), so a VOLATILE (default) subscriber
        # started after that one message never receives it and self.safety_mode stays
        # None forever if the mode hasn't changed since. Match the durability so the
        # already-published NORMAL value is delivered immediately on subscribe.
        safety_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(SafetyMode, '/io_and_status_controller/safety_mode', self._safety_cb, safety_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.get_logger().info(f"Loading SAC checkpoint: {checkpoint_path} (legacy={legacy})")
        load_kwargs = {}
        if legacy:
            load_kwargs['custom_objects'] = {
                'observation_space': LEGACY_OBSERVATION_SPACE,
                'action_space': LEGACY_ACTION_SPACE,
            }
        self.model = SAC.load(checkpoint_path, device='cpu', **load_kwargs)
        self.get_logger().info("Checkpoint loaded.")

        # Cumulative signed delta per joint sent so far this run -- checked against
        # MAX_TOTAL_DELTA_RAD before every step, independent of the per-step clamp.
        self.cumulative_delta = np.zeros(6, dtype=np.float32)

    def _joint_state_cb(self, msg):
        for i, name in enumerate(UR3E_JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.current_joint_velocities[i] = msg.velocity[idx]

    def _gripper_state_cb(self, msg):
        if 'finger_joint' in msg.name:
            self.current_gripper_pos = msg.position[msg.name.index('finger_joint')]

    def _safety_cb(self, msg):
        self.safety_mode = msg.mode

    def build_observation(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', 'tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            ee_pose = np.array([
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w,
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"tool0 TF lookup failed: {e}")
            return None

        goal_pos = REAL_BIN_XYZ if self.phase == 'release' else REAL_PICK_XYZ

        if self.legacy:
            return np.concatenate((
                self.current_joint_positions,
                self.current_joint_velocities,
                ee_pose,
                goal_pos,
                [self.current_gripper_pos],
            ), dtype=np.float32)

        phase_indicator = 1.0 if self.phase == 'release' else 0.0
        return np.concatenate((
            self.current_joint_positions,
            self.current_joint_velocities,
            ee_pose,
            goal_pos,
            [self.current_gripper_pos],
            [phase_indicator],
        ), dtype=np.float32)

    def safety_ok(self):
        # 1 == NORMAL in ur_dashboard_msgs/SafetyMode. None (no message yet) is NOT
        # treated as ok -- fail closed, not open.
        return self.safety_mode == 1

    def send_clamped_step(self, arm_delta_clamped):
        """Sends one single-point FollowJointTrajectory goal for the already-clamped
        delta, waits for the result, and verifies the arm actually moved to within a
        tight tolerance of the intended (clamped, tiny) target."""
        target = self.current_joint_positions + arm_delta_clamped
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server unavailable!")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(UR3E_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = target.astype(float).tolist()
        point.time_from_start = Duration(sec=int(STEP_DURATION_S),
                                          nanosec=int((STEP_DURATION_S % 1) * 1e9))
        goal.trajectory.points = [point]

        future = self._traj_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error("Step goal rejected.")
            return False

        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.05)
        result = res_future.result().result
        if result.error_code != 0:
            self.get_logger().error(f"Step failed: {result.error_string}")
            return False
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--phase', default='grasp', choices=['grasp', 'release'])
    parser.add_argument('--legacy', action='store_true',
                         help='Evaluate a pre-two-phase-extension (23-dim) checkpoint, e.g. ur3e_sac_model_281088_steps.zip')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = Phase2ClosedLoopDryRunNode(args.checkpoint, phase=args.phase, legacy=args.legacy)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait explicitly for the first real safety_mode reading rather than a fixed sleep --
    # the TRANSIENT_LOCAL discovery/replay handshake took ~8-10s in testing, not
    # instantaneous. Fail closed (abort, don't proceed silently) if it never arrives.
    wait_deadline = time.time() + 15.0
    while node.safety_mode is None and time.time() < wait_deadline:
        time.sleep(0.2)
    if node.safety_mode is None:
        node.get_logger().error("Never received a safety_mode reading within 15s -- aborting before any motion.")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        return
    time.sleep(0.5)  # let TF/joint_states finish filling in too

    node.get_logger().info(
        f"=== CLOSED-LOOP DRY RUN: clamp={CLAMP_FRACTION}x policy request, "
        f"{STEP_DURATION_S}s/step, max {MAX_STEPS} steps, "
        f"cumulative cap {MAX_TOTAL_DELTA_RAD}rad/joint, gripper NEVER actuated ==="
    )

    try:
        if not node.safety_ok():
            node.get_logger().error(f"Safety mode not NORMAL at start (mode={node.safety_mode}) -- aborting before any motion.")
            return

        for i in range(MAX_STEPS):
            obs = node.build_observation()
            if obs is None:
                node.get_logger().warn("No valid observation -- stopping.")
                break

            action, _ = node.model.predict(obs, deterministic=True)
            raw_delta = np.clip(action[:6], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
            clamped_delta = raw_delta * CLAMP_FRACTION

            # Hard cumulative cap, independent of the per-step clamp above.
            would_be_cumulative = node.cumulative_delta + clamped_delta
            if np.any(np.abs(would_be_cumulative) > MAX_TOTAL_DELTA_RAD):
                node.get_logger().warn(
                    f"[step {i:02d}] Cumulative cap ({MAX_TOTAL_DELTA_RAD}rad) would be exceeded -- stopping run here.")
                break

            gripper_raw = float(action[6])
            node.get_logger().info(
                f"[step {i:02d}] policy_raw_delta(rad)={np.round(raw_delta, 4).tolist()} "
                f"-> SENDING clamped(rad)={np.round(clamped_delta, 4).tolist()} "
                f"gripper_raw={gripper_raw:.3f} (NOT sent -- gripper never actuated by this script)"
            )

            if not node.send_clamped_step(clamped_delta):
                node.get_logger().error(f"[step {i:02d}] Step execution failed -- stopping run.")
                break

            time.sleep(0.3)  # let /joint_states and safety_mode catch up
            if not node.safety_ok():
                node.get_logger().error(
                    f"[step {i:02d}] Safety mode not NORMAL after step (mode={node.safety_mode}) -- stopping run immediately.")
                break

            node.cumulative_delta = would_be_cumulative
            node.get_logger().info(
                f"[step {i:02d}] OK. Cumulative delta so far(rad)={np.round(node.cumulative_delta, 4).tolist()}")

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("=== CLOSED-LOOP DRY RUN ENDED ===")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
