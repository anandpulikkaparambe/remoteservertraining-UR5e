#!/usr/bin/env python3
"""
Phase 2 RL Policy Dry-Run on Real Hardware -- observes real robot state and logs what
the trained SAC policy WOULD command, WITHOUT ever sending anything to a real actuator.

Built 2026-09-01 specifically to sanity-check the Phase 2 policy before considering any
live deployment on the real UR5e. See project_phase2_rl_status / project_real_hardware_setup
memory for the full risk rationale: this policy has NEVER achieved a full task success,
even in Gazebo sim, outputs joint-position deltas up to 0.8 rad/s per joint (far faster
than the ~1-2 deg/s used for every real move validated this session), and its
self-collision safety net was tuned against the Gazebo model's geometry -- not the real
gripper/camera bulk that triggered two genuine UR protective stops earlier this same
session on a much simpler, deterministic script.

Approximations made to build a real observation matching ur3e_env.py's 24-dim format
(6 joint pos + 6 joint vel + 7 EE pose [xyz,quat] + 3 goal pos + 1 gripper pos +
1 phase indicator), since real hardware has no equivalent for several of these:

- EE pose: ur3e_env.py uses TF base_link -> right_inner_finger_pad, which does not
  exist on the real robot's current TF tree (launched via the stock ur_robot_driver,
  no gripper in the URDF -- see project_real_hardware_setup memory). Substituted with
  base_link -> tool0, a KNOWN, BOUNDED ~0.17m offset from the trained reference point
  along the gripper's axis -- not a guess, just a different fixed point.
- Goal position (lego during 'grasp' / bin during 'release'): ur3e_env.py sources this
  live from Gazebo ground truth, which doesn't exist here. Substituted with FIXED real
  Cartesian positions sampled live via TF while the arm was physically at the already
  Freedrive-verified real_pick_position / real_bin_position waypoints (2026-09-01, same
  session, same real_hardware_waypoints.yaml as real_pick_and_place.py uses) -- real
  measured numbers, not computed/guessed ones, and NOT the same as the Gazebo sim's
  BIN_POSE_BASE_LINK constant, which reflects a different table/bin layout entirely.
- Gripper position: the real driver publishes the SAME 'finger_joint' name in the SAME
  0 (open) - 0.7 (closed) radian convention as the sim, just on a different topic
  (/robotiq_2f_gripper/joint_states instead of /joint_states) -- no conversion needed.
- Sensor noise: sim's domain-randomization Gaussian noise on position/velocity is
  omitted -- real sensor noise is real; adding synthetic noise on top would only
  distort this comparison.

NEVER publishes to any trajectory controller topic and NEVER calls the gripper action --
every predicted action is only logged for a human to inspect.
"""

import argparse
import math
import threading
import time

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState
from stable_baselines3 import SAC

UR3E_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Real Cartesian positions (base_link frame), sampled live via TF on 2026-09-01 while the
# arm was physically at the Freedrive-taught real_pick_position/real_bin_position
# waypoints (src/ur3e_sorting/config/real_hardware_waypoints.yaml).
REAL_PICK_XYZ = np.array([0.802, -0.260, 0.237], dtype=np.float32)
REAL_BIN_XYZ = np.array([0.415, 0.608, 0.294], dtype=np.float32)

MAX_JOINT_DELTA_RAD = 0.08  # matches ur3e_env.py's action_space bound
SIM_STEP_TIME = 0.1  # matches ur3e_env.py's sim_step_time, used only to report an
                      # implied deg/s rate -- nothing is actually executed at this rate.

# --legacy: evaluate a pre-two-phase-extension checkpoint (e.g. ur3e_sac_model_281088_steps.zip,
# 23-dim observation, no phase indicator, grasp-only -- see project_phase2_rl_status memory's
# 2026-09-01 Gazebo evaluation of this same checkpoint). Its pickled spaces also hit the same
# numpy-pickle incompatibility documented for the 490230-step checkpoint -- custom_objects
# bypasses that; the network weights themselves load fine.
from gymnasium import spaces
LEGACY_OBSERVATION_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
LEGACY_ACTION_SPACE = spaces.Box(
    low=np.array([-MAX_JOINT_DELTA_RAD] * 6 + [-math.pi], dtype=np.float32),
    high=np.array([MAX_JOINT_DELTA_RAD] * 6 + [math.pi], dtype=np.float32),
    dtype=np.float32,
)


class Phase2DryRunNode(Node):
    def __init__(self, checkpoint_path, phase, legacy=False):
        super().__init__('phase2_dry_run_node')

        self.phase = phase  # 'grasp' or 'release' -- fixed for the whole run, not
                             # auto-transitioned like the real env's phase logic.
        self.legacy = legacy
        self.current_joint_positions = np.zeros(6, dtype=np.float32)
        self.current_joint_velocities = np.zeros(6, dtype=np.float32)
        self.current_gripper_pos = 0.0

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(JointState, '/robotiq_2f_gripper/joint_states', self._gripper_state_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Loading SAC checkpoint: {checkpoint_path} (legacy={legacy})")
        load_kwargs = {}
        if legacy:
            load_kwargs['custom_objects'] = {
                'observation_space': LEGACY_OBSERVATION_SPACE,
                'action_space': LEGACY_ACTION_SPACE,
            }
        self.model = SAC.load(checkpoint_path, device='cpu', **load_kwargs)
        self.get_logger().info("Checkpoint loaded.")

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
            # 23-dim: no trailing phase-indicator element -- this checkpoint predates
            # the two-phase extension and never saw one during training.
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

    def predict_and_log(self, obs, step_idx):
        action, _ = self.model.predict(obs, deterministic=True)
        arm_delta = np.clip(action[:6], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
        gripper_raw = float(action[6])
        gripper_cmd_val = ((np.clip(gripper_raw, -math.pi, math.pi) + math.pi) / (2 * math.pi)) * 0.8

        max_delta_deg = float(np.max(np.abs(arm_delta))) * 180.0 / math.pi
        implied_rate_deg_s = max_delta_deg / SIM_STEP_TIME

        self.get_logger().info(
            f"[step {step_idx:03d}] arm_delta(rad)={np.round(arm_delta, 4).tolist()} "
            f"max={max_delta_deg:.2f}deg (~{implied_rate_deg_s:.1f} deg/s if this were executed) "
            f"gripper_raw={gripper_raw:.3f} -> gripper_cmd~{gripper_cmd_val:.3f}rad "
            f"(0=open,0.7=closed, real driver's convention)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to a SAC .zip checkpoint')
    parser.add_argument('--phase', default='grasp', choices=['grasp', 'release'])
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--legacy', action='store_true',
                         help='Evaluate a pre-two-phase-extension (23-dim) checkpoint, e.g. ur3e_sac_model_281088_steps.zip')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = Phase2DryRunNode(args.checkpoint, phase=args.phase, legacy=args.legacy)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(1.0)  # let subscriptions/TF fill in before the first observation

    node.get_logger().info("=== DRY RUN: no commands will be sent to the robot or gripper ===")
    try:
        for i in range(args.steps):
            obs = node.build_observation()
            if obs is None:
                node.get_logger().warn("Skipping step -- no valid observation yet.")
                time.sleep(0.5)
                continue
            node.predict_and_log(obs, i)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("=== DRY RUN COMPLETE -- nothing was ever sent to the robot ===")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
