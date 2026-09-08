#!/usr/bin/env python3
"""
Evaluates an old, pre-two-phase-extension SAC checkpoint (23-dim observation, no
phase indicator, no bin/release stage) -- e.g. ur3e_sac_model_281088_steps.zip, the
last checkpoint from the "reach-and-grasp only" era described in project_phase2_rl_status
memory, before the undocumented two-phase pick-and-place extension was added and every
checkpoint since has recorded zero full task successes.

This is NOT a thin wrapper around UR3eEnv.step()/reset() via the normal Gym interface:
the CURRENT step() has a classical MoveIt handoff-to-bin call baked directly into its
grasp-detection branch (the moment dist<0.05 and the gripper is closed, it calls
self.handoff_bridge.transport_to_bin() and sets self.phase='release') -- logic that did
not exist when this checkpoint was trained. Reusing that path here would silently test
something this checkpoint never learned, not reproduce its actual original behavior.

Instead: UR3eEnv.reset() is called normally (verified clean of any phase/transport
logic -- just home + open gripper + respawn lego), then this script drives its own
minimal step loop against the SAME underlying, bug-fixed infrastructure (joint
position/velocity tracking, joint-limit clamping, gz ground-truth queries, gz partition/
IP fixes) but with:
- A 23-dim observation (UR3eEnv's own 24-dim _get_obs() with the trailing phase
  indicator element dropped) -- matches this checkpoint's actual trained input shape.
- A simple, original-era success criterion: end-effector within 5cm of the target AND
  gripper closed -- no bin transport, no release phase, matching the project's own
  documented original task definition before the extension.
- Direct joint-trajectory/gripper publishing (same calls UR3eEnv.step() itself makes
  internally), not the reward/termination bookkeeping the current step() wraps them in.
"""

import argparse
import time

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from gymnasium import spaces
from stable_baselines3 import SAC
from ur3e_rl.ur3e_env import UR3eEnv

# This checkpoint's pickled observation_space/action_space were saved under a numpy
# version whose generator pickle format this environment's numpy can't deserialize
# (confirmed live: "__generator_ctor() takes from 0 to 1 positional arguments but 2
# were given" -- the same numpy-pickle incompatibility already documented for the
# 490230-step checkpoint resume in project_phase2_rl_status memory). Passing these
# explicitly bypasses that broken deserialization entirely -- the network weights
# themselves load fine, only the pickled space *metadata* is the problem.
LEGACY_OBSERVATION_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
LEGACY_ACTION_SPACE = spaces.Box(
    low=np.array([-0.08] * 6 + [-np.pi], dtype=np.float32),
    high=np.array([0.08] * 6 + [np.pi], dtype=np.float32),
    dtype=np.float32,
)


def build_obs_23(env):
    """UR3eEnv._get_obs() minus the trailing phase-indicator element."""
    obs24 = env._get_obs()
    return obs24[:23]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=150)
    args, _ = parser.parse_known_args()

    if not rclpy.ok():
        rclpy.init()

    print(f"Creating UR3eEnv (reusing current, bug-fixed infrastructure)...")
    env = UR3eEnv(env_id=0)

    print(f"Loading legacy checkpoint: {args.checkpoint}")
    model = SAC.load(
        args.checkpoint, device='cpu',
        custom_objects={
            'observation_space': LEGACY_OBSERVATION_SPACE,
            'action_space': LEGACY_ACTION_SPACE,
        },
    )
    obs_dim = model.observation_space.shape[0]
    if obs_dim != 23:
        print(f"[WARN] Loaded checkpoint's observation_space is {obs_dim}-dim, not the "
              f"expected 23 -- this script assumes the pre-extension shape. Proceeding "
              f"anyway, but results may be meaningless if this isn't actually a legacy checkpoint.")

    successes = 0
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        env.reset()  # safe: home + open gripper + respawn lego, no phase/transport logic
        time.sleep(0.5)

        success = False
        for step_idx in range(args.max_steps):
            obs = build_obs_23(env)
            action, _ = model.predict(obs, deterministic=True)

            arm_delta = np.clip(action[:6], -env.max_joint_delta_rad, env.max_joint_delta_rad)
            arm_target = np.clip(
                env.current_joint_positions + arm_delta, env.joint_pos_min, env.joint_pos_max
            )

            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            traj_msg = JointTrajectory()
            traj_msg.joint_names = env.ur3e_joint_names
            point = JointTrajectoryPoint()
            point.positions = arm_target.astype(float).tolist()
            point.time_from_start.sec = int(env.sim_step_time)
            point.time_from_start.nanosec = int((env.sim_step_time - int(env.sim_step_time)) * 1e9)
            traj_msg.points = [point]
            env.trajectory_pub.publish(traj_msg)

            gripper_val = action[6]
            gripper_cmd_val = float(((np.clip(gripper_val, -np.pi, np.pi) + np.pi) / (2 * np.pi)) * 0.8)
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = gripper_cmd_val
            goal_msg.command.max_effort = 50.0
            if env.gripper_client.wait_for_server(timeout_sec=0.1):
                env.gripper_client.send_goal_async(goal_msg)

            time.sleep(env.sim_step_time)

            env._update_ee_pose()
            env._update_target_pose()
            ee_pos = env.current_ee_pose[:3]
            dist = float(np.linalg.norm(ee_pos - env.target_pose))
            gripper_pos = getattr(env, 'current_gripper_pos', 0.0)
            gripper_closed = gripper_pos > 0.5

            if step_idx % 10 == 0 or (dist < 0.05):
                print(f"  step {step_idx:03d} dist={dist:.4f}m gripper_closed_amount={gripper_pos:.3f}")

            if dist < 0.05 and gripper_closed:
                gz_pos = env._get_gz_object_position('lego_red')
                if gz_pos is not None and gz_pos[2] > 0.86:
                    success = True
                    print(f"  SUCCESS at step {step_idx}: dist={dist:.4f}m, lego Z={gz_pos[2]:.4f}m (lifted)")
                    break

        if success:
            successes += 1
            print(f"Episode {ep + 1}: SUCCESS")
        else:
            print(f"Episode {ep + 1}: FAILED (no successful grasp within {args.max_steps} steps)")

    print(f"\n=== Result: {successes}/{args.episodes} successful grasps ===")
    env.node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
