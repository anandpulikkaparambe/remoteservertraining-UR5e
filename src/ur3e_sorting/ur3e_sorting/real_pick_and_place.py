#!/usr/bin/env python3
"""
Real-hardware Pick-and-Place for the physical UR5e + Robotiq 2F-140 + Wrist Camera rig.

Mirrors phase1_tf_pick_and_place.py's structure (retry execution, ground-truth
verification after every move, wraparound-safe joint targeting, step-by-step logging)
but targets real hardware instead of Gazebo sim -- the two scripts are NOT interchangeable:

- No MoveIt/OMPL: real hardware here only runs ur_robot_driver's ros2_control stack, not
  move_group. Every move is a raw single-point FollowJointTrajectory goal sent directly to
  scaled_joint_trajectory_controller -- there is no collision-aware path planning at all.
- No TF/Gazebo pose lookup: pick/bin/home are fixed joint-space waypoints taught by hand
  via Freedrive (real_hardware_waypoints.yaml), not detected live -- no perception pipeline
  is wired to real hardware yet, so this only ever picks from that one taught spot.
- Gripper is the real Robotiq Modbus driver (robotiq_2f_gripper_msgs/action/
  MoveTwoFingerGripper over /robotiq_2f_gripper_action), NOT phase1's sim
  control_msgs/action/GripperCommand -- and the position convention is the OPPOSITE of
  phase1's sim gripper: 0.0 = fully CLOSED, 0.14 = fully OPEN (finger-to-finger gap in
  metres for the 140mm gripper), vs. phase1's 0.0 = open, 0.6 = closed curl fraction.
- No trajectory caching: nothing here is planned/randomized (unlike OMPL), so every call
  to the same waypoint already produces the exact same single-point trajectory -- there is
  no varying path that caching would save time on.

2026-09-01: first working version, built directly from a manually-run, verified sequence
(home -> pick -> grasp -> lift -> bin -> release -> home) that completed with zero
protective stops. IMPORTANT -- do not add or change a waypoint by guessing/computing it:
the original sim-derived home_joints pose was found to self-collide with the real
gripper/camera partway through this exact kind of large reconfiguration (two real UR
protective stops, error C153A4, confirmed by direct observation -- see
project_real_hardware_setup memory). Every waypoint in real_hardware_waypoints.yaml was
taught via Freedrive and manually verified clear by a human watching the physical arm;
any new one must be taught the same way.

The lift step is a single-joint (shoulder_lift_joint) nudge on top of the pick waypoint,
not a real Cartesian lift -- this rig has no calibrated real IK set up yet (see
project_real_hardware_setup memory's uncalibrated-kinematics caveat), so a literal small
vertical move isn't reliably computable from the pick pose alone. Nudging the one joint
that empirically raises the tool in this arm configuration was the pragmatic proven
choice for 2026-09-01's working run, not a principled kinematic solution.
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint
import yaml

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

GRIPPER_OPEN = 0.14
GRIPPER_CLOSED = 0.0

# Matches the manually-verified 2026-09-01 run (~8.6deg shoulder_lift nudge from pick).
LIFT_SHOULDER_DELTA = -0.15


class RealPickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('real_pick_and_place_node')

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, MoveTwoFingerGripper, '/robotiq_2f_gripper_action')

        # Stored raw/unwrapped, matching phase1_tf_pick_and_place.py's joint_state_cb --
        # 5 of 6 joints have a real +/-360deg range, so wrapping into [-pi,pi] here would
        # make create_joint_goal's own nearest-equivalent-angle math compare against the
        # wrong number (see that file's joint_state_cb docstring for the full incident).
        self.latest_joint_state = None
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10)

        # Real ground-truth grasp signal from the gripper's own driver -- unlike the sim
        # script, which has no equivalent and just trusts the gripper action reported
        # success, this can actually confirm an object is physically held before lifting.
        self.object_grasped = None
        self._grasp_sub = self.create_subscription(
            Bool, '/robotiq_2f_gripper/object_grasped', self._grasp_cb, 10)

        self.waypoints = self._load_waypoints()

        self.get_logger().info("==================================================")
        self.get_logger().info("Real-Hardware Pick & Place -- UR5e + Robotiq 2F-140")
        self.get_logger().info("==================================================")

    def _load_waypoints(self):
        share_dir = get_package_share_directory('ur3e_sorting')
        path = f"{share_dir}/config/real_hardware_waypoints.yaml"
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {
            name: [pose[j] for j in JOINT_NAMES]
            for name, pose in data.items()
            if isinstance(pose, dict)
        }

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def _grasp_cb(self, msg):
        self.object_grasped = msg.data

    def _current_positions(self):
        if not self.latest_joint_state:
            return None
        return dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))

    def create_joint_goal(self, target_positions, duration_s):
        """Builds a single-point FollowJointTrajectory goal, shifting each target by the
        nearest multiple of 2*pi to the joint's CURRENT position -- same wraparound-safety
        logic as phase1_tf_pick_and_place.py's create_joint_goal, for the same reason:
        without it, a joint with +/-360deg range could be commanded the long way around
        depending on which winding it currently happens to sit on."""
        current = self._current_positions()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        positions = []
        for name, target in zip(JOINT_NAMES, target_positions):
            if current and name in current:
                cur = current[name]
                target = cur + (((target - cur) + math.pi) % (2 * math.pi) - math.pi)
            positions.append(target)
        point.positions = positions
        sec = int(duration_s)
        nsec = int(round((duration_s - sec) * 1e9))
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]
        return goal

    def verify_joint_target(self, target_positions, tol=0.03):
        """Ground-truth check: does the arm's actual joint state match the target, using
        the same shortest-angle modular delta as phase1_tf_pick_and_place.py's
        verify_joint_target -- a raw subtraction would false-fail a joint that landed on a
        different but physically-identical 2*pi branch than the literal target value."""
        current = self._current_positions()
        if current is None:
            self.get_logger().warn("    Verify: no joint state available yet.")
            return False
        ok = True
        for name, target in zip(JOINT_NAMES, target_positions):
            cur = current.get(name)
            if cur is None:
                ok = False
                continue
            delta = abs(((cur - target) + math.pi) % (2 * math.pi) - math.pi)
            if delta > tol:
                self.get_logger().warn(
                    f"    Verify FAILED: {name} actual={cur:.4f} target={target:.4f} delta={delta:.4f} (tol={tol})")
                ok = False
        return ok

    def execute_joint_move(self, target_positions, duration_s, name, attempts=2):
        """Sends a single-point FollowJointTrajectory goal and waits for the result,
        retrying on failure/verify-mismatch -- mirrors phase1's execute_move_with_retry,
        minus trajectory caching (see module docstring for why that doesn't apply here)."""
        for attempt in range(1, attempts + 1):
            if not self._traj_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"{name}: trajectory action server unavailable!")
                return False
            goal = self.create_joint_goal(target_positions, duration_s)
            future = self._traj_client.send_goal_async(goal)
            while not future.done():
                time.sleep(0.1)
            handle = future.result()
            if not handle or not handle.accepted:
                self.get_logger().error(f"{name}: goal rejected (attempt {attempt}/{attempts}).")
                continue
            res_future = handle.get_result_async()
            while not res_future.done():
                time.sleep(0.1)
            result = res_future.result().result
            if result.error_code != 0:
                self.get_logger().error(
                    f"{name}: FollowJointTrajectory failed: {result.error_string} (attempt {attempt}/{attempts}).")
                continue
            time.sleep(0.3)  # let /joint_states catch up before verifying
            if self.verify_joint_target(target_positions):
                self.get_logger().info(f"{name}: reached and verified.")
                return True
            self.get_logger().warn(
                f"{name}: reported success but failed ground-truth verification (attempt {attempt}/{attempts}).")
        return False

    def operate_gripper(self, position, speed=0.5, force=0.5):
        """Actuates the real Robotiq 2F-140 via its Modbus driver's action server. NOTE
        the convention here is the OPPOSITE of phase1_tf_pick_and_place.py's sim
        GripperCommand: position is the finger-to-finger gap in METRES (0.0=closed,
        0.14=fully open for the 140mm gripper), not a 0-0.7 curl fraction where 0=open."""
        self.get_logger().info(f"Gripper -> target_position={position:.3f}m")
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server unavailable.")
            return False
        goal = MoveTwoFingerGripper.Goal()
        goal.target_position = position
        goal.target_speed = speed
        goal.target_force = force
        future = self._gripper_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error("Gripper goal rejected.")
            return False
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)
        result = res_future.result().result
        return bool(result.success)

    def execute_pick_and_place(self):
        self.get_logger().info("\n--- Real-Hardware Pick & Place ---")

        try:
            home = self.waypoints['real_home_position']
            pregrasp = self.waypoints['real_pregrasp_position']
            pick = self.waypoints['real_pick_position']
            bin_pos = self.waypoints['real_bin_position']
        except KeyError as e:
            self.get_logger().error(f"Missing waypoint {e} in real_hardware_waypoints.yaml -- aborting.")
            return False

        lift = list(pick)
        lift[JOINT_NAMES.index('shoulder_lift_joint')] += LIFT_SHOULDER_DELTA

        self.get_logger().info("0. Returning to Home")
        if self.verify_joint_target(home, tol=0.03):
            self.get_logger().info("    Already at Home -- skipping move.")
        elif not self.execute_joint_move(home, duration_s=40.0, name="home"):
            self.get_logger().error("Failed to reach Home.")
            return False

        self.get_logger().info("1. Opening Gripper")
        self.operate_gripper(GRIPPER_OPEN)

        self.get_logger().info("2. Moving to Pregrasp")
        if not self.execute_joint_move(pregrasp, duration_s=25.0, name="pregrasp"):
            self.get_logger().error("Failed Pregrasp move.")
            return False

        self.get_logger().info("2b. Moving to Pick")
        if not self.execute_joint_move(pick, duration_s=15.0, name="pick"):
            self.get_logger().error("Failed Pick move.")
            return False

        self.get_logger().info("3. Grasping")
        self.operate_gripper(GRIPPER_CLOSED)
        time.sleep(1.0)
        if self.object_grasped is not True:
            self.get_logger().error(
                "Grasp not confirmed (object_grasped != true) -- releasing and aborting before lift.")
            self.operate_gripper(GRIPPER_OPEN)
            return False

        self.get_logger().info("4. Lifting")
        if not self.execute_joint_move(lift, duration_s=15.0, name="lift"):
            self.get_logger().error("Failed Lift move. Releasing to avoid an unplanned hold.")
            self.operate_gripper(GRIPPER_OPEN)
            return False

        self.get_logger().info("5. Moving to Bin")
        if not self.execute_joint_move(bin_pos, duration_s=35.0, name="bin"):
            self.get_logger().error("Failed Bin move. Releasing to avoid an unplanned hold.")
            self.operate_gripper(GRIPPER_OPEN)
            return False

        self.get_logger().info("6. Releasing")
        self.operate_gripper(GRIPPER_OPEN)
        time.sleep(1.0)

        self.get_logger().info("7. Returning to Home")
        self.execute_joint_move(home, duration_s=35.0, name="home_return")

        self.get_logger().info("Real-hardware pick and place completed successfully!\n")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = RealPickAndPlaceNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(1.0)
    try:
        node.execute_pick_and_place()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
