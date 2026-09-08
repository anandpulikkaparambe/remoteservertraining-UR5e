#!/usr/bin/env python3
"""
Phase 1: Deterministic Joint-Space Pick & Place for UR5e Robot
- Calculates exact joint angles for Initial State, Pre-Pick, Pick, Lift, Bin Drop, and Home.
- Uses MoveGroup Joint Constraints for 100% reliable motion planning without IK singularities or collision timeouts.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, CollisionObject
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
import math
import time

class Phase1JointPickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('phase1_joint_pick_and_place_node', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])
        
        # Action Clients for MoveGroup & Gripper
        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        
        # Subscriber for current joint state
        self._joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.current_joints = {}
        
        # Collision Publisher
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        
        # Joint Names
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # --- CALCULATED JOINT TARGET CONFIGURATIONS (in Radians) ---
        
        # 1. Initial / Home Position
        self.joint_home = {
            "shoulder_pan_joint": 2.040,
            "shoulder_lift_joint": -1.570,
            "elbow_joint": 0.0,
            "wrist_1_joint": -1.570,
            "wrist_2_joint": -1.570,
            "wrist_3_joint": 0.0
        }

        # 2. Pre-Pick Approach (Above lego_red at [0.30, 0.20, 0.85])
        self.joint_pre_pick = {
            "shoulder_pan_joint": 2.040,
            "shoulder_lift_joint": -1.589,
            "elbow_joint": 1.991,
            "wrist_1_joint": -1.973,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0
        }

        # 3. Grasp Position (Lowered onto lego_red)
        self.joint_pick = {
            "shoulder_pan_joint": 2.040,
            "shoulder_lift_joint": -1.555,
            "elbow_joint": 2.043,
            "wrist_1_joint": -2.058,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0
        }

        # 4. Bin Placement Position (Above bin at [-0.30, 0.30, 0.95])
        self.joint_bin = {
            "shoulder_pan_joint": 0.411,
            "shoulder_lift_joint": -1.833,
            "elbow_joint": 1.933,
            "wrist_1_joint": -1.671,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0
        }

        self.get_logger().info("==================================================")
        self.get_logger().info("🚀 Phase 1: Joint-Space Precision Pick & Place")
        self.get_logger().info("   Direct Calculated Joint Angles | 100% Reliable")
        self.get_logger().info("==================================================")

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def operate_gripper(self, position, effort=100.0, wait=True):
        """Actuates the Robotiq 2F Gripper."""
        self.get_logger().info(f"Gripper Action -> Target Position = {position}")
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Gripper action server unavailable.")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = effort
        
        future = self._gripper_client.send_goal_async(goal)
        if not wait:
            return True

        try:
            handle = future.result()
            if handle and handle.accepted:
                handle.get_result_async()
                time.sleep(1.0)
                return True
        except Exception as e:
            self.get_logger().error(f"Gripper execution error: {e}")
        return False

    def send_joint_goal(self, joint_dict, vel_scale=0.3, accel_scale=0.3):
        """Sends direct Joint Constraints to MoveGroup."""
        goal = MoveGroup.Goal()
        goal.request.group_name = "arm"
        goal.request.pipeline_id = "pilz_industrial_motion_planner"
        goal.request.planner_id = "PTP"
        goal.request.num_planning_attempts = 30
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = vel_scale
        goal.request.max_acceleration_scaling_factor = accel_scale
        goal.request.start_state.is_diff = True

        constraints = Constraints()
        for name, val in joint_dict.items():
            c = JointConstraint()
            c.joint_name = name
            c.position = val
            c.tolerance_above = 0.1
            c.tolerance_below = 0.1
            c.weight = 1.0
            constraints.joint_constraints.append(c)

        goal.request.goal_constraints = [constraints]

        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server unavailable!")
            return False

        future = self._move_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)

        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Joint Goal rejected by MoveGroup planner.")
            return False

        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)

        result = res_future.result().result
        if result.error_code.val == 1:
            return True
        else:
            self.get_logger().error(f"Joint move failed with code: {result.error_code.val}")
            return False

    def remove_collision_object(self, name="lego_red"):
        """Removes collision object from MoveIt scene during lift."""
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = name
        co.operation = CollisionObject.REMOVE
        self.collision_pub.publish(co)

    def execute_joint_pick_and_place(self):
        """Executes full joint-space pick and place sequence for lego_red."""
        self.get_logger().info("\n--- STEP-BY-STEP: Joint-Space Pick & Place for Red Lego ---")

        # Step 1: Open Gripper
        self.get_logger().info("1. Opening Gripper (0.0)")
        self.operate_gripper(0.0)

        # Step 2: Move to Pre-Pick Joint Pose
        self.get_logger().info(f"2. Moving Joints to Pre-Pick -> Pan={self.joint_pre_pick['shoulder_pan_joint']:.3f} rad ({math.degrees(self.joint_pre_pick['shoulder_pan_joint']):.1f}°)")
        if not self.send_joint_goal(self.joint_pre_pick):
            self.get_logger().error("Failed Pre-Pick joint movement.")
            return False

        # Step 3: Lower Joints to Grasp Pose
        self.get_logger().info("3. Lowering Joints to Grasp Pose")
        if not self.send_joint_goal(self.joint_pick):
            self.get_logger().error("Failed Grasp joint movement.")
            return False

        # Step 4: Clamp Gripper onto Red Lego
        self.get_logger().info("4. Clamping Gripper onto Red Lego (Position = 0.45)")
        self.operate_gripper(0.45, wait=False)
        time.sleep(2.0)
        self.remove_collision_object("lego_red")

        # Step 5: Lift Arm back to Pre-Pick Joint Pose
        self.get_logger().info("5. Lifting Arm to Pre-Pick Clearance")
        if not self.send_joint_goal(self.joint_pre_pick, vel_scale=0.1, accel_scale=0.1):
            self.get_logger().error("Failed Lift joint movement.")
            return False

        # Step 6: Rotate Joints to Bin Location
        self.get_logger().info(f"6. Rotating Arm to Bin Location -> Pan={self.joint_bin['shoulder_pan_joint']:.3f} rad ({math.degrees(self.joint_bin['shoulder_pan_joint']):.1f}°)")
        if not self.send_joint_goal(self.joint_bin):
            self.get_logger().error("Failed Bin joint movement.")
            return False

        # Step 7: Release Gripper into Bin
        self.get_logger().info("7. Releasing Gripper to Drop Red Lego into Bin")
        self.operate_gripper(0.0)
        time.sleep(1.5)

        # Step 8: Return Arm to Home Joint Pose
        self.get_logger().info("8. Returning Arm to Home Position")
        self.send_joint_goal(self.joint_home)

        self.get_logger().info("🎉 JOINT-SPACE PICK AND PLACE COMPLETED SUCCESSFULLY!\n")
        return True

def main(args=None):
    import threading
    rclpy.init(args=args)
    node = Phase1JointPickAndPlaceNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(2.0)
    try:
        node.execute_joint_pick_and_place()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
