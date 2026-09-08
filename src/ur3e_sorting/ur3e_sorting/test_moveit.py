#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import math

class MoveItActionTest(Node):
    def __init__(self):
        super().__init__('moveit_action_test', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.group_name = "arm" # Verified in SRDF
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10
        )
        self.current_joints = {}

    def joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def send_joint_goal(self):
        # Wait for joint state
        import time
        for i in range(30):
            if self.current_joints:
                break
            self.get_logger().info("Waiting for joint states...")
            time.sleep(1.0) # Blocking sleep is okay in this test script entry
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.current_joints:
            self.get_logger().error("No joint states received!")
            return

        self.get_logger().info(f"Current Joints: {self.current_joints}")
        
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.group_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.5
        goal_msg.request.max_acceleration_scaling_factor = 0.5

        # SANITY CHECK: Plan to CURRENT pose first to verify start state validity
        self.get_logger().info("Planning to CURRENT pose to check start state validity...")
        target_joints = self.current_joints.copy()
        
        # If that works, we can try a small offset
        # target_joints["shoulder_pan_joint"] += 0.1
        
        constraints = Constraints()
        for name, val in target_joints.items():
            c = JointConstraint()
            c.joint_name = name
            c.position = val
            c.tolerance_above = 0.05
            c.tolerance_below = 0.05
            c.weight = 1.0
            constraints.joint_constraints.append(c)
        
        goal_msg.request.goal_constraints = [constraints]

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending joint goal...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted!')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.error_code.val}')
        if result.error_code.val == 1:
             self.get_logger().info('SUCCESS!')
        else:
             self.get_logger().error(f'Planning failed with error code: {result.error_code.val}')
        
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveItActionTest()
    action_client.send_joint_goal()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
