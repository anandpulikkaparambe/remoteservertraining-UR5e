#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import tf_transformations
import time
import copy

class AutomationNode(Node):
    def __init__(self):
        super().__init__('automation_node')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Gripper Client
        from control_msgs.action import GripperCommand
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        self.get_logger().info("Automation Node Started. Waiting logic...")

    def operate_gripper(self, width):
        from control_msgs.action import GripperCommand
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            return False
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = 100.0
        future = self._gripper_client.send_goal_async(goal)
        try:
            handle = future.result()
            if not handle or not handle.accepted: return False
            handle.get_result_async().result() # Wait
        except Exception as e:
            self.get_logger().error(f"Gripper error: {e}")
            return False
        return True

    def create_pose_goal(self, pose):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "arm"
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        
        # Anti-Swing / Anti-Crash Parameters
        goal_msg.request.max_velocity_scaling_factor = 0.1      # Slow down to 10%
        goal_msg.request.max_acceleration_scaling_factor = 0.1  # Slow acceleration
        
        pcm = PositionConstraint()
        pcm.header.frame_id = "world"
        pcm.link_name = "tool0"
        pcm.target_point_offset.x = 0.0; pcm.target_point_offset.y = 0.0; pcm.target_point_offset.z = 0.0
        pcm.weight = 1.0
        
        box = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01] # Precise constraint
        box.primitives.append(primitive)
        p = Pose(); p.position = pose.position; p.orientation.w = 1.0
        box.primitive_poses.append(p)
        pcm.constraint_region = box
        
        ocm = OrientationConstraint()
        ocm.header.frame_id = "world"
        ocm.link_name = "tool0"
        ocm.orientation = pose.orientation
        ocm.absolute_x_axis_tolerance = 0.1 # Tighter tolerance (was 0.5)
        ocm.absolute_y_axis_tolerance = 0.1
        ocm.absolute_z_axis_tolerance = 0.1
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)
        goal_msg.request.goal_constraints = [constraints]
        return goal_msg

    def execute(self, goal_msg):
        self.get_logger().info("Sending Goal...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action Server not available")
            return False
        
        future = self._action_client.send_goal_async(goal_msg)
        while not future.done(): time.sleep(0.1)
        
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Goal Rejected")
            return False
            
        res_fut = handle.get_result_async()
        while not res_fut.done(): time.sleep(0.1)
        
        res = res_fut.result().result
        if res.error_code.val == 1: return True
        else:
            self.get_logger().error(f"Move Failed: {res.error_code.val}")
            return False

    def run_sequence(self):
        # --- PARAMETERS TO TUNE ---
        # 1. Start Position of Lego (from Manual Verification)
        PICK_X = 0.300
        PICK_Y = 0.618
        PICK_Z = 0.82
        
        # 2. Drop Position (Above Bin)
        PLACE_X = -0.3
        PLACE_Y = 0.3
        PLACE_Z = 0.95
        
        # 3. Gripper Mechanics
        # Distance from TCP (tool0) to actual fingertip center.
        # Derived from Manual Test: Tool Z (1.03) - Lego Z (0.82) = 0.21
        GRIPPER_OFFSET = 0.21 
        
        # 4. Approach/Lift Heights
        APPROACH_HEIGHT = 0.15
        
        # Define Poses
        pick_xyz = [PICK_X, PICK_Y, PICK_Z] 
        place_xyz = [PLACE_X, PLACE_Y, PLACE_Z]
        
        # Orientation (From User's Manual Grasp Log)
        # q = tf_transformations.quaternion_from_euler(3.14159, 0, 1.5708)
        q = [-0.614, 0.786, -0.065, 0.010]
        
        # 1. Pre-Pick
        target = Pose()
        target.position.x = pick_xyz[0]
        target.position.y = pick_xyz[1]
        target.position.z = pick_xyz[2] + GRIPPER_OFFSET + APPROACH_HEIGHT
        target.orientation.x = q[0]; target.orientation.y = q[1]; target.orientation.z = q[2]; target.orientation.w = q[3]
        
        self.get_logger().info("Moving to Pre-Pick...")
        self.operate_gripper(0.0) # Open
        if not self.execute(self.create_pose_goal(target)): return
        
        # 2. Pick (Lower Slowly)
        target.position.z = pick_xyz[2] + GRIPPER_OFFSET
        self.get_logger().info(f"Picking at Z={target.position.z:.3f}...")
        if not self.execute(self.create_pose_goal(target)): return
        
        # 3. Grasp
        # Note: In Gazebo without a grasp plugin, this relies on friction.
        # Ensure 'width' 0.6 provides enough contact force.
        self.operate_gripper(0.6) 
        time.sleep(1.0)
        
        # 4. Lift
        target.position.z += APPROACH_HEIGHT
        self.get_logger().info("Lifting...")
        if not self.execute(self.create_pose_goal(target)): return
        
        # 5. Move to Place
        target.position.x = place_xyz[0]
        target.position.y = place_xyz[1]
        target.position.z = place_xyz[2] + GRIPPER_OFFSET
        self.get_logger().info("Moving to Bin...")
        if not self.execute(self.create_pose_goal(target)): return
        
        # 6. Drop
        self.operate_gripper(0.0)
        time.sleep(1.0)
        
        self.get_logger().info("Sequence Complete!")

def main(args=None):
    rclpy.init(args=args)
    node = AutomationNode()
    
    # Delay to ensure system is ready
    time.sleep(5.0)
    node.run_sequence()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
