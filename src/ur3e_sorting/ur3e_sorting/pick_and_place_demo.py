#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import tf_transformations
import time
import copy
from control_msgs.action import GripperCommand

class PickAndPlaceDemoNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_demo_node')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        
        # Publisher for Planning Scene (RViz visualization)
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)
        
        self.get_logger().info("Pick and Place Demo Node Started.")
        self.get_logger().info("Use 'PlanningScene' display in RViz to see objects.")

    def publish_scene_objects(self):
        # Publish Collision Objects to visualize them in RViz
        # Matches pick_and_place_demo.world
        
        # 1. Red Lego (0.057 cube -- lego_red is the "Rubiks cube" model in
        # pick_and_place_demo.world, a different size/pose than the other three legos)
        self.publish_collision_box("lego_red", [0.3, 0.2, 0.85], [0.057, 0.057, 0.057], [1.0, 0.0, 0.0])
        
        # 2. Blue Lego
        self.publish_collision_box("lego_blue", [0.3, -0.2, 0.82], [0.032, 0.032, 0.02], [0.0, 0.0, 1.0])
        
        # 3. Green Lego
        q_green = tf_transformations.quaternion_from_euler(0, 0, 0.785)
        self.publish_collision_box("lego_green", [0.4, 0.0, 0.82], [0.032, 0.032, 0.02], [0.0, 1.0, 0.0], orientation=q_green)
        
        # 4. Yellow Lego
        q_yellow = tf_transformations.quaternion_from_euler(0, 0, 0.2)
        self.publish_collision_box("lego_yellow", [0.35, -0.1, 0.82], [0.032, 0.032, 0.02], [1.0, 1.0, 0.0], orientation=q_yellow)
        
        # 5. Bin
        self.publish_collision_box("bin", [-0.3, 0.3, 0.85], [0.2, 0.2, 0.1], [0.5, 0.5, 0.5])

        # 6. Table
        # Robot Base at 0.8.
        # Set Table Top to 0.75m (5cm gap) to avoid START_STATE_INVALID (Collision with Base).
        # Center = 0.375. Height = 0.75. Footprint (1.0x1.6) matches the real Gazebo table.
        self.publish_collision_box("table", [0.0, 0.0, 0.375], [1.0, 1.6, 0.75], [0.8, 0.6, 0.4])

    def publish_collision_box(self, name, position, size, color, orientation=None):
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = name
        
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2] 
        if orientation is None:
            pose.orientation.w = 1.0
        else:
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[3]

        co.primitives.append(primitive)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        
        self.collision_pub.publish(co)
        self.collision_pub.publish(co)

    def remove_collision_object(self, name):
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = name
        co.operation = CollisionObject.REMOVE
        self.collision_pub.publish(co)

    def operate_gripper(self, width):
        from control_msgs.action import GripperCommand
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper server not available")
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
        goal_msg.request.max_velocity_scaling_factor = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2
        
        pcm = PositionConstraint()
        pcm.header.frame_id = "world"
        pcm.link_name = "tool0"
        pcm.target_point_offset.x = 0.0; pcm.target_point_offset.y = 0.0; pcm.target_point_offset.z = 0.0
        pcm.weight = 1.0
        
        box = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]
        box.primitives.append(primitive)
        p = Pose(); p.position = pose.position; p.orientation.w = 1.0
        box.primitive_poses.append(p)
        pcm.constraint_region = box
        
        ocm = OrientationConstraint()
        ocm.header.frame_id = "world"
        ocm.link_name = "tool0"
        ocm.orientation = pose.orientation
        ocm.absolute_x_axis_tolerance = 0.5 # Relaxed from 0.1
        ocm.absolute_y_axis_tolerance = 0.5
        ocm.absolute_z_axis_tolerance = 0.5
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)
        goal_msg.request.goal_constraints = [constraints]
        return goal_msg

    def execute(self, goal_msg):
        self.get_logger().info("Sending Move Goal...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Move Action Server not available")
            return False
        
        future = self._action_client.send_goal_async(goal_msg)
        self.get_logger().info("Goal Sent. Planning...")
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
        # Publish objects first
        for i in range(5):
             self.publish_scene_objects()
             time.sleep(0.5)

        # --- COORDINATES ---
        # --- COORDINATES (WORLD FRAME) ---
        # Red Lego
        PICK_X = 0.3
        PICK_Y = 0.2
        PICK_Z = 0.82
        
        PLACE_X = -0.3
        PLACE_Y = 0.3
        PLACE_Z = 0.95
        
        # Offsets
        # TUNE THIS: If gripper is too high, DECREASE this value. If it hits the lego/table, INCREASE it.
        GRIPPER_OFFSET = 0.17 # Reduced from 0.21 to lower the gripper (~4cm lower)
        APPROACH_HEIGHT = 0.05 # Reduced to 5cm to keep wrist within workspace (Dist ~0.41m)
        
        # Orientation: Use known good quaternion
        # q = tf_transformations.quaternion_from_euler(math.pi, 0, 0) 
        q = [-0.614, 0.786, -0.065, 0.010]
        
        # 1. Pre-Pick
        target = Pose()
        target.position.x = PICK_X
        target.position.y = PICK_Y
        target.position.z = PICK_Z + GRIPPER_OFFSET + APPROACH_HEIGHT
        target.orientation.x = q[0]; target.orientation.y = q[1]; target.orientation.z = q[2]; target.orientation.w = q[3]
        
        self.get_logger().info("1. Move to Pre-Pick")
        self.operate_gripper(0.0) # Open
        
        # Use simple goal without path constraints for first move
        goal = MoveGroup.Goal()
        goal.request.group_name = "arm"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        
        # Simple Pose Target
        if not self.execute(self.create_pose_goal(target)): return
        
        # 2. Pick (Lower)
        target.position.z = PICK_Z + GRIPPER_OFFSET
        self.get_logger().info(f"2. Lowering to Pick (Z={target.position.z:.3f})")
        if not self.execute(self.create_pose_goal(target)): return
        
        # 3. Grasp
        self.get_logger().info("3. Grasping (Tight - 0.8)")
        # Lego is ~32mm wide. Robotiq 140mm open.
        # Ratio 0.8 -> Gap approx 28mm (Clamping force).
        self.operate_gripper(0.8) 
        time.sleep(2.0) # Wait for stability
        
        # REMOVE COLLISION OBJECT to avoid START_STATE_IN_COLLISION error
        self.get_logger().info("Removing lego_red collision object to allow lift...")
        self.remove_collision_object("lego_red")
        time.sleep(0.5)

        # 4. Lift (Slower to avoid slip)
        target.position.z += APPROACH_HEIGHT
        self.get_logger().info("4. Lifting (Slowly)")
        
        # Create a new goal with slower speed for lifting
        lift_goal = self.create_pose_goal(target)
        lift_goal.request.max_velocity_scaling_factor = 0.05 # Very slow
        lift_goal.request.max_acceleration_scaling_factor = 0.05
        
        if not self.execute(lift_goal): return
        
        # 5. Move to Place (Normal Speed)
        target.position.x = PLACE_X
        target.position.y = PLACE_Y
        target.position.z = PLACE_Z + GRIPPER_OFFSET
        self.get_logger().info("5. Moving to Bin")
        if not self.execute(self.create_pose_goal(target)): return
        
        # 6. Drop
        self.get_logger().info("6. Dropping")
        self.operate_gripper(0.0)
        time.sleep(1.0)
        
        # 7. Home/Up
        target.position.z += 0.1
        self.get_logger().info("7. Back Up")
        self.execute(self.create_pose_goal(target))

        self.get_logger().info("Demo Sequence Complete!")

def main(args=None):
    import threading
    from control_msgs.action import GripperCommand 
    rclpy.init(args=args)
    node = PickAndPlaceDemoNode()
    
    # Spin in a separate thread so callbacks can process
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Wait for things to settle
    time.sleep(2.0)
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
