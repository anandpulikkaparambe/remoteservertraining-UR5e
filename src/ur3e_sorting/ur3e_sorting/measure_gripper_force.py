#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
import time
import numpy as np

class GripperForceMonitor(Node):
    def __init__(self):
        super().__init__('gripper_force_monitor')
        
        # Subscribe to joint states
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10
        )
        
        # Gripper Action Client
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.latest_effort = None
        self.target_joint = "finger_joint" 
        
        self.get_logger().info("Gripper Force Monitor Started.")

    def joint_state_cb(self, msg):
        try:
            # Find index of finger_joint
            if self.target_joint in msg.name:
                idx = msg.name.index(self.target_joint)
                if len(msg.effort) > idx:
                    self.latest_effort = msg.effort[idx]
        except ValueError:
            pass

    def get_effort(self):
        return self.latest_effort

    def operate_gripper(self, width, effort=100.0):
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper server not available")
            return False
        
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = effort
        
        self.get_logger().info(f"Commanding Gripper POS={width}, MAX_EFFORT={effort}")
        future = self._gripper_client.send_goal_async(goal)
        
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
            # Print effort while moving
            current_effort = self.get_effort()
            if current_effort is not None:
                print(f"Current Effort: {current_effort:.4f} Nm", end='\r')
            
            if future.done():
                handle = future.result()
                if not handle.accepted:
                     self.get_logger().error("Goal rejected")
                     return False
                
                # Wait for result
                res_fut = handle.get_result_async()
                while not res_fut.done():
                    rclpy.spin_once(self, timeout_sec=0.05)
                    current_effort = self.get_effort()
                    if current_effort is not None:
                        print(f"Current Effort: {current_effort:.4f} Nm", end='\r')
                
                print("") # Newline
                return True
            
            if time.time() - start_time > 10.0:
                 self.get_logger().warn("Timeout waiting for grip")
                 return False
                 
        return False

    def run_test(self):
        self.get_logger().info("Starting Force Monitor Test...")
        
        # 1. Open
        self.operate_gripper(0.0)
        time.sleep(1.0)
        
        # 2. Close (Empty) - Expect low effort
        self.get_logger().info("Closing Empty...")
        self.operate_gripper(0.8)
        time.sleep(1.0)
        
        self.get_logger().info(f"Static Hold Effort: {self.get_effort()}")
        
        # 3. Open
        self.operate_gripper(0.0)

def main(args=None):
    rclpy.init(args=args)
    node = GripperForceMonitor()
    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
