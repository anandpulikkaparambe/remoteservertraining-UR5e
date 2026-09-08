#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
import time

class GripperTestNode(Node):
    def __init__(self):
        super().__init__('gripper_test_node')
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.get_logger().info("Gripper Test Node Started.")

    def operate_gripper(self, width):
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper server not available")
            return False
        
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = 100.0
        
        self.get_logger().info(f"Sending gripper command: {width}...")
        future = self._gripper_client.send_goal_async(goal)
        
        # Wait for result
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                try:
                    handle = future.result()
                    if not handle.accepted:
                        self.get_logger().error("Goal rejected")
                        return False
                    
                    res_future = handle.get_result_async()
                    while rclpy.ok():
                         rclpy.spin_once(self, timeout_sec=0.1)
                         if res_future.done():
                             result = res_future.result().result
                             self.get_logger().info(f"Result received. Reached: {result.position} vs Target: {width}")
                             return True
                except Exception as e:
                    self.get_logger().error(f"Error: {e}")
                    return False
        return False

    def run_sequence(self):
        self.get_logger().info("Testing Gripper Movement...")
        
        # 1. Open (0.0 based on demo script)
        self.get_logger().info("1. Opening Gripper (0.0)")
        self.operate_gripper(0.0)
        time.sleep(2.0)

        # 2. Close Partway (0.5)
        self.get_logger().info("2. Closing Gripper (0.5)")
        self.operate_gripper(0.5)
        time.sleep(2.0)

        # 3. Close Further (0.8)
        self.get_logger().info("3. Closing Gripper Detailed (0.8)")
        self.operate_gripper(0.8)
        time.sleep(2.0)
        
        # 4. Open Again
        self.get_logger().info("4. Opening Gripper (0.0)")
        self.operate_gripper(0.0)
        
        self.get_logger().info("Test Complete.")

def main(args=None):
    rclpy.init(args=args)
    node = GripperTestNode()
    try:
        node.run_sequence()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
