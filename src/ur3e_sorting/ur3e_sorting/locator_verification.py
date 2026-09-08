#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import time

class LocatorVerificationNode(Node):
    def __init__(self):
        super().__init__('locator_verification_node')
        
        # Buffer for TF (Ground Truth)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriber for Perception Output
        self.perception_sub = self.create_subscription(
            PoseStamped,
            '/detected_object_pose',
            self.perception_callback,
            10
        )
        
        self.get_logger().info('Locator Verification Node Started.')
        self.get_logger().info('Waiting for /detected_object_pose...')

    def perception_callback(self, msg):
        # 1. Get Perception Coordinates
        p_x = msg.pose.position.x
        p_y = msg.pose.position.y
        p_z = msg.pose.position.z
        
        self.get_logger().info('--- Verification Step ---')
        self.get_logger().info(f'Perception Says: X={p_x:.3f}, Y={p_y:.3f}, Z={p_z:.3f}')
        
        # 2. Get Ground Truth from TF
        try:
            # Try multiple frame candidates
            candidates = ['lego_red', 'lego_red/link', 'conveyor_world/lego_red', 'conveyor_world/lego_red/link']
            transform = None
            found_frame = ""
            
            for frame in candidates:
                try:
                    transform = self.tf_buffer.lookup_transform('world', frame, rclpy.time.Time())
                    found_frame = frame
                    break
                except:
                    continue

            if transform is None:
                self.get_logger().error(f'Could not find ground truth for lego_red in frames: {candidates}')
                try:
                    self.get_logger().info(f'All available frames: {self.tf_buffer.all_frames_as_string()}')
                except:
                    pass
                return

            gt_x = transform.transform.translation.x
            gt_y = transform.transform.translation.y
            gt_z = transform.transform.translation.z
            
            self.get_logger().info(f'Ground Truth ({found_frame}): X={gt_x:.3f}, Y={gt_y:.3f}, Z={gt_z:.3f}')
            
            # 3. Calculate Error
            dist_error = np.sqrt((gt_x - p_x)**2 + (gt_y - p_y)**2 + (gt_z - p_z)**2)
            self.get_logger().info(f'Total Error   : {dist_error:.4f} m')
            
            if dist_error < 0.05: # 5cm tolerance
                self.get_logger().info('✅ ACCURATE MATCH')
            else:
                self.get_logger().warn('❌ HIGH MISMATCH')

        except Exception as e:
            self.get_logger().error(f'Could not get ground truth TF: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LocatorVerificationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
