#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PointStamped
import geometry_msgs.msg
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
import tf_transformations

class LegoDetectorNode(Node):
    def __init__(self):
        super().__init__('lego_detector_node')
        
        # Parameters
        self.target_frame = "world"
        self.camera_frame = "camera/link/camera" # Default, updated by camera_info
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseArray, '/lego_detections', 10)
        self.debug_pub = self.create_publisher(Image, '/lego_detector/debug', 10)
        
        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Publish Static TF for Camera (Missing in Bridge)
        # World -> Camera Frame
        # Pose from World File: 0.3 0.0 2.0, 0 1.5708 0
        static_ts = geometry_msgs.msg.TransformStamped()
        static_ts.header.stamp = self.get_clock().now().to_msg()
        static_ts.header.frame_id = "world"
        static_ts.child_frame_id = "camera/link/camera"
        static_ts.transform.translation.x = 0.3
        static_ts.transform.translation.y = 0.0
        static_ts.transform.translation.z = 2.0
        
        # Euler (0, 1.5708, 0) -> Quaternion
        # Gazebo Camera Link usually X-forward. Optical frame Z-forward. 
        # But let's trust the frame ID matches the optical simulation logic or adjust if points are rotated.
        # Ideally, we should check what ROS expects. 
        # Using 0, 1.5708, 0 quat: 0.0, 0.707, 0.0, 0.707
        q = tf_transformations.quaternion_from_euler(0, 1.5708, 0)
        static_ts.transform.rotation.x = q[0]
        static_ts.transform.rotation.y = q[1]
        static_ts.transform.rotation.z = q[2]
        static_ts.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(static_ts)
        
        # Tools
        self.bridge = CvBridge()
        self.camera_model = None
        
        # Color Ranges (HSV)
        # Relaxed thresholds for robustness
        self.colors = {
            "red":    ([0, 50, 50], [10, 255, 255]),
            "blue":   ([110, 50, 50], [130, 255, 255]),
            "green":  ([50, 50, 50], [70, 255, 255]),
            "yellow": ([20, 50, 50], [40, 255, 255])
        }
        
        self.get_logger().info("Lego Detector Node Started")

    def info_callback(self, msg):
        self.camera_model = msg
        self.camera_frame = msg.header.frame_id
        # We only need info once mainly, but keeping it updated is fine
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg):
        if self.camera_model is None:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        # Process each color
        for color_name, (lower, upper) in self.colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Clean noise
            kernel = np.ones((3,3), np.uint8) # Reduced kernel
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 20: # Filter small noise (Lego is ~200px, but be safe)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Draw debug
                        cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
                        cv2.putText(cv_image, color_name, (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Deproject
                        # Z depth?
                        # Camera Z=2.0, Object Z=0.82. Diff = 1.18m.
                        # BUT, this distance is along the optical axis.
                        # Assuming camera is looking straight down:
                        Z_dist = 1.18 
                        
                        # Pinhole Model
                        x_cam = (cX - self.cx) * Z_dist / self.fx
                        y_cam = (cY - self.cy) * Z_dist / self.fy
                        z_cam = Z_dist
                        
                        # Transform to World
                        p_cam = PointStamped()
                        p_cam.header.frame_id = self.camera_frame
                        p_cam.header.stamp = rclpy.time.Time().to_msg() # Use latest available transform
                        p_cam.point.x = x_cam
                        p_cam.point.y = y_cam
                        p_cam.point.z = z_cam
                        
                        try:
                            # Use latest common time or just latest transform
                            # In simulation, better to use Time() for lookupLatest
                            p_world = self.tf_buffer.transform(p_cam, self.target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
                            
                            pose = Pose()
                            pose.position.x = p_world.point.x
                            pose.position.y = p_world.point.y
                            pose.position.z = 0.82 # Force Z to verify/safety (Lego height)
                            # Orientation? Keep default (upright)
                            pose.orientation.w = 1.0
                            
                            pose_array.poses.append(pose)
                            
                        except Exception as e:
                            self.get_logger().warn(f"TF Transform failed: {e}")
                            pass

        self.pose_pub.publish(pose_array)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = LegoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
