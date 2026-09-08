#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

from rclpy.qos import qos_profile_sensor_data
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs  # noqa: F401 -- registers PoseStamped conversions for Buffer.transform()

DEFAULT_MODEL_PATH = os.environ.get(
    'YOLO_MODEL_PATH',
    '/ros2_ws/YoloV8_v2/runs/lego_color_20ep/weights/best.pt',
)

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        # Target Color Parameter
        self.declare_parameter('target_color', 'Red')
        self.target_color = self.get_parameter('target_color').get_parameter_value().string_value
        
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return

        self.bridge = CvBridge()
        
        # Subscribers (Using qos_profile_sensor_data for maximum compatibility)
        # 2026-09-09 (Vast.ai pure-RL/wrist-camera experimental track): switched from the
        # fixed overhead camera (/camera/...) to the wrist-mounted D435
        # (/wrist_camera/...) -- the real robot's actual camera placement, unlike the
        # overhead one which doesn't exist on the real rig. See
        # intel_rgbd_cam_d435.urdf.xacro's new camera_head_depth sensor (the wrist camera
        # previously had no depth stream at all) and ros_gz_bridge.yaml's matching bridge
        # entry.
        self.image_sub = self.create_subscription(
            Image,
            '/wrist_camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        # Depth is published as RELIABLE by gz_bridge, so we must match it.
        # RGB seems to work with sensor_data, but Depth is stricter.
        self.depth_sub = self.create_subscription(
            Image,
            '/wrist_camera/depth_image',
            self.depth_callback,
            10
        )

        self.latest_depth_msg = None

        # Camera Intrinsics -- wrist D435 real values (2026-09-09), not the overhead
        # camera's guessed 640x480/1.1rad. Matches camera_head's <camera> block in
        # intel_rgbd_cam_d435.urdf.xacro exactly (horizontal_fov=1.5184, 424x240).
        self.width = 424
        self.height = 240
        self.fov = 1.5184
        self.focal_length = self.width / (2 * np.tan(self.fov / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2

        # Camera Extrinsics (2026-09-09): the wrist camera moves with the arm every step,
        # unlike the overhead camera this file used to assume -- a fixed world-frame offset
        # (the old cam_x_w/y_w/z_w constants) would be wrong the instant the arm moves off
        # its spawn pose. Extrinsics are now looked up live via TF in image_callback
        # instead (see CAMERA_OPTICAL_FRAME below), not stored as constants here.
        CAMERA_OPTICAL_FRAME = 'camera_head_color_optical_frame'
        self.camera_optical_frame = CAMERA_OPTICAL_FRAME
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/detected_object_pose', 10)
        self.debug_pub = self.create_publisher(Image, '/yolo/debug_image', 10)
        
        # VISUALIZATION MARKER (For RViz)
        from visualization_msgs.msg import Marker
        self.marker_pub = self.create_publisher(Marker, '/detected_object_marker', 10)

        self.get_logger().info('YoloDetector Node Initialized (Async Depth)')
        self.get_logger().info(f'Params: fx={self.focal_length:.2f}, optical_frame={self.camera_optical_frame}')

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def image_callback(self, rgb_msg):
        # self.get_logger().info('Received Image!') # DEBUG
        
        if self.latest_depth_msg is None:
            self.get_logger().warn('Waiting for depth image...', throttle_duration_sec=2.0)
            return
            
        depth_msg = self.latest_depth_msg
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Run inference
        # 0.05 to catch EVERYTHING (we filter by ROI later)
        results = self.model(cv_image, verbose=False, conf=0.05)
        
        # Log classes once to be sure
        if not hasattr(self, 'classes_logged'):
            self.get_logger().info(f"Model Classes: {self.model.names}")
            self.classes_logged = True
        
        best_lego = None
        min_dist_to_center = float('inf')
        
        # Defaults for visualization
        detected_class = "Unknown"

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                # Aspect Ratio & Center
                aspect_ratio = float(w) / h if h > 0 else 0
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # --- SEARCH ZONE CONFIG ---
                # 2026-09-09: scaled to self.width/self.height (was hardcoded 640/380 for
                # the overhead camera's 640x480 frame -- on the wrist camera's 424x240
                # frame those constants exceeded the actual image, silently making the ROI
                # filter a no-op). The original rationale (cut off the bottom strip to
                # exclude the robot base) was derived from the overhead view; the wrist
                # camera's occlusion pattern is different (arm/gripper can appear anywhere
                # in frame depending on pose) -- kept as a proportional analog for now
                # rather than redesigned, flagged here rather than assumed still correct.
                ROI_X_MIN = 0
                ROI_X_MAX = self.width
                ROI_Y_MIN = 0
                ROI_Y_MAX = int(self.height * 380 / 480)  # same ~79% cutoff fraction as before
                
                # Visualize Search Zone (Blue Box)
                cv2.rectangle(cv_image, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (255, 255, 0), 2)
                cv2.putText(cv_image, "SEARCH ZONE", (ROI_X_MIN+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

                # Class info (RESTORED)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # RED REFERENCE (User Request)
                if class_name == self.target_color:
                    self.get_logger().info(f"*** {self.target_color} LEGO FOUND! Reference Size: {area} | Ratio: {aspect_ratio:.2f} ***")

                # DEBUG LOG ALL CANDIDATES
                self.get_logger().info(f"CANDIDATE: {class_name} Conf:{conf:.2f} Area:{area} X:{center_x}")

                # --- FILTERING LOGIC ---
                
                # 0. ROI Filter
                if center_x < ROI_X_MIN or center_x > ROI_X_MAX or center_y > ROI_Y_MAX:
                     self.get_logger().info(f"  -> REJECTED {class_name}: Outside Zone (Y={center_y})")
                     continue

                # 1. Size Filter
                # Real Lego is likely 2000-4000. Large Ghost is >10000.
                if area > 8000: # Tightened from 12000
                    self.get_logger().info(f"  -> REJECTED {class_name}: Too Big ({area})")
                    continue 
                if area < 50: 
                    continue 
                    
                # 2. Aspect Ratio Filter (Relaxed)
                if aspect_ratio < 0.2 or aspect_ratio > 5.0: 
                    self.get_logger().info(f"  -> REJECTED {class_name}: Bad Ratio ({aspect_ratio:.2f})")
                    continue
                
                # Class info
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # FINAL ACCEPTANCE LOG
                self.get_logger().info(f"ACCEPTED: {class_name} ({conf:.2f}) at [{x1}, {y1}] Area: {area}")

                # Draw on image
                color = (0, 255, 0)
                if class_name == 'Red': color = (0, 0, 255)
                elif class_name == 'Blue': color = (255, 0, 0)
                elif class_name == 'Yellow': color = (0, 255, 255)
                elif class_name == 'Green': color = (0, 128, 0)
                
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Find Center
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                
                # Logic: Pick ANY Lego that is close to the center and valid depth
                if True: 
                    # Get Depth
                    cX = np.clip(cX, 0, self.width - 1)
                    cY = np.clip(cY, 0, self.height - 1)
                    depth_val = cv_depth[cY, cX]
                    
                    self.get_logger().info(f"  -> Depth at center: {depth_val}")

                    if not np.isnan(depth_val):
                         # DEPTH FILTER:
                         # Camera Z=2.0. Table Z=0.8. Expected Dist = 1.2m.
                         # Robot Arm is taller (>0.8), so Dist < 1.1m.
                         # DISABLED FOR DEBUGGING
                         # if depth_val < 1.15:
                         #     self.get_logger().info(f"SKIPPED {class_name} (Too High/Close: {depth_val:.3f})")
                         #     continue
                         pass
                             
                         # Calc distance to image center to pick the "main" one
                         dist = (cX - self.cx)**2 + (cY - self.cy)**2

                         # PRIORITY LOGIC (Dynamic Target)
                         if class_name == self.target_color:
                             if detected_class != self.target_color: # First Priority Color we see override anything else
                                 min_dist_to_center = dist
                                 best_lego = (cX, cY, depth_val)
                                 detected_class = class_name
                             elif dist < min_dist_to_center: # Closer Priority Color replaces previous one
                                 min_dist_to_center = dist
                                 best_lego = (cX, cY, depth_val)
                                 detected_class = class_name
                         
                         # If it's NOT Priority Color, only pick it if we haven't found one yet
                         elif detected_class != self.target_color and dist < min_dist_to_center:
                             min_dist_to_center = dist
                             best_lego = (cX, cY, depth_val)
                             detected_class = class_name

        # If we found a target
        if best_lego:
            cX, cY, Z_c = best_lego

            # --- 3D Projection Math ---
            # 1. Camera (optical) frame -- standard pinhole convention: X=right, Y=down,
            #    Z=forward. This part is unchanged and correct regardless of where the
            #    camera physically is.
            X_c = (cX - self.cx) * Z_c / self.focal_length
            Y_c = (cY - self.cy) * Z_c / self.focal_length

            # 2. Camera-frame -> world-frame, via a LIVE TF lookup (2026-09-09).
            # The wrist camera moves with the arm every step, unlike the old fixed
            # overhead camera this file used to assume -- a static algebraic offset
            # (the previous X_w = cam_x_w - Y_c / Z_w = 0.815 hack) would only ever be
            # correct for the one arm pose it happened to be tuned against. Transforms
            # the raw camera-frame point through whatever the current
            # camera_head_color_optical_frame -> world transform actually is, same
            # tf_buffer.transform(...) pattern ur3e_env.py's
            # _update_target_pose_from_perception already uses for its own world->
            # base_link step.
            try:
                camera_pose = PoseStamped()
                camera_pose.header.frame_id = self.camera_optical_frame
                camera_pose.header.stamp = rclpy.time.Time().to_msg()
                camera_pose.pose.position.x = float(X_c)
                camera_pose.pose.position.y = float(Y_c)
                camera_pose.pose.position.z = float(Z_c)
                camera_pose.pose.orientation.w = 1.0
                pose_msg = self.tf_buffer.transform(
                    camera_pose, 'world', timeout=rclpy.duration.Duration(seconds=0.2)
                )
                pose_msg.header = rgb_msg.header
                pose_msg.header.frame_id = 'world'
            except Exception as e:
                self.get_logger().warn(
                    f"Could not transform detection from {self.camera_optical_frame} "
                    f"into world: {e}"
                )
                pose_msg = None

            if pose_msg is not None:
                self.pose_pub.publish(pose_msg)

                # --- PUBLISH VISUAL MARKER FOR RVIZ ---
                from visualization_msgs.msg import Marker
                marker = Marker()
                marker.header.frame_id = "world"
                marker.id = 999
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position = pose_msg.pose.position
                # Shift marker up slightly so it sits ON the table, not halfway through
                marker.pose.position.z += 0.01
                marker.scale.x = 0.032; marker.scale.y = 0.064; marker.scale.z = 0.02 # 2x4 Brick
                marker.color.a = 1.0

                # Set Color based on Class
                if detected_class == 'Red': marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
                elif detected_class == 'Blue': marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
                elif detected_class == 'Green': marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
                elif detected_class == 'Yellow': marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0
                else: marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0 # White default

                self.marker_pub.publish(marker)
            # --------------------------------------

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            debug_msg.header = rgb_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {e}')

        # Show local window
        cv2.imshow("YOLO Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    detector = YoloDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
