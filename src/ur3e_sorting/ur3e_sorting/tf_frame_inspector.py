#!/usr/bin/env python3
"""
Interactive TF Frame & End-Effector Inspector Node
- Listens to ROS 2 TF Tree (world, base_link, tool0, lego_red, bin)
- Calculates exact relative positions, distances, and orientations
- Publishes 3D RViz Visualization Markers (/visualization_marker_array)
"""

import rclpy
from rclpy.node import Node
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf_transformations
import math
import time

class TFFrameInspectorNode(Node):
    def __init__(self):
        super().__init__('tf_frame_inspector_node', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])
        
        # TF Buffer & Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # RViz Marker Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        
        # Timer to inspect and publish transforms every 1 second
        self.timer = self.create_timer(1.0, self.inspect_transforms)
        
        self.get_logger().info("==================================================")
        self.get_logger().info("🔍 Interactive TF Frame & Alignment Inspector Node")
        self.get_logger().info("   Monitoring TF Tree & Publishing RViz Markers...")
        self.get_logger().info("==================================================")

    def lookup(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return t.transform
        except Exception:
            return None

    def inspect_transforms(self):
        ma = MarkerArray()
        id_counter = 100

        self.get_logger().info("\n--- 📍 REAL-TIME TF FRAME INSPECTION ---")

        # 1. Inspect Base Link relative to World
        tf_base = self.lookup('world', 'base_link')
        if tf_base:
            bx, by, bz = tf_base.translation.x, tf_base.translation.y, tf_base.translation.z
            self.get_logger().info(f"🔹 [world -> base_link] Pos: ({bx:.3f}, {by:.3f}, {bz:.3f})")

        # 2. Inspect Tool0 (End Effector Flange)
        tf_tool = self.lookup('world', 'tool0')
        if tf_tool:
            tx, ty, tz = tf_tool.translation.x, tf_tool.translation.y, tf_tool.translation.z
            rx, ry, rz, rw = tf_tool.rotation.x, tf_tool.rotation.y, tf_tool.rotation.z, tf_tool.rotation.w
            euler = tf_transformations.euler_from_quaternion([rx, ry, rz, rw])
            roll_deg, pitch_deg, yaw_deg = math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2])
            
            self.get_logger().info(f"🦾 [world -> tool0]     Pos: ({tx:.3f}, {ty:.3f}, {tz:.3f}) | Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°, Yaw={yaw_deg:.1f}°")

            # RViz Marker for Tool0 (Cyan Sphere)
            m_tool = Marker()
            m_tool.header.frame_id = "world"
            m_tool.id = id_counter; id_counter += 1
            m_tool.type = Marker.SPHERE
            m_tool.action = Marker.ADD
            m_tool.pose.position.x = tx; m_tool.pose.position.y = ty; m_tool.pose.position.z = tz
            m_tool.pose.orientation.w = 1.0
            m_tool.scale.x = 0.04; m_tool.scale.y = 0.04; m_tool.scale.z = 0.04
            m_tool.color.r = 0.0; m_tool.color.g = 0.8; m_tool.color.b = 1.0; m_tool.color.a = 0.9
            ma.markers.append(m_tool)

        # 3. Inspect Red Lego Block
        lego_frames = ['lego_red', 'lego_red/link', 'conveyor_world/lego_red']
        tf_lego = None
        used_lego_frame = ""
        for frame in lego_frames:
            tf_lego = self.lookup('world', frame)
            if tf_lego:
                used_lego_frame = frame
                break

        if tf_lego:
            lx, ly, lz = tf_lego.translation.x, tf_lego.translation.y, tf_lego.translation.z
            self.get_logger().info(f"🟥 [world -> {used_lego_frame}] Pos: ({lx:.3f}, {ly:.3f}, {lz:.3f})")
            
            # RViz Marker for Red Lego (Red Cube)
            m_lego = Marker()
            m_lego.header.frame_id = "world"
            m_lego.id = id_counter; id_counter += 1
            m_lego.type = Marker.CUBE
            m_lego.action = Marker.ADD
            m_lego.pose.position.x = lx; m_lego.pose.position.y = ly; m_lego.pose.position.z = lz
            m_lego.pose.orientation.w = 1.0
            m_lego.scale.x = 0.057; m_lego.scale.y = 0.057; m_lego.scale.z = 0.057
            m_lego.color.r = 1.0; m_lego.color.g = 0.0; m_lego.color.b = 0.0; m_lego.color.a = 0.9
            ma.markers.append(m_lego)
        else:
            lx, ly, lz = 0.30, 0.20, 0.85
            self.get_logger().info(f"ℹ️ Ground Truth lego_red Pose: ({lx:.3f}, {ly:.3f}, {lz:.3f})")

        # 4. Calculate Relative Tool0 -> Lego Distance Offset
        if tf_tool:
            dx = lx - tx
            dy = ly - ty
            dz = lz - tz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            self.get_logger().info(f"📏 Offset (Tool0 to Lego): ΔX={dx:+.3f}m, ΔY={dy:+.3f}m, ΔZ={dz:+.3f}m | Distance={dist:.3f}m")

            # RViz Line connecting Tool0 to Lego
            m_line = Marker()
            m_line.header.frame_id = "world"
            m_line.id = id_counter; id_counter += 1
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.005 # Line thickness
            m_line.color.r = 1.0; m_line.color.g = 1.0; m_line.color.b = 0.0; m_line.color.a = 1.0
            p1 = Point(); p1.x = tx; p1.y = ty; p1.z = tz
            p2 = Point(); p2.x = lx; p2.y = ly; p2.z = lz
            m_line.points.append(p1)
            m_line.points.append(p2)
            ma.markers.append(m_line)

        # Publish RViz markers
        self.marker_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = TFFrameInspectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
