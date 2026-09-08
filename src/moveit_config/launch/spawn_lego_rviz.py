#!/usr/bin/env python3
"""
Spawn LEGO bricks as virtual collision objects in MoveIt planning scene.
No Gazebo required - works with RViz and fake hardware only.

Usage:
    python3 spawn_lego_rviz.py
"""

import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import time

def spawn_lego_objects():
    """Create and publish LEGO collision objects to MoveIt planning scene."""
    
    rclpy.init()
    node = rclpy.create_node('lego_spawner')
    
    # Publisher to planning scene
    scene_pub = node.create_publisher(PlanningScene, 'planning_scene', 10)
    
    # Wait for publishers to connect
    time.sleep(0.5)
    
    # Create planning scene message
    planning_scene = PlanningScene()
    planning_scene.is_diff = True
    
    # Define LEGO positions (modify these as needed)
    lego_positions = [
        {'x': 0.5, 'y': 0.2, 'z': 0.05, 'id': 'lego_1', 'color': 'blue'},
        {'x': 0.45, 'y': 0.25, 'z': 0.05, 'id': 'lego_2', 'color': 'red'},
        {'x': 0.55, 'y': 0.25, 'z': 0.05, 'id': 'lego_3', 'color': 'green'},
        {'x': 0.5, 'y': 0.15, 'z': 0.05, 'id': 'lego_4', 'color': 'yellow'},
        {'x': 0.4, 'y': 0.3, 'z': 0.05, 'id': 'lego_5', 'color': 'orange'},
    ]
    
    print("\n" + "="*60)
    print("LEGO SPAWNER - RViz Planning Scene")
    print("="*60)
    
    # Create collision objects for each LEGO
    for lego_data in lego_positions:
        # Create collision object
        collision_obj = CollisionObject()
        collision_obj.header.frame_id = "base_link"
        collision_obj.id = lego_data['id']
        collision_obj.operation = CollisionObject.ADD
        
        # LEGO brick size (2cm x 2cm x 2cm)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.02, 0.02]
        
        # LEGO position in workspace
        pose = Pose()
        pose.position.x = lego_data['x']
        pose.position.y = lego_data['y']
        pose.position.z = lego_data['z']
        pose.orientation.w = 1.0
        
        collision_obj.primitives.append(box)
        collision_obj.primitive_poses.append(pose)
        
        # Add to planning scene
        planning_scene.world.collision_objects.append(collision_obj)
        
        print(f"✓ {lego_data['id']:8} ({lego_data['color']:6}) @ ({lego_data['x']:.2f}, {lego_data['y']:.2f}, {lego_data['z']:.2f})")
    
    # Publish planning scene with all LEGO objects
    scene_pub.publish(planning_scene)
    print("\n✓ Published all LEGO objects to MoveIt planning scene!")
    print("✓ LEGO bricks should now appear as boxes in RViz")
    print("✓ MoveIt will avoid collisions with spawned LEGO\n")
    print("="*60)
    print("Node running... Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Keep node running to maintain published objects
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    spawn_lego_objects()
