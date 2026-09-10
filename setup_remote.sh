#!/bin/bash
set -e
cd /ros2_ws

apt-get update && apt-get install -y python3-pip \
    ros-humble-moveit \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    && rm -rf /var/lib/apt/lists/*

pip3 install --no-cache-dir -U packaging

pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

pip3 install --no-cache-dir --timeout 120 --retries 10 \
    ultralytics opencv-python "numpy<1.24" gymnasium stable_baselines3 tensorboard

mv /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml \
   /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml.disabled || true

apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys warehouse_ros_mongo && \
    rm -rf /var/lib/apt/lists/*

source /opt/ros/humble/setup.bash
colcon build

chmod +x vastai_train_entrypoint.sh

echo "SETUP_COMPLETE"
