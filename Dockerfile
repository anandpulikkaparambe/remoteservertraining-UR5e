FROM osrf/ros:humble-desktop-full

# Install system dependencies, pip, and clean up apt cache
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-moveit \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# Newer setuptools calls packaging.utils.canonicalize_version() with an
# argument the apt-installed packaging lacks, which breaks colcon's
# ament_python builds; upgrading packaging via pip resolves the mismatch.
RUN pip3 install --no-cache-dir -U packaging

# Install CPU-only torch first so ultralytics/stable_baselines3 don't pull in
# the full CUDA toolkit (several GB of wheels) as a transitive dependency.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies for YOLOv8 and the RL training stack.
# numpy pinned <1.24 -- ROS2 Humble's apt-installed transforms3d (pulled in by
# tf_transformations) calls np.maximum_sctype(np.float) at import time: np.float was
# removed in NumPy 1.24 and np.maximum_sctype in NumPy 2.0, so anything >=1.24 (not just
# >=2) crashes every script that imports tf_transformations. Confirmed live: 1.26.4
# still breaks on np.float; 1.23.5 (last pre-1.24 release) is the actual floor. An
# unpinned install here pulls latest numpy and breaks this immediately. Verified
# torch/opencv-python/gymnasium/stable_baselines3/ultralytics all still import fine
# under 1.23.5 despite opencv-python's own (non-fatal) declared numpy>=2 requirement.
RUN pip3 install --no-cache-dir --timeout 120 --retries 10 ultralytics opencv-python "numpy<1.24" gymnasium stable_baselines3 tensorboard

# Disable the CHOMP planner plugin's pluginlib registration. MoveGroupContext keeps one
# un-namespaced "default" planning plugin (separate from the pipeline_id-keyed pipeline
# map that ur.gazebo.launch.py's MoveItConfigsBuilder actually builds and that every real
# planning request here selects from by name) -- confirmed live via move_group's own boot
# log: "Multiple planning plugins available. You should specify the '~planning_plugin'
# parameter. Using 'chomp_interface/CHOMPPlanner' for now." Explicitly setting the
# `planning_plugin` ROS param to ompl_interface/OMPLPlanner (see ur.gazebo.launch.py) did
# NOT fix this -- the ambiguity resolution appears to ignore it in this exact MoveIt
# build. With CHOMP silently winning, every pose-goal MoveGroup request (group "arm")
# failed with error -16 (INVALID_GOAL_CONSTRAINTS) / "Only joint-space goals are
# supported", tracing straight into chomp_planner. Removing CHOMP's plugin manifest from
# pluginlib's discovery leaves only one real candidate, so the ambiguous default resolves
# to OMPL correctly. moveit-planners-chomp itself is left installed (only the discovery
# manifest is renamed) so this is easy to reverse if a future MoveIt/ros-humble point
# release fixes the underlying ambiguity-resolution bug.
RUN mv /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml \
       /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml.disabled

# Create the workspace directory
WORKDIR /ros2_ws

# Copy the source code into the workspace
# We only copy the src directory and the weights, to avoid bringing in build artifacts
COPY src /ros2_ws/src
COPY YoloV8_v2 /ros2_ws/YoloV8_v2
COPY vastai_train_entrypoint.sh /ros2_ws/vastai_train_entrypoint.sh
RUN chmod +x /ros2_ws/vastai_train_entrypoint.sh

# Install ROS 2 dependencies
# warehouse_ros_mongo is skipped: it's an optional MoveIt warehouse-storage
# plugin (unused by this project) whose apt package no longer exists upstream.
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys warehouse_ros_mongo && \
    rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Setup entrypoint to source both system and workspace installations
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Set the default command (can be overridden when running the container)
CMD ["/bin/bash"]
