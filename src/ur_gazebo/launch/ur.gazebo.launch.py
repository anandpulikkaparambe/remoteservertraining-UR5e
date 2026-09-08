"""
Launch Gazebo simulation with a UR robot.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
import yaml

def generate_launch_description():
    # Package names
    package_name_gazebo = 'ur_gazebo'
    package_name_moveit = 'moveit_config'

    # Get package paths as strings
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_share_gazebo = get_package_share_directory(package_name_gazebo)
    pkg_share_moveit = get_package_share_directory(package_name_moveit)
    pkg_share_ur_description = get_package_share_directory("ur_description")

    pkg_share_robotiq_140 = get_package_share_directory("robotiq_2f_gripper_description")
    pkg_share_ur3e_sorting = get_package_share_directory('ur3e_sorting')

    # Paths
    gazebo_models_path = os.path.join(pkg_share_gazebo, 'models')
    gazebo_worlds_path = os.path.join(pkg_share_gazebo, 'worlds')
    ros_gz_bridge_config_file_path = os.path.join(pkg_share_gazebo, 'config', 'ros_gz_bridge.yaml')

    # Launch Configurations
    world_file = LaunchConfiguration('world_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    ur_type = LaunchConfiguration('ur_type')
    gripper = LaunchConfiguration('gripper')
    namespace = LaunchConfiguration('namespace')
    gz_partition = LaunchConfiguration('gz_partition')
    headless = LaunchConfiguration('headless')
    launch_rviz = LaunchConfiguration('launch_rviz')

    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument("gripper", default_value="robotiq_140", description="End effector type (e.g. none, robotiq_140, robotiq_85, vacuum)"),
        DeclareLaunchArgument("robot_name", default_value="ur", description="The name for the robot"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation (Gazebo) clock if true"),
        DeclareLaunchArgument("world_file", default_value="pick_and_place_demo.world", description="World file name"),
        DeclareLaunchArgument("ur_type", default_value="ur5e", description="Type/series of UR robot"),
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?"),
        DeclareLaunchArgument("spawn_x", default_value="0.0", description="Robot spawn X"),
        DeclareLaunchArgument("spawn_y", default_value="0.0", description="Robot spawn Y"),
        DeclareLaunchArgument("spawn_z", default_value="0.0", description="Robot spawn Z (Root at 0, URDF handles offset)"),
        DeclareLaunchArgument("spawn_yaw", default_value="0.0", description="Robot spawn Yaw"),
        DeclareLaunchArgument("run_yolo", default_value="false", description="Run YOLO perception node"),
        DeclareLaunchArgument("run_sorting", default_value="false", description="Run Sorting Logic node"),
        DeclareLaunchArgument("namespace", default_value="", description="ROS namespace for this whole stack (robot_state_publisher, bridge, controllers, move_group, rviz) -- set to a unique value to run a second instance alongside one already using the default ''. `ros2 run ur3e_rl train_sac --num-envs N` auto-targets instance i (i>0) at namespace 'envI', so use that exact naming ('env1', 'env2', ...) if you plan to train against it."),
        DeclareLaunchArgument("gz_partition", default_value="", description="GZ_PARTITION for this instance's gz-transport bus -- set to a unique value per concurrent instance (matching `namespace` above, e.g. 'env1') so their Gazebo processes don't cross-talk."),
        DeclareLaunchArgument("headless", default_value="false", description="Run gz sim server-only (-s, no GUI window) to cut CPU/RAM -- recommended for any instance beyond the first."),
    ]

    ld = LaunchDescription(declared_arguments)

    # Spawn Configs
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # Set Gazebo Resource Path for Meshes
    # We append the Parent Directories of the packages so that "package://pkg_name" resolution works
    # Gazebo usually looks into paths in GZ_SIM_RESOURCE_PATH.
    gz_resource_path = (
        os.path.join(pkg_share_robotiq_140, '..') + ':' +
        os.path.join(pkg_share_ur_description, '..') + ':' +
        os.path.join(pkg_share_ur3e_sorting, '..') + ':' +
        gazebo_models_path
    )
    
    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        gz_resource_path
    )

    # Pin gz-transport discovery to loopback so it never depends on wifi/LAN
    # being up: without this, gz-transport's multicast discovery heartbeat
    # fails with "Exception sending a multicast message: Network is
    # unreachable" whenever there's no interface with a valid route (e.g.
    # wifi disconnected), which stalls pub/sub between gzserver and the
    # ROS-GZ bridge and looks like training/sim randomly pausing.
    #
    # 2026-09-04: GZ_IP alone was confirmed live to do NOTHING on this stack --
    # this is Ignition Fortress (binary is `ign gazebo`, launched with
    # --force-version 6), whose actual transport library
    # (libignition-transport11.so) only reads IGN_IP, not GZ_IP. Checked directly
    # via `strings` on the installed .so: it contains the symbol "IGN_IP", not
    # "GZ_IP". GZ_IP is the newer (post Ignition->Gazebo rename, Garden/Harmonic+)
    # variable name -- setting only that on a Fortress install silently does
    # nothing, no error, which is exactly why this "fix" never actually worked
    # despite being in this file since 2026-08-19. Set both: IGN_IP is the one
    # that's actually load-bearing here, GZ_IP is kept harmlessly in case any
    # newer-Gazebo-aware tool in this stack does check it.
    set_ign_ip = SetEnvironmentVariable('IGN_IP', '127.0.0.1')
    set_gz_ip = SetEnvironmentVariable('GZ_IP', '127.0.0.1')

    # Isolates this instance's gz-transport bus from any other concurrently-running
    # instance (see namespace/headless args above) -- without a unique partition per
    # instance, two `gz sim` processes on the same machine would see and cross-talk on
    # each other's topics/services even though ROS-side namespacing keeps the ROS graphs
    # separate.
    #
    # Only set the env var when gz_partition is actually non-blank -- unconditionally
    # calling SetEnvironmentVariable('GZ_PARTITION', '') sets GZ_PARTITION to an EMPTY
    # STRING, which gz-transport treats differently from the variable being absent
    # entirely: absent falls back to its own auto-generated (but consistent, shared)
    # default partition, while present-but-empty pins this instance to a distinct empty
    # partition that a separate, truly-unset process (e.g. train_sac run in another
    # terminal, which never touches GZ_PARTITION at all) never joins. That mismatch is
    # exactly what caused instance 0's `_get_gz_object_position` gz-transport queries to
    # time out after this arg was first added -- confirmed live via the resulting
    # "no message received on .../pose/info" warning once that failure was made visible
    # instead of being swallowed by a bare except (see _get_gz_object_position).
    set_gz_partition = SetEnvironmentVariable(
        'GZ_PARTITION', gz_partition,
        condition=IfCondition(PythonExpression(["'", gz_partition, "' != ''"]))
    )

    # Robot Description (URDF)
    urdf_xacro_path = PathJoinSubstitution([
        pkg_share_ur_description,
        "urdf",
        ["ur_", gripper, ".urdf.xacro"]
    ])
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ", urdf_xacro_path, " ur_type:=", ur_type, " use_camera:=true",
        " robot_name:=", robot_name, " namespace:=", namespace
    ])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # Robot State Publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='both',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )

    # Gazebo ROS Bridge
    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        namespace=namespace,
        parameters=[{'config_file': ros_gz_bridge_config_file_path}],
        output='screen'
    )

    # Controller Manager
    # Controller Manager is handled by the Gazebo plugin (gz_ros2_control)
    # controller_manager_node = Node(...) 

    # Controllers. gripper_controller (GripperActionController, driving the Robotiq
    # mimic-jointed finger_joint) is spawned for the robotiq_85/robotiq_140 gripper
    # paths -- both declare a command_interface on finger_joint (ur.ros2_control.xacro).
    controllers = ["joint_state_broadcaster", "arm_controller", "gripper_controller"]
    delays = [15.0, 20.0, 25.0]

    for controller, delay in zip(controllers, delays):
        ld.add_action(
            RegisterEventHandler(
                OnProcessStart(
                    target_action=start_gazebo_ros_bridge_cmd,
                    on_start=[
                        TimerAction(
                            period=delay,
                            actions=[
                                Node(
                                    package="controller_manager",
                                    executable="spawner",
                                    namespace=namespace,
                                    # relative (not "/controller_manager") so this
                                    # resolves under `namespace` -- required for a
                                    # second concurrent instance to reach its own
                                    # controller_manager instead of instance 0's.
                                    arguments=[controller, "-c", "controller_manager"],
                                    parameters=[{'use_sim_time': True}],
                                    output='screen'
                                )
                            ]
                        )
                    ]
                )
            )
        )

    # Gazebo launch. -s (server-only, no GUI window) when headless:=true -- the gzclient
    # GUI is the single heaviest process in this stack (CPU+RAM), so any instance beyond
    # the first should normally run headless.
    gz_args_flags = PythonExpression(["'-s -r -v 4 ' if '", headless, "' == 'true' else '-r -v 4 '"])
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [gz_args_flags, PathJoinSubstitution([gazebo_worlds_path, world_file])],
            'use_sim_time': 'true'
        }.items()
    )

    # Spawn Robot
    start_gazebo_ros_spawner_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=namespace,
        output='screen',
        arguments=[
            # relative -- resolves under `namespace` to match robot_state_publisher's
            # own (also relative/namespaced) robot_description topic.
            '-topic', 'robot_description',
            '-name', robot_name,
            '-allow_renaming', 'true',
            '-x', spawn_x, '-y', spawn_y, '-z', spawn_z,
            '-R', '0.0', '-P', '0.0', '-Y', spawn_yaw
        ]
    )

    # MoveIt Config
    # Now that we have path strings, we can pass them to MoveItConfigsBuilder better?
    # Or just use the one that works.
    # MoveIt Config
    moveit_config = (
        MoveItConfigsBuilder("ur", package_name=package_name_moveit)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["pilz_industrial_motion_planner", "ompl", "ompl_ruckig", "chomp"],
            default_planning_pipeline="ompl_ruckig"
        )
        .to_moveit_configs()
    )
    
    # Overriding robot_description to ensure it uses the one with the gripper
    moveit_params = moveit_config.to_dict()
    moveit_params.update(robot_description)
    # move_group's MoveGroupContext keeps one un-namespaced "default" planning pipeline
    # (separate from the pipeline_id-keyed map built by .planning_pipelines() above,
    # which every actual planning request here selects from by name). That default is
    # resolved by pluginlib scanning ALL planner-interface plugins installed system-wide
    # (both ompl_interface/OMPLPlanner and chomp_interface/CHOMPPlanner are always
    # visible here, regardless of which pipelines are listed above) -- confirmed live via
    # move_group's own boot log: "Multiple planning plugins available. You should specify
    # the '~planning_plugin' parameter. Using 'chomp_interface/CHOMPPlanner' for now."
    # That silently-chosen default is NOT cosmetic: every MoveGroup pose-goal request
    # here (group "arm") failed with error -16 (INVALID_GOAL_CONSTRAINTS) / "Only
    # joint-space goals are supported", tracing straight into chomp_planner, even though
    # the request's own pipeline_id ("ompl"/"ompl_ruckig") logged as selected correctly.
    # Pinning this explicitly resolves the ambiguity in OMPL's favor.
    moveit_params['planning_plugin'] = 'ompl_interface/OMPLPlanner'

    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[moveit_params, {'use_sim_time': use_sim_time}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        namespace=namespace,
        output="log",
        arguments=["-d", os.path.join(pkg_share_moveit, "config", "moveit.rviz")],
        parameters=[
            moveit_params,
        ],
        condition=IfCondition(launch_rviz),
    )

    # Add actions
    ld.add_action(set_ign_ip)
    ld.add_action(set_gz_ip)
    ld.add_action(set_gz_partition)
    ld.add_action(set_env_vars_resources)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_gazebo_ros_bridge_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd)
    # Controller Manager is internal to Gazebo plugin, no need to add external node.
    ld.add_action(run_move_group_node)
    ld.add_action(rviz_node)

    # Automatically pop up the wrist camera feed
    # ld.add_action(
    #     Node(
    #         package='rqt_image_view',
    #         executable='rqt_image_view',
    #         name='wrist_camera_view',
    #         arguments=['/wrist_camera/image_raw'],
    #     )
    # )

    # Optional perception and sorting nodes
    run_yolo = LaunchConfiguration('run_yolo')
    run_sorting = LaunchConfiguration('run_sorting')

    # Re-enabled 2026-09-08 (Vast.ai/YOLO-integration fork) -- was fully commented out
    # before, so run_yolo was a dead launch arg with nothing actually consuming it.
    # namespace=namespace added to match sorting_node below (multi-instance isolation).
    ld.add_action(
        Node(
            package='ur3e_sorting',
            executable='yolo_detector',
            name='perception_node',
            namespace=namespace,
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(run_yolo)
        )
    )

    ld.add_action(
        Node(
            package='ur3e_sorting',
            executable='sorting_node',
            name='sorting_node',
            namespace=namespace,
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(run_sorting)
        )
    )

    return ld
