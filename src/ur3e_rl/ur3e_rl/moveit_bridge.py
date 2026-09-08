import time
import rclpy
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class MoveItHandoffBridge:
    """Phase A (classical) half of the hybrid planner+RL architecture (see hybrid
    planner+RL spec, Recommendation 3): drives the arm from wherever reset() left it to a
    fixed, tight-tolerance pre-grasp standoff above the target via OMPL/MoveIt. RL (Phase
    B, UR3eEnv.step()) then starts every episode from this pose and only has to learn the
    final approach + grasp, not full-arm reachability.

    Deliberately reuses the exact MoveGroup/ExecuteTrajectory action-client pattern and
    pre-grasp pose already verified across 3 clean Phase 1 runs
    (phase1_tf_pick_and_place.py's create_robust_pose_goal / execute_move_with_retry /
    verify_pose_target), scoped down to the single pre-grasp move this design needs --
    the handoff pose is not a fresh guess, see HANDOFF_STANDOFF_M below.
    """

    # Distance from tool0 to the gripper tip -- identical constant to
    # phase1_tf_pick_and_place.py's proven GRIPPER_OFFSET for the Robotiq 2F-140 build.
    # MoveGroup's position constraint here targets tool0 (a standard planning-group
    # link known to the SRDF), not the fingertip frame UR3eEnv tracks for its own
    # observations/reward.
    GRIPPER_OFFSET_M = 0.17

    # Handoff standoff above the grasp point. Reused verbatim from Phase 1's pre_grasp_z
    # (grasp_z + 0.05m) rather than picked fresh -- of every pose in this pipeline, this
    # is the one with 3 clean end-to-end verified runs behind it. See spec section 2.
    HANDOFF_STANDOFF_M = 0.05

    def __init__(self, node, tf_buffer, get_joint_state):
        self.node = node
        self.tf_buffer = tf_buffer
        self._get_joint_state = get_joint_state
        # Relative (not "/move_action" etc.) so these resolve under `node`'s own namespace
        # -- required so a namespaced UR3eEnv (see its `namespace` param) talks to its own
        # move_group instance instead of always hitting the default (unnamespaced) one.
        self._move_client = ActionClient(node, MoveGroup, 'move_action')
        self._execute_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
        self.collision_pub = node.create_publisher(CollisionObject, 'collision_object', 10)
        self._scene_initialized = False

    def ensure_scene(self):
        """Registers the table and bin as MoveIt obstacles exactly once per process.
        Geometry identical to phase1_tf_pick_and_place.py's add_static_scene_objects (table
        5cm trimmed off the real 0.80m tabletop so the grasp approach itself isn't flagged
        as touching it; bin box matches its real Gazebo collision geometry -- see world
        file's <model name="bin"> pose/walls)."""
        if self._scene_initialized:
            return
        self._add_box("table", [0.0, 0.0, 0.375], [1.9, 2.6, 0.75])  # X,Y grown to match world file's enlarged table, 2026-09-03
        self._add_box("bin", [0.415, 1.158, 0.85], [0.21, 0.21, 0.10])  # XY matches world file's bin pose (Y-offset bug fixed), 2026-09-03
        self._scene_initialized = True
        time.sleep(0.5)

    def _add_box(self, name, center_xyz, size_xyz):
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size_xyz)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = center_xyz
        pose.orientation.w = 1.0
        co.primitives.append(primitive)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        self.collision_pub.publish(co)

    def _build_goal(self, tool0_pos, frame_id, vel_scale, accel_scale, position_box,
                     orientation_tolerance, allowed_planning_time, num_planning_attempts):
        q = [0.0, 1.0, 0.0, 0.0]  # top-down tool0 orientation, same as Phase 1

        pcm = PositionConstraint()
        pcm.header.frame_id = frame_id
        pcm.link_name = "tool0"
        pcm.weight = 1.0
        box = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [position_box, position_box, position_box]
        box.primitives.append(primitive)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = tool0_pos
        pose.orientation.w = 1.0
        box.primitive_poses.append(pose)
        pcm.constraint_region = box

        ocm = OrientationConstraint()
        ocm.header.frame_id = frame_id
        ocm.link_name = "tool0"
        ocm.orientation.x, ocm.orientation.y, ocm.orientation.z, ocm.orientation.w = q
        ocm.absolute_x_axis_tolerance = orientation_tolerance
        ocm.absolute_y_axis_tolerance = orientation_tolerance
        ocm.absolute_z_axis_tolerance = orientation_tolerance
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)

        goal = MoveGroup.Goal()
        goal.request.group_name = "arm"
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnectkConfigDefault"
        goal.request.num_planning_attempts = num_planning_attempts
        goal.request.allowed_planning_time = allowed_planning_time
        goal.request.max_velocity_scaling_factor = vel_scale
        goal.request.max_acceleration_scaling_factor = accel_scale
        goal.request.goal_constraints = [constraints]

        joint_state = self._get_joint_state()
        if joint_state is not None:
            goal.request.start_state.joint_state = joint_state
            goal.request.start_state.is_diff = False
        else:
            goal.request.start_state.is_diff = True
        return goal

    def _execute_move(self, goal):
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("MoveGroup action server unavailable for handoff!")
            return False
        future = self._move_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)
        handle = future.result()
        if not handle.accepted:
            self.node.get_logger().warn("Handoff goal rejected by MoveGroup planner.")
            return False
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)
        result = res_future.result().result
        return result.error_code.val == 1

    def _verify(self, target_xyz, frame_id, tol):
        try:
            t = self.tf_buffer.lookup_transform(
                frame_id, 'tool0', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
        except Exception:
            return False
        actual = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        return max(abs(actual[i] - target_xyz[i]) for i in range(3)) <= tol

    def _handoff_to_pose(self, target_xyz, frame_id, attempts, vel_scale, accel_scale,
                          position_box, orientation_tolerance, allowed_planning_time,
                          num_planning_attempts, verify_tol):
        """Shared retry loop behind handoff_to_pregrasp: build a
        goal, execute it, verify tool0 actually arrived via ground-truth TF (never trust a
        reported MoveGroup/ExecuteTrajectory success alone -- same class of false-success
        bug documented throughout phase1_tf_pick_and_place.py's verify_* methods), retry on
        failure. Returns (ok, target_xyz) -- target_xyz is always returned so the caller can
        log the pose/distance per episode regardless of outcome (see spec section 2)."""
        self.ensure_scene()
        for attempt in range(1, attempts + 1):
            goal = self._build_goal(
                target_xyz, frame_id, vel_scale=vel_scale, accel_scale=accel_scale,
                position_box=position_box, orientation_tolerance=orientation_tolerance,
                allowed_planning_time=allowed_planning_time,
                num_planning_attempts=num_planning_attempts
            )
            if self._execute_move(goal):
                time.sleep(0.3)
                if self._verify(target_xyz, frame_id, tol=verify_tol):
                    return True, target_xyz
            self.node.get_logger().warn(f"Handoff attempt {attempt}/{attempts} failed or unverified -- retrying.")
        return False, target_xyz

    def handoff_to_pregrasp(self, target_xyz_base_link, attempts=3, standoff_m=None):
        """Drives tool0 to the pre-grasp standoff above target_xyz_base_link (given in
        base_link frame, matching UR3eEnv._update_target_pose's frame) via MoveIt/OMPL.
        See _handoff_to_pose for the retry/verify mechanics.

        standoff_m overrides the instance default HANDOFF_STANDOFF_M when given -- lets the
        caller (UR3eEnv.reset(), see its curriculum_level-derived standoff) vary how far
        above the target RL's episode actually starts, instead of the fixed 5cm this project
        used for its entire history through 2026-09-04. See project_phase2_rl_status memory
        / the curriculum-widening implementation spec for why this was previously a no-op:
        curriculum_level was computed correctly but never consumed anywhere."""
        standoff = standoff_m if standoff_m is not None else self.HANDOFF_STANDOFF_M
        pregrasp = [
            float(target_xyz_base_link[0]),
            float(target_xyz_base_link[1]),
            float(target_xyz_base_link[2]) + self.GRIPPER_OFFSET_M + standoff,
        ]
        return self._handoff_to_pose(
            pregrasp, "base_link", attempts, vel_scale=0.3, accel_scale=0.3,
            position_box=0.02, orientation_tolerance=0.05,
            allowed_planning_time=10.0, num_planning_attempts=30, verify_tol=0.025
        )
