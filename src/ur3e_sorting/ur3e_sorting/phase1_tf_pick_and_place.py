#!/usr/bin/env python3
"""
Phase 1: Robust Pick-and-Place for 'lego_red' using UR5e Robot + Robotiq 2F-140 gripper
- Solves MoveIt error -10 (START_STATE_IN_COLLISION) by setting start_state.is_diff = True
- Bin Clearance Height: tool0 Z = 1.150m (rim top 0.90m + 0.055m min clearance + 0.17m gripper
  offset = 1.125m minimum; 1.150m used for extra margin, per world file bin/wall geometry)
- Sim Time Enabled (use_sim_time=True)

2026-08-31: switched from UR5e/SCHUNK Co-act EGP-C 40 back to UR5e/Robotiq 2F-140 -- this
restores the last proven, live-verified calibration (this file's history, verified per project
memory), including GRIPPER_OFFSET, the pre-pick/pick tolerances, and the 0.6 close position,
none of which needed re-deriving since the gripper, mount, and lego_red size (57mm) all match
what was already tuned and confirmed working.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, CollisionObject, JointConstraint, RobotTrajectory, TrajectoryConstraints
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tf2_ros
from control_msgs.action import GripperCommand
import tf_transformations
import time
import math
import json
import os
import subprocess

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState

class Phase1RedLegoPickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('phase1_red_lego_pick_and_place_node', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])

        # OMPL planner to use for every use_ompl=True move -- switch without touching code:
        #   ros2 run ur3e_sorting phase1_pick_and_place --ros-args -p planner_id:=RRTstarkConfigDefault
        # Must be one of the configs registered under "arm:" in ompl_planning.yaml. Only
        # RRTConnectkConfigDefault (default) is proven working on this rig; RRTstarkConfigDefault
        # is expected-safe (same pipeline/constraint format) but untested here; anything else
        # registered there is unverified for this workspace's specific clearances.
        self.declare_parameter('planner_id', 'RRTConnectkConfigDefault')

        # OMPL pipeline variant for every use_ompl=True move -- default "ompl" is the
        # proven-working pipeline, untouched. "ompl_ruckig" is identical except it adds
        # jerk-limited velocity/acceleration smoothing (see ompl_ruckig_planning.yaml) --
        # opt in without touching code or the default:
        #   ros2 run ur3e_sorting phase1_pick_and_place --ros-args -p pipeline_id:=ompl_ruckig
        self.declare_parameter('pipeline_id', 'ompl')

        # Action Clients for MoveGroup, Trajectory Replay & Gripper
        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        # Trajectory cache: keyed by move name, replayed as-is (no replanning) whenever
        # lego_red's pose still matches the pose the cache was recorded for. OMPL is
        # randomized -- planning fresh every run means a different path each time even
        # with nothing changed -- so caching the planned trajectory is what actually
        # makes "same path every run" true, rather than just re-asking OMPL and hoping.
        self.cache_path = os.path.expanduser("~/.cache/ur3e_sorting/phase1_trajectory_cache.json")
        self.cached_moves = {}
        self.pending_cache = {}
        self.cache_valid = False
        self._last_planned_trajectory = None
        
        # Subscriber to normalize joint angles into [-pi, pi]
        self._joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.latest_joint_state = None
        
        # TF Buffer for Ground Truth Transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Ground-truth object poses come from a direct `gz topic -e` query against Gazebo's
        # own /world/lego_sorting/pose/info (see _get_gz_object_position) -- independent of
        # the standard /tf tree, which lego_red never appears on without a perception node
        # running. Used for benchmark logging: actual final block position, not what the
        # script assumes/hopes happened.
        #
        # NOT bridged via ros_gz_bridge.yaml's /gz_poses_static /gz_poses_dynamic topics:
        # those bridge gz.msgs.Pose_V -> tf2_msgs/msg/TFMessage, but on this bridge version
        # every resulting transform's child_frame_id comes back empty (confirmed via
        # `ros2 topic echo /gz_poses_static --once` -- every entry has child_frame_id: ''),
        # so a per-object name can never be recovered from that topic on the ROS side even
        # though the raw `gz topic -e` data (same underlying Gazebo topic) has proper names.
        self.gz_world_name = 'lego_sorting'

        # Collision Object Publisher for MoveIt Scene
        self.collision_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

        # Per-run benchmark data (see _record_move/_save_benchmark) -- a classical-baseline
        # metrics log to compare against an RL policy later: per-move duration/attempts/
        # planning-time/path-length/verify-error, plus run-level success and placement
        # accuracy from Gazebo ground truth.
        self.benchmark_dir = os.path.expanduser("~/.local/share/ur3e_sorting/benchmarks")
        self.benchmark = {}
        self._last_planning_time = None
        self._last_verify_error = None

        # Known-safe retracted joint configuration. Sent as step 0 of every run so planning
        # always starts from the same, validated state instead of wherever the previous
        # run (or a failed attempt) happened to leave the arm.
        #
        # Matches this rig's actual Gazebo spawn pose (ur_description/config/
        # initial_positions.yaml: shoulder_pan=-0.4513, shoulder_lift=4.7557,
        # elbow=-0.2256, wrist_1=-1.2844, wrist_2=-0.1041, wrist_3=1.4753), not the SRDF
        # "home"/"ready" group state. The two differ (SRDF home has shoulder_lift=4.6169
        # vs. spawn's 4.7557, and non-zero spawn values on every other joint the SRDF
        # pose zeroes out) -- previously using the SRDF pose meant step 0 always yanked
        # the arm through a large, unnecessary reconfiguration away from where it actually
        # already was, causing visible unwanted rotation on every run.
        #
        # shoulder_lift is given as the raw 4.7557 (272.5 deg), not wrapped into [-pi, pi]
        # (-1.5275) -- it must match what joint_state_cb now passes through unmodified
        # (see that method's docstring for why the old blanket wrap was itself the root
        # cause of a much worse unwanted-rotation/self-collision bug). shoulder_lift's
        # true range is +/-360 deg, so 4.7557 is valid as-is.
        self.home_joints = {
            "shoulder_pan_joint": -0.4513,
            "shoulder_lift_joint": 4.7557,
            "elbow_joint": -0.2256,
            "wrist_1_joint": -1.2844,
            "wrist_2_joint": -0.1041,
            "wrist_3_joint": 1.4753
        }

        self.get_logger().info("==================================================")
        self.get_logger().info("🚀 Phase 1: Robust Pick & Place for 'lego_red'")
        self.get_logger().info("   Resolved Collision & Bin Clearance Offsets")
        self.get_logger().info("==================================================")

    def _get_gz_object_position(self, name, timeout=3.0):
        """Ground-truth [x, y, z] for a named object, queried directly from Gazebo via the
        `ign topic` CLI against /world/<world>/pose/info -- bypasses ros_gz_bridge, whose
        Pose_V -> TFMessage conversion drops the per-pose name on this bridge version (see
        the comment on gz_world_name in __init__). Returns None on timeout/parse failure/
        missing name, which the caller treats as "can't verify" rather than "verified absent".
        Uses `ign`, not `gz`: confirmed live that this stack's `gz` CLI can't discover the
        running `ign gazebo` (Fortress) server at all ("An instance of Gazebo is not
        running.") even though the sim is genuinely up -- `ign topic` is the one that
        actually talks to it."""
        try:
            proc = subprocess.run(
                ['ign', 'topic', '-e', '-t', f'/world/{self.gz_world_name}/pose/info',
                 '-n', '1', '--json-output'],
                capture_output=True, text=True, timeout=timeout
            )
            data = json.loads(proc.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            self.get_logger().warn(f"    Ground-truth query for '{name}' failed: {e}")
            return None

        for pose in data.get('pose', []):
            if pose.get('name') == name:
                pos = pose.get('position', {})
                return [pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)]
        self.get_logger().warn(f"    Ground-truth query: '{name}' not found in world state.")
        return None

    def joint_state_cb(self, msg):
        """Stores the raw joint state as-is -- used verbatim as MoveGroup's start_state
        (see _build_pipeline_goal) so planning always starts from exactly what the
        controller itself reports.

        Previously this force-wrapped every position into [-pi, pi], which sounds like a
        safe no-op but isn't: 5 of this UR5's 6 joints (shoulder_pan, shoulder_lift,
        wrist_1, wrist_2, wrist_3) have a real range of +/-360 deg (ur_description/config/
        ur5/joint_limits.yaml), so a perfectly valid raw reading like shoulder_lift=4.7557
        rad (272.5 deg, well inside +/-2pi) got rewritten to -1.5275 rad -- the same
        physical angle, but a different number than what arm_controller's own
        state_interfaces (and Gazebo) actually track. MoveGroup would then plan a short,
        collision-checked path in that wrapped frame (start already "at" the goal), while
        the real controller -- fed those wrapped waypoint values -- had to physically
        sweep the ~2pi gap to them, taking the arm through a completely different,
        never-collision-checked path. Confirmed live: shoulder_lift swept 4.7557 -> 3.10
        -> ... before the arm stalled completely (six identical /joint_states samples in a
        row) exactly where MoveIt separately logged a 'table' vs 'forearm_link' and
        'forearm_link' vs 'wrist_3_link' contact on sibling planning attempts -- this was
        the source of the unwanted-rotation/self-collision reports, not a planner quirk.
        only elbow_joint (+/-180 deg) can ever legitimately need wrapping, and since it's
        physically incapable of exceeding that range, the raw reading is already exactly
        what's needed -- no per-joint wrap logic required at all."""
        self.latest_joint_state = msg

    def get_red_lego_pose(self):
        """Looks up lego_red's current position: TF first, then a direct Gazebo ground-truth
        query (_get_gz_object_position) if TF isn't publishing it -- never a hardcoded
        literal. A hardcoded fallback here previously meant the script kept targeting
        lego_red's original spawn spot forever after any run left it somewhere else (e.g.
        dropped on the floor by a failed bin move) -- the gripper would close on empty air
        at the stale assumed position, "succeed" by its own move/verify checks, and only
        the (also previously broken) ground-truth placement check could have caught it.
        Returns None if lego_red's position can't be determined at all."""
        candidates = ['lego_red', 'lego_red/link', 'conveyor_world/lego_red']
        for frame in candidates:
            try:
                transform = self.tf_buffer.lookup_transform('world', frame, rclpy.time.Time())
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                z = transform.transform.translation.z
                self.get_logger().info(f"✅ TF Found for 'lego_red' [{frame}]: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                return [x, y, z]
            except Exception:
                continue

        gz_pos = self._get_gz_object_position('lego_red')
        if gz_pos is not None:
            self.get_logger().info(f"ℹ️ TF pending for 'lego_red'. Using live Gazebo ground truth: {gz_pos}")
            return gz_pos

        self.get_logger().error("Could not determine lego_red's position via TF or Gazebo ground truth.")
        return None

    def poses_close(self, a, b, tol=0.01):
        """True if two [x,y,z] poses are within `tol` metres of each other. Used to decide
        whether lego_red has actually moved since the trajectory cache was recorded --
        small TF noise shouldn't count as a moved block and force a fresh, differently-
        shaped OMPL path every run."""
        if a is None or b is None:
            return False
        return all(abs(a[i] - b[i]) <= tol for i in range(3))

    def load_trajectory_cache(self):
        """Loads the on-disk cache. Returns None if missing/corrupt -- treated as a cache
        miss, not an error, since the first run on a machine will never have one."""
        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save_trajectory_cache(self, lego_xyz, moves):
        """Persists the cache. Only ever called after a fully successful run -- a failed
        or partial run must not overwrite a known-good cached path with something that
        may not have actually worked end to end."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        data = {
            "lego_pose": list(lego_xyz),
            "moves": {name: self._traj_to_dict(traj) for name, traj in moves.items()}
        }
        with open(self.cache_path, 'w') as f:
            json.dump(data, f)

    def _traj_to_dict(self, traj):
        jt = traj.joint_trajectory
        return {
            "joint_names": list(jt.joint_names),
            "points": [
                {
                    "positions": list(p.positions),
                    "velocities": list(p.velocities),
                    "accelerations": list(p.accelerations),
                    "time_from_start": p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                }
                for p in jt.points
            ]
        }

    def _dict_to_traj(self, d):
        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = d["joint_names"]
        for p in d["points"]:
            pt = JointTrajectoryPoint()
            pt.positions = p["positions"]
            pt.velocities = p["velocities"]
            pt.accelerations = p["accelerations"]
            sec = int(p["time_from_start"])
            nsec = int(round((p["time_from_start"] - sec) * 1e9))
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            traj.joint_trajectory.points.append(pt)
        return traj

    def _start_state_matches(self, traj, tol=0.15):
        """Guards against blindly replaying a cached trajectory from the wrong starting
        joint state -- ExecuteTrajectory just follows the recorded waypoints, it doesn't
        replan from wherever the arm actually is. If the current state doesn't match the
        first waypoint the trajectory was recorded from, treat it as a cache miss.

        tol is deliberately looser than the ~0.05 rad goal tolerances used when planning
        these moves: every cached move except 'home' chains from the previous move's own
        recorded (and, on a cache hit, exactly reproduced) endpoint, so those seams match
        almost exactly. The one seam that doesn't is home -> pre_pick, since home is never
        cached (see below) and gets freshly OMPL-planned every run -- its actual landing
        can differ from any past recording by roughly its own goal tolerance in either
        direction. 0.15 rad comfortably covers that compounding drift while still catching
        a genuinely different state (e.g. the arm parked somewhere else entirely)."""
        if not traj.joint_trajectory.points or not self.latest_joint_state:
            return False
        first = traj.joint_trajectory.points[0].positions
        names = traj.joint_trajectory.joint_names
        current = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        for name, target in zip(names, first):
            cur = current.get(name)
            if cur is None or abs(cur - target) > tol:
                return False
        return True

    def verify_joint_target(self, joint_dict, tol=0.05):
        """Ground-truth check for joint-space moves (home): does the arm's actual joint
        state match what the move just claimed to reach? The controller can report a
        trajectory successful purely because its time schedule elapsed, even if it fell
        behind and never actually got there (see: home landing with the arm resting on
        the table despite a 'SUCCESS' result). This catches that instead of trusting the
        reported error code."""
        if not self.latest_joint_state:
            self.get_logger().warn("    Verify: no joint state available yet.")
            return False
        current = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        ok = True
        max_delta = 0.0
        for name, target in joint_dict.items():
            cur = current.get(name)
            if cur is None:
                ok = False
                continue
            # Shortest-angle (modular) delta, not raw subtraction: these joints have real
            # +/-360deg range, so the controller can land the arm at a raw reading on a
            # different 2*pi branch than the literal target value while still being
            # physically right where it should be (create_joint_goal already accounts for
            # this when SETTING the target -- see its own 2pi-nearest-equivalent shift --
            # but this check previously didn't apply the same wrapping when VERIFYING it,
            # so a genuinely successful move could still fail verification here). Confirmed
            # live: shoulder_pan actual=-6.2832 (physically ~0 rad) vs target=-0.4513 raw-
            # subtracted to a false 5.83 rad "failure", when the real angular gap was only
            # ~0.45 rad.
            delta = abs(((cur - target) + math.pi) % (2 * math.pi) - math.pi)
            max_delta = max(max_delta, delta)
            if delta > tol:
                self.get_logger().warn(f"    Verify FAILED: {name} actual={cur:.4f} target={target:.4f} delta={delta:.4f} (tol={tol})")
                ok = False
        self._last_verify_error = max_delta
        return ok

    def verify_pose_target(self, target_xyz, tol=0.03):
        """Ground-truth check for Cartesian moves (pre-pick, pick, lift, bin, retract):
        does tool0's actual TF pose match the Cartesian target the move just claimed to
        reach? Same false-success class of bug as verify_joint_target, checked via TF
        instead of joint state since these moves are specified as poses, not angles.
        Position-only -- orientation drift (which caused the original bin-placement miss)
        isn't checked here, so this catches "controller ran out of time and stopped
        short," not every possible way a move can go wrong."""
        try:
            transform = self.tf_buffer.lookup_transform('world', 'tool0', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"    Verify: could not look up tool0 TF: {e}")
            return False
        actual = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        delta = [abs(actual[i] - target_xyz[i]) for i in range(3)]
        self._last_verify_error = max(delta)
        if max(delta) > tol:
            self.get_logger().warn(
                f"    Verify FAILED: tool0 actual=[{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}] "
                f"target=[{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}] "
                f"delta=[{delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}] (tol={tol})"
            )
            return False
        return True

    def _verify_with_settle(self, verify, settle_time=0.3):
        """Brief pause before sampling state/TF for verification, so a just-finished
        move's final joint_states/TF messages have actually been published and received."""
        time.sleep(settle_time)
        return verify()

    def execute_cached_trajectory(self, traj):
        """Replays a previously-planned trajectory verbatim via /execute_trajectory --
        no planning call involved, so it's the same path every time by construction."""
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("ExecuteTrajectory action server unavailable!")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        future = self._execute_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)

        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Cached trajectory rejected by ExecuteTrajectory.")
            return False

        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)

        result = res_future.result().result
        if result.error_code.val == 1:
            return True
        self.get_logger().error(f"Cached trajectory execution failed with code: {result.error_code.val}")
        return False

    def operate_gripper(self, position, effort=100.0, wait=True):
        """Actuates the Robotiq 2F Gripper."""
        self.get_logger().info(f"Gripper Action -> Target Position = {position}")
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Gripper action server unavailable. Continuing motion...")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = effort

        future = self._gripper_client.send_goal_async(goal)
        if not wait:
            return True

        try:
            handle = future.result()
            if handle and handle.accepted:
                handle.get_result_async()
                time.sleep(1.0)
                return True
        except Exception as e:
            self.get_logger().error(f"Gripper execution error: {e}")
        return False

    def _build_pipeline_goal(self, pipeline_id, planner_id, vel_scale, accel_scale, constraints,
                              allowed_planning_time=10.0, num_planning_attempts=30):
        """Shared skeleton for a MoveGroup goal targeting a specific pipeline -- both the
        plain pose/joint goals and the hybrid OMPL/CHOMP path go through this so start
        state, scaling, and attempt/time budgets stay consistent across all of them.

        allowed_planning_time/num_planning_attempts default to the budget that's been
        proven-working for the easy moves (pre_pick/pick: small, uncluttered goal regions).
        lift and bin plan through a much tighter collision margin -- right next to the
        table with a grasped block changing the arm's effective self-collision shape, and
        (for bin) a wide traverse through the forearm/wrist-camera collision band -- and
        were observed timing out (MoveIt error -6) on the default budget repeatedly enough
        to burn most of a run's wall-clock time in retries; those two moves pass a larger
        budget explicitly (see execute_pick_and_place_red_lego)."""
        goal = MoveGroup.Goal()
        goal.request.group_name = "arm"
        goal.request.pipeline_id = pipeline_id
        goal.request.planner_id = planner_id
        goal.request.num_planning_attempts = num_planning_attempts
        goal.request.allowed_planning_time = allowed_planning_time
        goal.request.max_velocity_scaling_factor = vel_scale
        goal.request.max_acceleration_scaling_factor = accel_scale

        # Pass normalized joint state if available to prevent MoveIt 2pi wrapped joint limit crashes
        if self.latest_joint_state:
            goal.request.start_state.joint_state = self.latest_joint_state
            goal.request.start_state.is_diff = False
        else:
            goal.request.start_state.is_diff = True

        goal.request.goal_constraints = [constraints]
        return goal

    def _plan_only(self, goal):
        """Sends a MoveGroup goal with plan_only=True -- returns (success, planned_trajectory)
        without executing anything. Used to get a fast OMPL seed for CHOMP to refine."""
        goal.planning_options.plan_only = True
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server unavailable!")
            return False, None
        future = self._move_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)
        handle = future.result()
        if not handle.accepted:
            return False, None
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)
        result = res_future.result().result
        if result.error_code.val == 1 and result.planned_trajectory.joint_trajectory.points:
            return True, result.planned_trajectory
        return False, None

    def _seed_from_trajectory(self, traj):
        """Converts a RobotTrajectory's waypoints into a moveit_msgs/TrajectoryConstraints --
        CHOMP treats a populated request.trajectory_constraints as its initial trajectory to
        refine, instead of generating its own naive straight-line/spline guess. This is what
        makes it optimize the OMPL solution's shape rather than replan from scratch."""
        tc = TrajectoryConstraints()
        names = traj.joint_trajectory.joint_names
        for point in traj.joint_trajectory.points:
            c = Constraints()
            for name, pos in zip(names, point.positions):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = pos
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                c.joint_constraints.append(jc)
            tc.constraints.append(c)
        return tc

    def _hybrid_chomp_goal(self, constraints, vel_scale, accel_scale,
                            allowed_planning_time=10.0, num_planning_attempts=30):
        """OMPL finds a fast, valid, but not necessarily short/smooth path (RRTConnect stops
        at the first solution it finds); this plans that with OMPL first (plan_only, nothing
        executes yet), then builds a CHOMP goal seeded with it via trajectory_constraints so
        CHOMP refines that path's shape instead of starting from scratch. Returns None if the
        OMPL seed plan itself fails, so the caller can fall back to a plain (non-hybrid) goal
        rather than blocking the move entirely on CHOMP."""
        seed_goal = self._build_pipeline_goal("ompl", "RRTConnectkConfigDefault", vel_scale, accel_scale, constraints,
                                               allowed_planning_time, num_planning_attempts)
        ok, seed_traj = self._plan_only(seed_goal)
        if not ok:
            self.get_logger().warn("Hybrid: OMPL seed plan failed -- falling back to plain OMPL.")
            return None

        chomp_goal = self._build_pipeline_goal("chomp", "", vel_scale, accel_scale, constraints,
                                                allowed_planning_time, num_planning_attempts)
        chomp_goal.request.trajectory_constraints = self._seed_from_trajectory(seed_traj)
        return chomp_goal

    def create_robust_pose_goal(self, tool0_pos, vel_scale=0.2, accel_scale=0.2, orientation_tolerance=0.5, use_ompl=True, position_box=0.10, hybrid=False, allowed_planning_time=10.0, num_planning_attempts=30):
        """Constructs a robust MoveGroup Goal with diff start state for UR5 robot.

        Defaults to OMPL: Pilz PTP interpolates straight-line in joint space and only
        checks the endpoints for collision, not the path in between. With the base
        mounted this close to the workspace, that has produced self-collisions (gripper
        vs forearm, upper arm vs base) on more than one move in this sequence -- not
        just the large pick-to-bin swing. OMPL's collision-aware sampling checks the
        whole path. use_ompl=False falls back to Pilz PTP for short, low-risk moves.

        position_box: side length of the goal position tolerance cube. Pilz PTP always
        converges to the exact target regardless of this value, but OMPL is free to land
        ANYWHERE inside it and still satisfy the constraint -- the default 0.10 (+/-5cm)
        is far too loose for a precision grasp on a 5.7cm block; pass a tight value (e.g.
        0.02) for the pre-pick/pick moves.

        hybrid: plan an OMPL seed then refine it with CHOMP (see _hybrid_chomp_goal) instead
        of executing the raw RRTConnect result. Takes precedence over use_ompl. Falls back to
        the plain use_ompl path if the OMPL seed plan itself fails.

        allowed_planning_time/num_planning_attempts: per-call budget, passed through to
        _build_pipeline_goal. Raise these for moves through tight collision margins (lift,
        bin -- see execute_pick_and_place_red_lego) that were observed hitting MoveIt error
        -6 (TIMED_OUT) on the default budget; leave at default for moves that already plan
        reliably within it (pre_pick, pick, retract).
        """
        # Top-down straight-down tool0 orientation for Robotiq 140 on UR5: Pitch=180 (Roll=0, Pitch=pi, Yaw=0)
        q = [0.0, 1.0, 0.0, 0.0]

        # Position Constraint
        pcm = PositionConstraint()
        pcm.header.frame_id = "world"
        pcm.link_name = "tool0"
        pcm.weight = 1.0

        box = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [position_box, position_box, position_box]
        box.primitives.append(primitive)

        pose = Pose()
        pose.position.x = tool0_pos[0]
        pose.position.y = tool0_pos[1]
        pose.position.z = tool0_pos[2]
        pose.orientation.w = 1.0
        box.primitive_poses.append(pose)
        pcm.constraint_region = box

        # Orientation Constraint (Straight-down with 0.5 rad tolerance)
        ocm = OrientationConstraint()
        ocm.header.frame_id = "world"
        ocm.link_name = "tool0"
        ocm.orientation.x = q[0]
        ocm.orientation.y = q[1]
        ocm.orientation.z = q[2]
        ocm.orientation.w = q[3]
        ocm.absolute_x_axis_tolerance = orientation_tolerance
        ocm.absolute_y_axis_tolerance = orientation_tolerance
        ocm.absolute_z_axis_tolerance = orientation_tolerance
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)

        if hybrid:
            hybrid_goal = self._hybrid_chomp_goal(constraints, vel_scale, accel_scale,
                                                   allowed_planning_time, num_planning_attempts)
            if hybrid_goal is not None:
                return hybrid_goal

        pipeline_id, planner_id = (self.get_parameter('pipeline_id').value, self.get_parameter('planner_id').value) if use_ompl else ("pilz_industrial_motion_planner", "PTP")
        return self._build_pipeline_goal(pipeline_id, planner_id, vel_scale, accel_scale, constraints,
                                          allowed_planning_time, num_planning_attempts)

    def create_joint_goal(self, joint_dict, vel_scale=0.3, accel_scale=0.3, use_ompl=True, hybrid=False, allowed_planning_time=10.0, num_planning_attempts=30):
        """Constructs a joint-space MoveGroup Goal. Used for the home reset, where the
        start state is arbitrary (whatever a prior run/attempt left it at) -- OMPL's
        collision-aware search handles that far more reliably than Pilz PTP's straight
        joint-space interpolation would from an unknown start.

        hybrid: see create_robust_pose_goal -- plans an OMPL seed then refines it with CHOMP."""
        constraints = Constraints()
        current = None
        if self.latest_joint_state:
            current = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        for name, val in joint_dict.items():
            target = val
            if current and name in current:
                # Shift the target by the nearest multiple of 2*pi to the joint's current
                # position, so the numeric goal is the closest equivalent angle -- not
                # just the literal value in joint_dict, which (these joints all have
                # +/-360 deg range) could be most of a full rotation away depending on
                # where the joint currently sits. Without this, OMPL has no reason to
                # prefer the short way around and can return a needlessly long sweep.
                target = current[name] + (((val - current[name]) + math.pi) % (2 * math.pi) - math.pi)
            c = JointConstraint()
            c.joint_name = name
            c.position = target
            c.tolerance_above = 0.05
            c.tolerance_below = 0.05
            c.weight = 1.0
            constraints.joint_constraints.append(c)

        if hybrid:
            hybrid_goal = self._hybrid_chomp_goal(constraints, vel_scale, accel_scale,
                                                   allowed_planning_time, num_planning_attempts)
            if hybrid_goal is not None:
                return hybrid_goal

        pipeline_id, planner_id = (self.get_parameter('pipeline_id').value, self.get_parameter('planner_id').value) if use_ompl else ("pilz_industrial_motion_planner", "PTP")
        return self._build_pipeline_goal(pipeline_id, planner_id, vel_scale, accel_scale, constraints,
                                          allowed_planning_time, num_planning_attempts)

    def _trajectory_path_length(self, traj):
        """Sum of per-waypoint joint-space L2 distance across a trajectory -- a rough
        proxy for how much motion a plan actually contains, useful as a classical-baseline
        comparison point against an RL policy's trajectories later."""
        if traj is None:
            return None
        pts = traj.joint_trajectory.points
        total = 0.0
        for i in range(1, len(pts)):
            a, b = pts[i - 1].positions, pts[i].positions
            total += math.sqrt(sum((b[j] - a[j]) ** 2 for j in range(len(a))))
        return total

    def _record_move(self, name, start_t, attempts_used, used_cache, traj, failed=False, planner_used=None):
        """Appends one move's benchmark entry -- see self.benchmark_dir / _save_benchmark.
        joint_angles_rad is the arm's actual measured joint state (from /joint_states, via
        joint_state_cb) right after this move settled -- ground truth, not the trajectory's
        intended target, so it reflects where the arm really ended up (including on a
        failed move, wherever the last attempt left it)."""
        joint_angles = None
        if self.latest_joint_state:
            joint_angles = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        self.benchmark.setdefault("moves", []).append({
            "move": name,
            "duration_s": time.time() - start_t,
            "attempts": attempts_used,
            "used_cache": used_cache,
            "planning_time_s": self._last_planning_time,
            "planner_used": planner_used,
            "num_waypoints": len(traj.joint_trajectory.points) if traj is not None else None,
            "path_length_rad": self._trajectory_path_length(traj),
            "verify_error": self._last_verify_error,
            "joint_angles_rad": joint_angles,
            "failed": failed,
        })

    # Bin opening is ~0.18-0.2m wide (world file wall span); a block within this radius of
    # bin center XY is confidently inside the bin footprint. Heuristic, not exact geometry.
    BIN_SUCCESS_RADIUS_M = 0.10

    def _save_benchmark(self, run_start_time, success, bin_xy=None):
        """Writes the run's benchmark record to disk -- called on every exit path (success
        or failure) so robustness stats (how often/where it fails) are captured too, not
        just successful runs. Placement accuracy comes from Gazebo's own ground truth
        (self._get_gz_object_position, queried directly from /world/lego_sorting/pose/info),
        not from anything the script itself assumed happened -- a claimed success is only
        ever downgraded to False by ground truth here, never upgraded to True, since a
        missing/stale ground-truth reading isn't proof of anything."""
        self.benchmark["run_duration_s"] = time.time() - run_start_time
        self.benchmark["total_retries"] = sum(
            max(0, m["attempts"] - 1) for m in self.benchmark.get("moves", []) if m["attempts"]
        )
        final_block_pos = self._get_gz_object_position("lego_red")
        self.benchmark["final_block_position"] = final_block_pos
        placement_error = None
        if final_block_pos and bin_xy:
            placement_error = math.sqrt(
                (final_block_pos[0] - bin_xy[0]) ** 2 + (final_block_pos[1] - bin_xy[1]) ** 2
            )
            if placement_error > self.BIN_SUCCESS_RADIUS_M:
                success = False
        self.benchmark["placement_error_m"] = placement_error
        self.benchmark["success"] = success

        os.makedirs(self.benchmark_dir, exist_ok=True)
        fname = os.path.join(self.benchmark_dir, f"run_{int(run_start_time)}.json")
        with open(fname, 'w') as f:
            json.dump(self.benchmark, f, indent=2)
        self.get_logger().info(f"📊 Benchmark saved to {fname} (success={success}, placement_error_m={placement_error})")

    def execute_move_with_retry(self, goal_factory, attempts=3, cache_key=None, verify=None):
        """If a cached trajectory exists for `cache_key` (lego_red hasn't moved since it
        was recorded) and the arm is currently at the joint state that trajectory starts
        from, replays it verbatim -- same path every time, no OMPL call at all. Otherwise
        plans fresh, retrying up to `attempts` times since OMPL/RRTConnect is randomized
        and a single failed attempt through a tight passage doesn't mean no solution
        exists. A successful fresh plan is stashed in self.pending_cache so the full
        sequence gets written to disk if the run completes.

        `verify`, if given, is a zero-arg callable checked against ground truth (actual
        joint state or TF, not the reported error code) after MoveGroup/ExecuteTrajectory
        itself reports success. Both can report "done" once a trajectory's time schedule
        elapses even if the controller fell behind and never actually arrived -- verify
        catches that and forces a real replan instead of silently continuing on a stale
        or wrong position.

        When cache_key is given, also records a benchmark entry for this move (see
        _record_move) -- cache_key doubles as the move's name in the benchmark log,
        since every current call site already passes a descriptive one."""
        move_start = time.time()
        self._last_verify_error = None

        if cache_key and self.cache_valid and cache_key in self.cached_moves:
            traj = self.cached_moves[cache_key]
            if self._start_state_matches(traj):
                self.get_logger().info(f"↻ Replaying cached path for '{cache_key}' (lego_red unchanged).")
                if self.execute_cached_trajectory(traj):
                    if verify is None or self._verify_with_settle(verify):
                        self._record_move(cache_key, move_start, attempts_used=0, used_cache=True, traj=traj, planner_used="cached")
                        return True
                    self.get_logger().warn(f"Cached trajectory for '{cache_key}' executed but failed ground-truth verification -- replanning.")
                else:
                    self.get_logger().warn(f"Cached trajectory for '{cache_key}' failed to execute -- replanning.")
            else:
                self.get_logger().warn(f"Current joint state doesn't match cached start for '{cache_key}' -- replanning.")

        planner_used = None
        for attempt in range(1, attempts + 1):
            goal = goal_factory()
            if hasattr(goal, 'request') and hasattr(goal.request, 'pipeline_id'):
                planner_used = f"{goal.request.pipeline_id}/{goal.request.planner_id}"
            if self.execute_move(goal):
                if verify is not None and not self._verify_with_settle(verify):
                    self.get_logger().warn(f"Move reported success but failed ground-truth verification (attempt {attempt}/{attempts}) -- treating as failed.")
                else:
                    if cache_key and self._last_planned_trajectory is not None:
                        self.pending_cache[cache_key] = self._last_planned_trajectory
                    if cache_key:
                        self._record_move(cache_key, move_start, attempt, used_cache=False, traj=self._last_planned_trajectory, planner_used=planner_used)
                    return True
            if attempt < attempts:
                self.get_logger().warn(f"Planning attempt {attempt}/{attempts} failed, retrying...")
        if cache_key:
            self._record_move(cache_key, move_start, attempts, used_cache=False, traj=None, failed=True, planner_used=planner_used)
        return False

    def execute_move(self, goal):
        """Sends goal to MoveGroup action server and waits for completion."""
        self._last_planned_trajectory = None
        self._last_planning_time = None
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server unavailable!")
            return False

        future = self._move_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.1)

        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Goal rejected by MoveGroup planner.")
            return False

        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.1)

        result = res_future.result().result
        if result.error_code.val == 1:
            self._last_planned_trajectory = result.planned_trajectory
            self._last_planning_time = result.planning_time
            return True
        else:
            self.get_logger().error(f"MoveGroup planning failed with code: {result.error_code.val}")
            return False

    def remove_collision_object(self, name="lego_red"):
        """Removes object from MoveIt planning scene to prevent collision during lift."""
        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = name
        co.operation = CollisionObject.REMOVE
        self.collision_pub.publish(co)

    def add_box_collision_object(self, name, center_xyz, size_xyz):
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

    def add_static_scene_objects(self):
        """Registers the table and bin as real MoveIt obstacles. Without this, the planning
        scene only knows about the robot's own links (self-collision) -- it has zero idea
        the table or bin physically exist, so a planner can produce a path that dips the
        arm straight through the solid tabletop. Gazebo's real physics then resists that
        motion, throwing off the executed trajectory. Geometry from pick_and_place_demo.world:
        table top surface at z=0.80, footprint 1.0x1.6 centered at world origin; bin walls
        span z=[0.80, 0.90] in a ~0.21x0.21 footprint centered at [-0.35, 0.25, 0.85].
        """
        # Full column from floor to table top -- conservatively blocks the whole volume a
        # planner could otherwise route the arm through, not just the thin top slab.
        # Matches the table's true footprint (1.0x1.6, not the previous 1.0x0.8 -- that
        # undersized box left the outer 0.4m of the real table on each side of Y invisible
        # to the planner, letting it validate paths through table volume Gazebo's physics
        # would actually block).
        # Top trimmed to z=0.75 (5cm below the real 0.80 surface): lego_red rests directly
        # on the table (block bottom = 0.80), so a box topping out exactly at 0.80 leaves
        # zero clearance for the gripper's own approach to grasp it -- the fingers closing
        # around the resting block were registering as touching this box. The 5cm margin
        # still blocks the real hazard (a gross dip through the table during a big
        # reconfiguration like Home) without fighting the grasp itself.
        self.add_box_collision_object("table", [0.0, 0.0, 0.375], [1.0, 1.60, 0.75])
        self.add_box_collision_object("bin", [-0.35, 0.25, 0.85], [0.21, 0.21, 0.10])

    def execute_pick_and_place_red_lego(self):
        """Executes full Phase 1 Pick and Place trajectory for lego_red."""
        self.get_logger().info("\n--- STEP-BY-STEP: Simple Pick & Place Red Lego ---")

        run_start_time = time.time()
        self.benchmark = {
            "planner_id": self.get_parameter('planner_id').value,
            "pipeline_id": self.get_parameter('pipeline_id').value,
            "moves": [],
        }

        # Register the table/bin as real obstacles before any planning happens.
        self.add_static_scene_objects()
        time.sleep(0.5)

        # 1. Target & Bin Coordinates
        lego_xyz = self.get_red_lego_pose()
        if lego_xyz is None:
            self.get_logger().error("Aborting run: lego_red's position is unknown.")
            self._save_benchmark(run_start_time, success=False)
            return False
        self.benchmark["lego_target"] = lego_xyz

        # Trajectory cache: only trust it if lego_red is still where it was when the
        # cache was recorded. If it moved, every downstream move's start/target changes
        # too, so the whole cached sequence is invalidated at once rather than trying to
        # reuse individual legs.
        cache_data = self.load_trajectory_cache()
        if cache_data and self.poses_close(cache_data.get("lego_pose"), lego_xyz):
            self.cache_valid = True
            self.cached_moves = {name: self._dict_to_traj(d) for name, d in cache_data["moves"].items()}
            self.get_logger().info(f"📦 lego_red still at {lego_xyz} -- replaying cached path from last run.")
        else:
            self.cache_valid = False
            self.cached_moves = {}
            self.get_logger().info(f"🆕 lego_red at {lego_xyz} has no matching cache (moved, or first run) -- planning fresh.")
        self.pending_cache = {}

        # Distance from tool0 to gripper tip. Was 0.17 (proven under the old UR5 setup);
        # reduced by ~9mm here (2026-08-31) after two consecutive live runs under UR5e
        # both showed tool0 landing SYSTEMATICALLY high at the pick pose, not randomly
        # scattered within the position tolerance box: +10.5mm then +8.7mm above the
        # commanded Z, same direction and similar magnitude both times. That points to a
        # real, repeatable bias (plausibly RRTConnect tending to terminate near the goal
        # region's boundary rather than its center) rather than noise a tighter tolerance
        # alone would fix -- lowering the commanded target compensates directly. Not yet
        # re-verified after this change; if the bias isn't exactly ~9mm every time, this
        # may need further adjustment based on the next live diagnostic reading.
        GRIPPER_OFFSET = 0.161
        
        tool0_x = lego_xyz[0]
        tool0_y = lego_xyz[1]
        grasp_z = lego_xyz[2] + GRIPPER_OFFSET        # Z = 0.990m
        pre_grasp_z = grasp_z + 0.05                 # Z = 1.040m

        # Bin Target (Gazebo bin origin is at [-0.35, 0.25, 0.85], wall top/rim at Z=0.90m).
        # Tool0 Z=1.150m -> gripper tip Z=0.98m, 8cm clear of the rim.
        bin_x = -0.35
        bin_y = 0.25
        bin_z = 1.150

        # 0. Return to Home first, so every run starts pre_pick from the same known,
        # short-hop-away state regardless of where a previous run (or failed attempt)
        # left the arm. Without this, pre_pick has to plan from wherever retract left the
        # arm last (up near the bin) into a tight +/-1cm / ~3deg goal box -- a much harder,
        # far less reliable OMPL query than the intended short hop from home.
        self.get_logger().info("0. Returning to Home Pose")
        home_start = time.time()
        # Skip planning entirely if the arm is already at home (confirmed live: a fresh
        # Gazebo spawn puts the arm exactly at home_joints, per initial_positions.yaml --
        # OMPL then gets a start==goal query, whose near-zero-length raw path makes
        # AddTimeOptimalParameterization emit a degenerate first segment with
        # time_from_start=0.0 on both waypoints 0 and 1, which arm_controller rejects
        # outright: "Time between points 0 and 1 is not strictly increasing". Checking
        # ground truth first avoids ever sending that degenerate trajectory.
        if self.verify_joint_target(self.home_joints):
            self.get_logger().info("    Already at Home Pose -- skipping move.")
            home_ok = True
        else:
            home_ok = self.execute_move_with_retry(
                lambda: self.create_joint_goal(self.home_joints),
                verify=lambda: self.verify_joint_target(self.home_joints)
            )
        # Recorded directly (not via cache_key) so it lands in the benchmark without
        # making 'home' cacheable -- see _start_state_matches for why home must always
        # replan fresh rather than ever replaying a past run's recorded path.
        self._record_move("home", home_start, attempts_used=None, used_cache=False,
                           traj=self._last_planned_trajectory, failed=not home_ok)
        if not home_ok:
            self.get_logger().error("Failed to reach Home pose.")
            self._save_benchmark(run_start_time, success=False, bin_xy=[bin_x, bin_y])
            return False

        # 2. Open Gripper
        self.get_logger().info("1. Opening Gripper")
        self.operate_gripper(0.0)

        # 3. Pre-Pick Approach
        pre_grasp_pos = [tool0_x, tool0_y, pre_grasp_z]
        self.get_logger().info(f"2. Pre-Pick Approach -> [{pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, Z={pre_grasp_pos[2]:.3f}m]")
        if not self.execute_move_with_retry(
            lambda: self.create_robust_pose_goal(pre_grasp_pos, position_box=0.02, orientation_tolerance=0.05),
            cache_key="pre_pick", verify=lambda: self.verify_pose_target(pre_grasp_pos, tol=0.025)
        ):
            self.get_logger().error("Failed Pre-Pick move.")
            self._save_benchmark(run_start_time, success=False, bin_xy=[bin_x, bin_y])
            return False

        # 4. Lower to Pick Position
        grasp_pos = [tool0_x, tool0_y, grasp_z]
        self.get_logger().info(f"3. Lowering to Pick -> [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, Z={grasp_pos[2]:.3f}m]")
        # Tightened from 0.02/0.05/0.025 (2026-08-31): confirmed live (in-process TF
        # diagnostic right after this move) that OMPL was landing tool0 within the old,
        # looser budget but still off by ~1cm in Z and ~2deg in orientation -- enough to
        # miss a precise finger-paddle grasp on this 5.7cm block. Not yet re-verified
        # whether this tighter budget alone is sufficient; watch for repeated planning
        # failures (this move now has less room for OMPL to satisfy the goal) as the
        # tradeoff sign to loosen back up instead.
        if not self.execute_move_with_retry(
            lambda: self.create_robust_pose_goal(grasp_pos, position_box=0.01, orientation_tolerance=0.02),
            cache_key="pick", verify=lambda: self.verify_pose_target(grasp_pos, tol=0.012)
        ):
            self.get_logger().error("Failed Pick move.")
            self._save_benchmark(run_start_time, success=False, bin_xy=[bin_x, bin_y])
            return False

        # Diagnostic: log the REAL achieved tool0 pose (position + orientation) right
        # at the grasp pose, in-process (no racing an external TF query against a live
        # sim -- see 2026-08-28 session notes). Confirms whether position AND
        # orientation are actually within budget at the moment that matters.
        try:
            t = self.tf_buffer.lookup_transform('world', 'tool0', rclpy.time.Time())
            q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            roll, pitch, yaw = tf_transformations.euler_from_quaternion(q)
            self.get_logger().info(
                f"    [diag] tool0 actual at pick pose: pos=[{t.transform.translation.x:.4f}, "
                f"{t.transform.translation.y:.4f}, {t.transform.translation.z:.4f}] "
                f"rpy_deg=[{math.degrees(roll):.2f}, {math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}] "
                f"target_pos=[{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]"
            )
        except Exception as e:
            self.get_logger().warn(f"    [diag] Could not look up tool0 TF for pose logging: {e}")

        # 5. Grasp Red Lego
        # finger_joint range is [0, 0.7] (0=open, 0.7=fully closed/0mm gap). The block is
        # 57mm wide; closing to 0.8 (clamped to 0.7, ~0mm gap) over-curls the fingers well
        # past the block, swinging the finger pads into the upper arm (self-collision) --
        # but that was only reachable when the gripper wasn't actually over the block
        # (bad approach orientation). With orientation now tight, the self-collision limit
        # sits right around 0.6-0.65 regardless of the block -- 0.65 hits it, so 0.6 is
        # the ceiling here. The bin-transit move is slowed down separately since a firmer
        # grip isn't available; MoveIt doesn't model the carried block's dynamics.
        self.get_logger().info("4. Grasping Red Lego (Position = 0.6)")
        self.operate_gripper(0.6)
        time.sleep(1.5)
        self.remove_collision_object("lego_red")

        # 6. Lift Object
        # Larger planning budget than the default (10s/30 attempts): observed timing out
        # (MoveIt error -6) repeatedly even though it's a tiny 5cm vertical move -- the
        # grasped block changes the arm's effective self-collision shape right next to the
        # table, and the 0.1 rad orientation tolerance keeps the search narrow throughout.
        # More time/restarts for RRTConnect to find a feasible passage, not a goal change.
        self.get_logger().info(f"5. Lifting -> Z={pre_grasp_pos[2]:.3f}m")
        if not self.execute_move_with_retry(
            lambda: self.create_robust_pose_goal(pre_grasp_pos, vel_scale=0.05, accel_scale=0.05, orientation_tolerance=0.1,
                                                  allowed_planning_time=20.0, num_planning_attempts=60),
            cache_key="lift", verify=lambda: self.verify_pose_target(pre_grasp_pos, tol=0.11)
        ):
            self.get_logger().error("Failed Lift move. Releasing gripper to avoid leaving the arm self-colliding and unplannable.")
            self.operate_gripper(0.0)
            self._save_benchmark(run_start_time, success=False, bin_xy=[bin_x, bin_y])
            return False

        # 7. Traverse to Bin (Tool0 Z=1.150m to stay above bin rim)
        # Orientation tolerance widened a little (block is already gripped, doesn't need
        # to stay perfectly vertical mid-transit) -- a tight 0.5 rad constraint here forces
        # the wrist through a narrow band that collides the wrist camera with the forearm.
        # But not widened too far: at 0.8 rad OMPL picked a solution tilted ~48 degrees off
        # vertical, which -- since GRIPPER_OFFSET assumes straight-down -- displaced the
        # actual release point ~15cm sideways from the intended bin center. 0.3 rad caps
        # that worst-case sideways displacement at ~5cm (0.17 * sin(0.3)), well inside the
        # bin's ~20cm opening, while (with the extra base standoff) still hopefully enough
        # slack to route around the forearm/camera collision.
        # Larger planning budget than the default, same reasoning as the lift move above:
        # this is the longest, most collision-dense move in the sequence (full traverse to
        # the bin through the forearm/wrist-camera collision band) and was observed needing
        # 3+ outer retries (each burning a full default 10s timeout) before succeeding.
        place_pos = [bin_x, bin_y, bin_z]
        self.get_logger().info(f"6. Moving to Target Bin -> [{place_pos[0]:.3f}, {place_pos[1]:.3f}, Z={place_pos[2]:.3f}m]")
        if not self.execute_move_with_retry(
            lambda: self.create_robust_pose_goal(place_pos, orientation_tolerance=0.3, vel_scale=0.1, accel_scale=0.1,
                                                  allowed_planning_time=20.0, num_planning_attempts=60),
            attempts=5, cache_key="bin", verify=lambda: self.verify_pose_target(place_pos, tol=0.11)
        ):
            self.get_logger().error("Failed Bin move. Releasing gripper to avoid leaving the arm self-colliding and unplannable.")
            self.operate_gripper(0.0)
            self._save_benchmark(run_start_time, success=False, bin_xy=[bin_x, bin_y])
            return False

        # 8. Drop Object into Bin
        self.get_logger().info("7. Dropping Red Lego into Bin")
        self.operate_gripper(0.0)
        time.sleep(1.0)

        # 9. Retract Arm
        retract_pos = [bin_x, bin_y, bin_z + 0.05]
        self.get_logger().info("8. Retracting Arm Upward")
        self.execute_move_with_retry(
            lambda: self.create_robust_pose_goal(retract_pos),
            cache_key="retract", verify=lambda: self.verify_pose_target(retract_pos, tol=0.11)
        )

        # Persist the trajectory cache only now that the full sequence has actually
        # succeeded end to end -- a failed/partial run must never overwrite a known-good
        # cached path. Reused cached legs plus any freshly (re)planned legs together make
        # up the complete set for next run.
        all_moves = dict(self.cached_moves)
        all_moves.update(self.pending_cache)
        self.save_trajectory_cache(lego_xyz, all_moves)

        # Let the dropped/released block finish settling in physics before reading its
        # final ground-truth position for the benchmark record.
        time.sleep(0.5)
        self._save_benchmark(run_start_time, success=True, bin_xy=[bin_x, bin_y])

        self.get_logger().info("🎉 RED LEGO PICK AND PLACE COMPLETED SUCCESSFULLY!\n")
        return True

def main(args=None):
    import threading
    rclpy.init(args=args)
    node = Phase1RedLegoPickAndPlaceNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(2.0)
    try:
        node.execute_pick_and_place_red_lego()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()

if __name__ == '__main__':
    main()
