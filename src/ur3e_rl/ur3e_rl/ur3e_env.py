import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
import json
import datetime
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs  # noqa: F401 -- registers PoseStamped conversions for Buffer.transform()
import time
import threading
from ur3e_rl.moveit_bridge import MoveItHandoffBridge

class UR3eEnv(gym.Env):
    """
    Custom Environment that follows Gymnasium interface for UR5e Gazebo Sim-to-Real.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, env_id=0, namespace=''):
        super(UR3eEnv, self).__init__()

        # namespace must match the `namespace` launch arg of the ur_gazebo stack this
        # env is meant to talk to -- required to run multiple Gazebo+MoveIt instances in
        # parallel (see ur.gazebo.launch.py's namespace/gz_partition args). All topics
        # below are relative (no leading "/") specifically so this node's own namespace
        # resolves them, instead of every instance colliding on the same global topic.
        self.namespace = namespace

        # Fix 1: Initialize ROS 2 INSIDE the worker/environment process,
        # never in the main script, to prevent the 'Poison Fork' crash.
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f'ur3e_rl_env_node_{env_id}', namespace=namespace)

        # --- ROS 2 Interfaces ---
        self.joint_states_sub = self.node.create_subscription(
            JointState,
            'joint_states',
            self._joint_state_callback,
            10)

        # Target-source toggle (2026-09-08, Vast.ai/YOLO-integration fork): 'ground_truth'
        # (default, unchanged behavior -- direct Gazebo query, see _update_target_pose)
        # or 'perception', which instead uses yolo_detector.py's /detected_object_pose
        # topic. Env-var driven so existing call sites (train_sac.py etc.) don't need
        # touching. NOT yet validated end-to-end -- see the perception-mode branch in
        # _update_target_pose for the known gaps (no hand-eye calibration, no noise
        # characterization) before trusting results trained under this mode.
        self.target_source = os.environ.get('UR3E_TARGET_SOURCE', 'ground_truth')
        # Synthetic noise added to the observed target position in _get_obs() (meters,
        # Gaussian std). Default 0.0 = unchanged prior behavior (noiseless ground truth).
        # Set from a real calibration test's measured std once one exists (see
        # _update_target_pose_from_perception's known-gap note) -- an unmeasured guess here
        # would just be a second unverified number stacked on top of the first.
        self.target_noise_std_m = float(os.environ.get('UR3E_TARGET_NOISE_STD_M', '0.0'))
        self._perception_pose = None  # last received PoseStamped.pose.position, world frame
        self._perception_pose_stamp = None
        if self.target_source == 'perception':
            self.perception_sub = self.node.create_subscription(
                PoseStamped,
                'detected_object_pose',
                self._perception_pose_callback,
                10)

        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            'arm_controller/joint_trajectory',
            10)

        # Robotiq 2F-140 gripper, driven through gripper_controller (GripperActionController)
        # like phase1_tf_pick_and_place.py's own gripper client. Relative (not
        # "/gripper_controller/gripper_cmd") for the same per-instance-isolation reason as
        # trajectory_pub above.
        self.gripper_client = ActionClient(
            self.node,
            GripperCommand,
            'gripper_controller/gripper_cmd')
            
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # State variables
        self.current_joint_positions = np.zeros(6, dtype=np.float32)
        self.current_joint_velocities = np.zeros(6, dtype=np.float32)
        self.current_ee_pose = np.zeros(7, dtype=np.float32) # x,y,z, qx,qy,qz,qw
        self.latest_joint_state = None  # raw JointState msg, kept for MoveGroup start_state

        self.target_pose = np.zeros(3, dtype=np.float32) # X, Y, Z target (lego's live position -- see _update_target_pose)
        self._target_pose_valid = False # True once target_pose holds a real fetched value this episode
        self.gz_world_name = 'lego_sorting'
        self.curriculum_level = 0.0 # Starts at 0.0 (Easy) and goes to 1.0 (Hard)
        self.current_step = 0
        self.episode_shaping_sum = 0.0  # reset every episode in reset(); see step()'s SHAPING_FLOOR

        # Reach-and-grasp only (2026-09-01 revert, see observation_space comment below for
        # why): always 'grasp', never transitions. Kept as a string (not removed outright)
        # so the CSV logger column and _active_goal_pose() below don't need touching, and
        # so a future re-introduction of a release/transport phase is a small diff, not a
        # re-add of removed plumbing.
        self.phase = 'grasp'
        self.last_transport_pose = [0.0, 0.0, 0.0]
        self.last_transport_ok = False

        # Hybrid planner+RL architecture (see hybrid planner+RL spec, Recommendation 3):
        # reset() now hands the arm to RL already parked at a pre-grasp standoff via
        # MoveIt (see self.handoff_bridge below), not at the far-away home pose. The old
        # 500-step budget was tuned for full home->target traversal; 50 steps is the
        # original estimate for this grasp-only, final-approach-only episode (2026-09-01:
        # reverted from a widened 90-step two-phase budget back to this, since the
        # release-phase leg it was padding for no longer exists). A first estimate needing
        # a real sweep, not a tuned value.
        #
        # Pure-RL toggle (2026-09-09, Vast.ai experimental track): UR3E_USE_CLASSICAL_HANDOFF
        # (default "true", unchanged behavior) lets reset() skip Phase A's MoveIt/OMPL
        # handoff entirely when set to "false" -- RL then controls the arm from the fixed
        # home pose all the way to grasp, not just the final approach. The reward function
        # and kill-switches were already start-distance-agnostic (see the Target Lost
        # kill-switch's own comment further down, already written to tolerate a large
        # starting EE-to-target distance) -- the only thing that mechanically needs to
        # change for a full-reach episode is this step budget, back to the project's own
        # prior full-traversal value. Still a first estimate, not re-tuned for this exact
        # home pose / lego position -- expect to revisit after a real run.
        self.use_classical_handoff = os.environ.get('UR3E_USE_CLASSICAL_HANDOFF', 'true').lower() == 'true'
        self.max_episode_steps = 50 if self.use_classical_handoff else 500  # Safety timeout per episode (does NOT reduce total training steps)

        # Phase A (classical) handoff bridge -- drives the arm to the pre-grasp standoff via
        # OMPL/MoveIt at the start of every reset(), before RL (Phase B) takes over.
        self.handoff_bridge = MoveItHandoffBridge(
            self.node, self.tf_buffer, lambda: self.latest_joint_state
        )
        self.last_handoff_pose = [0.0, 0.0, 0.0]
        self.last_handoff_ok = False

        # Ground-truth-vs-YOLO diagnostic (2026-09-09, Vast.ai experimental track): only
        # populated once per episode, in perception mode, purely for logging -- see
        # reset()'s perception branch and _fetch_ground_truth_target_pose(). The RL-visible
        # self.target_pose always stays the raw perception value; this is not used to
        # correct/blend it. None until the first successful comparison.
        self.last_ground_truth_pose = [0.0, 0.0, 0.0]
        self.last_perception_error_m = None

        # --- Curriculum-widened reach (2026-09-04) ---
        # curriculum_level was previously computed correctly by train_sac.py's
        # CurriculumCallback but never consumed anywhere in this file -- every episode in
        # this project's history through 2026-09-04 started RL at the same fixed ~5-10cm
        # standoff regardless of curriculum_level. See project_phase2_rl_status memory /
        # the curriculum-widening implementation spec for the full diagnosis. Now wired:
        # reset() derives both the handoff standoff and the lego spawn-position
        # randomization radius from curriculum_level, ramping 0.0->1.0 over training.
        #
        # MAX_STANDOFF_M is a long-run ceiling, not a same-session target -- curriculum_level
        # only advances +0.05 per 20-episode window at >=90% rolling success (see
        # CurriculumCallback), so reaching 1.0 (the full 30cm standoff) realistically takes
        # far more training than a single day's budget. On a short timeline, expect
        # curriculum_level to sit well under 0.5 -- i.e. realistic reach for a short run is
        # closer to ~12-15cm standoff, not anywhere near this ceiling.
        self.MIN_STANDOFF_M = 0.05  # matches this project's entire prior fixed value
        self.MAX_STANDOFF_M = 0.30

        # Lateral (XY) spawn-position randomization radius, in meters, around the previously
        # fixed lego spawn point (world x=0.802, y=0.290 -- see reset()'s respawn_cmd).
        # Bounds derived from this rig's actual constraints, not copied from the standoff
        # pattern (a different quantity -- Z-distance vs. XY-radius):
        #   - Target Lost kill-switch fires at > 1.0m from base (see step()). The fixed
        #     spawn point's own base-relative distance was measured live at ~0.843m
        #     (2026-09-03 hardware CSV, Target_X/Y/Z ~= [0.802, -0.260, 0.010] in base_link
        #     frame) -- leaving only ~0.157m of margin in the worst (away-from-base)
        #     direction before the kill-switch would fire on spawn position alone, before
        #     the arm ever moves. A conservative radius keeps randomized spawns well clear
        #     of that boundary.
        #   - The table's MoveIt collision box (moveit_bridge.py's ensure_scene(), ~1.9m x
        #     2.6m) is not the binding constraint -- the lego rests ON the table, and the
        #     robot's <1m reach envelope is far smaller than the table's footprint, so the
        #     kill-switch distance above is what actually limits this radius.
        # 0.05m (5cm) is a conservative starting ceiling given the ~0.157m worst-case margin
        # above -- worth re-deriving with a live in-sim measurement (actual achieved
        # Target_X/Y/Z spread at curriculum_level=1.0) before trusting it at the top of the
        # ramp; flagged here rather than silently assumed correct.
        self.MIN_SPAWN_RADIUS_M = 0.0  # curriculum_level=0: behaves exactly like the old fixed spawn
        self.MAX_SPAWN_RADIUS_M = 0.05

        # Current episode's standoff/spawn-radius (meters), set each reset() from
        # curriculum_level -- logged to the CSV so a curriculum ramp is visible in raw data,
        # not just inferred from the Curriculum_Lvl column. See the implementation spec's
        # acceptance-check requirement.
        self.current_standoff_m = self.MIN_STANDOFF_M
        self.current_spawn_radius_m = self.MIN_SPAWN_RADIUS_M

        # Potential-based distance shaping (2026-09-04, replacing the raw -dist term -- see
        # the curriculum-widening implementation spec, section 3): reward each step is
        # (prev_dist - dist) instead of accumulating a flat -dist penalty for merely being
        # far away. This is what actually made the 2026-08-20 fail-fast reward-hacking
        # exploit possible in the first place (a long attempt costs more accumulated penalty
        # than one instant collision) -- SHAPING_FLOOR (see step()) capped the exploit's
        # relative cheapness but never removed the underlying incentive. Potential-based
        # shaping removes it structurally: a policy that never closes distance never earns
        # this term, rather than being penalized more the longer it (honestly) tries.
        # Initialized properly in reset() to the post-handoff distance; None here is only a
        # pre-first-reset placeholder.
        self._prev_dist = None

        # Bounded per-step joint motion (see spec section 6): action[0:6] is now a delta
        # added to the arm's current joint positions, capped at +/-max_joint_delta_rad,
        # instead of an absolute [-pi, pi] target sent with a fixed 0.1s arrival window --
        # that combination previously let the policy command up to ~162 deg/step jumps.
        # 0.08 rad over the unchanged 0.1s sim_step_time caps commanded joint speed at
        # ~0.8 rad/s. action[6] (gripper) is unchanged: still an absolute command.
        self.max_joint_delta_rad = 0.08

        # We need mapping because /joint_states isn't guaranteed to be in a specific order
        self.ur3e_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Hard position limits per joint (ur_description/config/ur5/joint_limits.yaml),
        # in ur3e_joint_names order. elbow_joint is artificially halved to +/-180deg
        # (physical self-collision with shoulder_lift, not a controller limit); the rest
        # are the UR5's full +/-360deg continuous range. step() clips arm_target to these
        # (minus a small margin) before publishing -- without it, repeated same-direction
        # deltas can walk a joint's raw, unwrapped position past its real limit. The
        # controller then can't track the out-of-range setpoint, the joint physically
        # stalls at the boundary, and -- since MoveIt/OMPL also refuses to plan from a
        # start state it can't reconcile with that stalled joint -- every subsequent
        # reset()'s handoff fails too, wedging the episode loop permanently against the
        # table. See [[project-phase2-rl-status]].
        joint_limit_margin = 0.05  # rad, keep commanded targets clear of the true limit
        self.joint_pos_min = np.array([
            -2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi
        ], dtype=np.float32) + joint_limit_margin
        self.joint_pos_max = np.array([
            2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi
        ], dtype=np.float32) - joint_limit_margin

        # --- Gym Spaces ---
        # Action space: action[0:6] = bounded per-step joint deltas (rad, see
        # max_joint_delta_rad above), action[6] = absolute Gripper command (0=open,
        # 1=closed mapped to bounds).
        action_low = np.array([-self.max_joint_delta_rad] * 6 + [-np.pi], dtype=np.float32)
        action_high = np.array([self.max_joint_delta_rad] * 6 + [np.pi], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Observation space: 6 joint angles + 6 joint velocities + 7 EE pose (xyz, quat) + 3
        # Goal Pos (lego) + 1 Gripper Pos = 23 observations.
        #
        # 2026-09-01: reverted from the 24-D two-phase (pick-then-place) formulation back
        # to reach-and-grasp only, matching the thesis's actual scoped task. The two-phase
        # extension (phase indicator, bin goal, transport-to-bin classical handoff inside
        # step()) never achieved a single recorded success across ~500k training steps;
        # this task is deliberately narrower so a policy has a real chance of converging.
        # See project_phase2_rl_status memory.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        
        # Simulation step timing parameters
        self.sim_step_time = 0.1 # seconds to execute single action per step

        self.node.get_logger().info(f'UR5e Gymnasium Environment {env_id} Initialized')
        
        # Fix 4: Run rclpy.spin() in a background thread so ROS 2 receives
        # sensor data continuously — even while the neural network is doing
        # backpropagation in the main learning thread.
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin_thread.start()

        # Persistent background Gazebo pose subscriber (2026-09-09 fix -- see
        # _get_gz_object_position). Spawns ONE long-lived `ign topic -e` process that
        # streams every pose update, parsed continuously into self._gz_pose_cache.
        # Callers then do an in-memory dict lookup instead of paying a fresh subprocess
        # spawn + topic discovery/subscribe round-trip on every single call, which is
        # what was timing out ~54% of the time (334/615 warn lines over one 5hr run)
        # once train_sac's own CPU load and real_time_factor=0's uncapped physics were
        # both competing for the same cores. See [[project-phase2-rl-status]].
        self._gz_pose_cache = {}
        self._gz_pose_cache_lock = threading.Lock()
        self._gz_pose_cache_time = 0.0
        self._gz_pose_proc = None
        self._gz_pose_thread_stop = threading.Event()
        self._gz_pose_thread = threading.Thread(target=self._gz_pose_subscriber_loop, daemon=True)
        self._gz_pose_thread.start()

        # --- Custom Hardware Logger ---
        os.makedirs("./rl_logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hardware_csv_path = f"./rl_logs/hardware_log_env{env_id}_{timestamp}.csv"
        
        # Create and write headers so the table is fully labeled in Excel
        with open(self.hardware_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                "Global_Step", "Curriculum_Lvl", "Dist_to_Target", "Reward", "Terminated",
                "Table_Collision", "Self_Collision",
                "J1_Pos", "J2_Pos", "J3_Pos", "J4_Pos", "J5_Pos", "J6_Pos",
                "J1_Vel", "J2_Vel", "J3_Vel", "J4_Vel", "J5_Vel", "J6_Vel",
                "Gripper_Pos", "EE_X", "EE_Y", "EE_Z",
                "Target_X", "Target_Y", "Target_Z",
                "Handoff_X", "Handoff_Y", "Handoff_Z", "Handoff_OK",
                "Act_J1", "Act_J2", "Act_J3", "Act_J4", "Act_J5", "Act_J6", "Act_Gripper",
                "Phase", "Goal_X", "Goal_Y", "Goal_Z",
                "Transport_X", "Transport_Y", "Transport_Z", "Transport_OK",
                "Placement_Error_M",
                # Curriculum-widened reach (2026-09-04): Handoff_OK above already lets a
                # rising handoff-failure rate at higher curriculum levels be cross-checked
                # against these two new columns -- see the implementation spec's §1d.
                "Standoff_M", "Spawn_Radius_M", "Contact_Detected",
                # Ground-truth-vs-YOLO diagnostic (2026-09-09): only populated in
                # perception mode, once per episode -- see reset()'s perception branch.
                # Blank in ground_truth mode (nothing to compare against itself).
                "GT_X", "GT_Y", "GT_Z", "Perception_Error_M"
            ]
            writer.writerow(headers)
            
        self.global_step_count = 0

    def set_curriculum_level(self, level):
        """ Update the difficulty level of the environment (0.0 to 1.0) """
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def _perception_pose_callback(self, msg):
        """ Cache the latest YOLO-detected object pose (world frame) -- consumed by
        _update_target_pose() when target_source == 'perception'. """
        self._perception_pose = msg.pose.position
        self._perception_pose_stamp = self.node.get_clock().now()

    def _joint_state_callback(self, msg):
        """ Update internal state when new joint data arrives """
        self.latest_joint_state = msg  # raw, unwrapped -- see MoveItHandoffBridge start_state usage
        for i, target_joint in enumerate(self.ur3e_joint_names):
            if target_joint in msg.name:
                idx = msg.name.index(target_joint)
                self.current_joint_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.current_joint_velocities[i] = msg.velocity[idx]
        
        # Track Robotiq finger joint
        if 'finger_joint' in msg.name:
            idx = msg.name.index('finger_joint')
            self.current_gripper_pos = msg.position[idx]

    def _update_ee_pose(self):
        """ Fetch the latest end-effector pose using TF2 -- averaged across both
        gripper finger pads (not just one), so 'distance to target' measures
        something close to the actual grasp center instead of a single asymmetric
        fingertip. A single-pad measurement puts the true fingertip-to-object-center
        distance at roughly the object's own half-width (2.85cm for the 5.7cm lego)
        even in a perfectly centered grasp, before any additional TF-frame offset --
        that ate most of the 5cm success margin (confirmed live 2026-09-02: an
        episode held at dist=0.0502, 0.2mm over the threshold, for 40s straight
        during what looked like a real, visually-confirmed grasp) and plausibly cost
        real grasps that never registered as under-threshold. See
        [[project-phase2-rl-status]].
        """
        try:
            t_right = self.tf_buffer.lookup_transform(
                'base_link', 'right_inner_finger_pad',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            t_left = self.tf_buffer.lookup_transform(
                'base_link', 'left_inner_finger_pad',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # Position: true midpoint between the two pads -- this is what should sit
            # near the object's center during a good grasp, unlike either pad alone.
            # Orientation: kept from the right pad only (unchanged) -- averaging two
            # quaternions naively doesn't give a meaningful "midpoint orientation,"
            # and orientation isn't used in the dist calculation anyway, only
            # position is. Still a 7-element (xyz+quat) array either way, so this
            # doesn't change observation_space shape / break checkpoint compatibility.
            self.current_ee_pose = np.array([
                (t_right.transform.translation.x + t_left.transform.translation.x) / 2.0,
                (t_right.transform.translation.y + t_left.transform.translation.y) / 2.0,
                (t_right.transform.translation.z + t_left.transform.translation.z) / 2.0,
                t_right.transform.rotation.x,
                t_right.transform.rotation.y,
                t_right.transform.rotation.z,
                t_right.transform.rotation.w
            ], dtype=np.float32)
        except Exception as e:
            self.node.get_logger().warn(f'Could not fetch EE pose: {e}')

    def _gz_pose_subscriber_loop(self):
        """Background thread body (2026-09-09): keeps ONE `ign topic -e` process alive,
        continuously streaming /world/<world>/pose/info into self._gz_pose_cache instead
        of spawning a new `ign topic -e` subprocess (with its own discovery/subscribe
        handshake) on every single _get_gz_object_position() call -- that per-call design
        was what actually timed out under CPU contention, not the query itself (confirmed
        live: identical query completes in ~0.25s run standalone). Restarts the process
        with a short backoff if it ever exits (Gazebo restart, transport hiccup, etc)."""
        topic = f'/world/{self.gz_world_name}/pose/info'
        while not self._gz_pose_thread_stop.is_set():
            try:
                self._gz_pose_proc = subprocess.Popen(
                    ['ign', 'topic', '-e', '-t', topic, '--json-output'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
                )
            except Exception as e:
                self.node.get_logger().warn(f'_gz_pose_subscriber_loop: failed to start `ign topic -e`: {e}')
                time.sleep(2.0)
                continue

            buf = ''
            for chunk in iter(lambda: self._gz_pose_proc.stdout.read(4096), ''):
                if self._gz_pose_thread_stop.is_set():
                    break
                buf += chunk
                # Messages aren't reliably newline-delimited (same quirk the old per-call
                # `-n 1` path hit with concatenated JSON, confirmed live 2026-09-03) --
                # repeatedly raw_decode the first complete JSON value off the front of buf
                # and drop it, same technique as before, just looped over a live stream.
                while True:
                    stripped = buf.lstrip()
                    if not stripped:
                        buf = ''
                        break
                    try:
                        data, idx = json.JSONDecoder().raw_decode(stripped)
                    except json.JSONDecodeError:
                        break  # incomplete message -- wait for more data
                    buf = stripped[idx:]
                    poses = {
                        p.get('name'): [p.get('position', {}).get('x', 0.0),
                                        p.get('position', {}).get('y', 0.0),
                                        p.get('position', {}).get('z', 0.0)]
                        for p in data.get('pose', []) if p.get('name')
                    }
                    with self._gz_pose_cache_lock:
                        self._gz_pose_cache = poses
                        self._gz_pose_cache_time = time.time()

            if self._gz_pose_thread_stop.is_set():
                break
            self.node.get_logger().warn(
                f'_gz_pose_subscriber_loop: `ign topic -e` on {topic} exited unexpectedly, restarting in 2s'
            )
            time.sleep(2.0)

    def _stop_gz_pose_subscriber(self):
        self._gz_pose_thread_stop.set()
        if self._gz_pose_proc is not None:
            self._gz_pose_proc.terminate()
        self._gz_pose_thread.join(timeout=2.0)

    def _get_gz_object_position(self, name, timeout=3.0):
        """Ground-truth [x, y, z] (world frame) for a named object, read from
        self._gz_pose_cache -- kept fresh by the persistent background subscriber
        started in __init__ (_gz_pose_subscriber_loop) instead of querying Gazebo
        per-call. `timeout` is now the cache-staleness threshold, not a subprocess
        timeout: the old per-call `ign topic -e` subprocess design paid a fresh topic
        discovery/subscribe round-trip on every single call, which timed out ~54% of
        calls (334/615 warn lines) over one 5hr run once train_sac's own CPU load and
        real_time_factor=0's uncapped physics were both competing for the same cores --
        see [[project-phase2-rl-status]]. Returns None if the cache is stale/empty or
        the name isn't in the latest snapshot."""
        with self._gz_pose_cache_lock:
            cache = self._gz_pose_cache
            cache_time = self._gz_pose_cache_time
        age = time.time() - cache_time
        if cache_time == 0.0 or age > timeout:
            self.node.get_logger().warn(
                f"_get_gz_object_position('{name}'): pose cache is stale ({age:.1f}s old) -- "
                "background `ign topic -e` subscriber may be down (check GZ_PARTITION matches "
                "the Gazebo instance this env is meant to talk to)."
            )
            return None
        pos = cache.get(name)
        if pos is None:
            self.node.get_logger().warn(
                f"_get_gz_object_position('{name}'): not found in pose list, saw: {list(cache.keys())}"
            )
        return pos

    def _update_target_pose(self):
        """Fetch the dynamic physical position of the red lego object, in base_link frame.

        'lego_red' is never actually published on the standard TF tree by this rig's
        ros_gz_bridge config -- it only bridges Gazebo pose info to the non-standard
        /gz_poses_static /gz_poses_dynamic topics, which TransformListener never
        subscribes to, and whose messages carry an empty child_frame_id on this bridge
        version anyway (see phase1_tf_pick_and_place.py's identical _get_gz_object_position
        for the same problem solved the same way). So the TF lookup below always throws and
        falls through to a direct Gazebo ground-truth query -- previously the bare
        `except: pass` here left target_pose frozen at its np.zeros(3) init forever, which
        every hardware_log_*.csv confirmed: Target_X/Y/Z read exactly 0.0000 across every
        recorded step of every run. That meant the whole reward (`dist = ee_pos -
        target_pose`) was training the policy to reach for base_link's own origin, not the
        lego -- see [[project-phase2-rl-status]].

        The gz query only runs once per episode (cached via _target_pose_valid, reset to
        False in reset()) since lego_red is static mid-episode -- spawning a subprocess on
        every 10Hz training step would make training crawl.

        target_source == 'perception': skips the ground-truth path entirely and sources
        target_pose from yolo_detector.py's /detected_object_pose instead (see
        _perception_pose_callback). Unlike ground truth this is NOT cached once-per-episode
        -- it re-reads the latest detection every call, since the whole point is exposing
        the policy to real (or simulated) perception noise/latency rather than a single
        clean per-episode snapshot.
        """
        if self.target_source == 'perception':
            self._update_target_pose_from_perception()
            return
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'lego_red',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            self.target_pose = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ], dtype=np.float32)
            self._target_pose_valid = True
            return
        except Exception:
            pass

        if self._target_pose_valid:
            return  # already have a real value cached for this episode

        gt_pose = self._fetch_ground_truth_target_pose()
        if gt_pose is not None:
            self.target_pose = gt_pose
            self._target_pose_valid = True

    def _fetch_ground_truth_target_pose(self):
        """Ground-truth lego position (base_link frame), via Gazebo's own pose info --
        the same query/transform this method's caller used inline before it was pulled out
        here (2026-09-09) so the perception-mode diagnostic in reset() below can reuse it
        for logging without duplicating the subprocess+transform logic. Returns None on
        query/transform failure -- callers keep whatever they already had."""
        gz_pos = self._get_gz_object_position('lego_red')
        if gz_pos is None:
            return None
        try:
            world_pose = PoseStamped()
            world_pose.header.frame_id = 'world'
            world_pose.header.stamp = rclpy.time.Time().to_msg()
            world_pose.pose.position.x, world_pose.pose.position.y, world_pose.pose.position.z = gz_pos
            world_pose.pose.orientation.w = 1.0
            base_pose = self.tf_buffer.transform(
                world_pose, 'base_link', timeout=rclpy.duration.Duration(seconds=0.2)
            )
            return np.array([
                base_pose.pose.position.x,
                base_pose.pose.position.y,
                base_pose.pose.position.z
            ], dtype=np.float32)
        except Exception as e:
            self.node.get_logger().warn(f'Could not transform gz ground-truth lego pose into base_link: {e}')
            return None

    def _update_target_pose_from_perception(self):
        """ target_source == 'perception' branch of _update_target_pose. Transforms
        yolo_detector.py's latest /detected_object_pose (world frame) into base_link,
        same as the ground-truth path's own world->base_link step. No detection received
        yet this run -> keeps the last known target_pose (zero-init before the first one).

        KNOWN GAP, not fixed by this code (see project_rl_architecture_spec memory /
        THESIS_RESULTS_REPORT.md Table D.2): yolo_detector.py's world-frame extrinsics are
        hardcoded for a fixed overhead sim camera, not this rig's wrist-mounted real
        camera, and no calibration test (mean/std error) has been run -- do not trust
        results trained under this mode as representative of real deployment accuracy
        until both are addressed.
        """
        if self._perception_pose is None:
            return
        try:
            world_pose = PoseStamped()
            world_pose.header.frame_id = 'world'
            world_pose.header.stamp = rclpy.time.Time().to_msg()
            world_pose.pose.position = self._perception_pose
            world_pose.pose.orientation.w = 1.0
            base_pose = self.tf_buffer.transform(
                world_pose, 'base_link', timeout=rclpy.duration.Duration(seconds=0.2)
            )
            self.target_pose = np.array([
                base_pose.pose.position.x,
                base_pose.pose.position.y,
                base_pose.pose.position.z
            ], dtype=np.float32)
            self._target_pose_valid = True
        except Exception as e:
            self.node.get_logger().warn(f'Could not transform perception target pose into base_link: {e}')

    def _active_goal_pose(self):
        """Single chokepoint for 'what is the RL policy currently being scored against'.
        Reach-and-grasp only (2026-09-01): always the live lego position. Kept as its own
        method (rather than inlining self.target_pose at call sites) so a future
        re-introduction of a release/bin phase only needs to change this one place."""
        return self.target_pose

    def _is_arm_colliding(self):
        """ Quickly checks if any critical arm joint (elbow/wrists) dips into the table. """
        critical_links = ['forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
        for link in critical_links:
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link',
                    link, 
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
                if t.transform.translation.z < 0.02:
                    self.node.get_logger().warn(f"Table Collision Detected at {link}!")
                    return True
            except Exception:
                pass
        return False

    # Layer 3 self-collision safety net for the RL fine-approach phase (spec section 5).
    # Pair selection is grounded in this rig's own SRDF (moveit_config/config/ur.srdf),
    # not guessed: the spec's own placeholder pairs (finger_pad vs forearm_link/wrist_1/2/
    # 3_link) are every one of them marked reason="Never" in that file -- the SRDF setup
    # assistant's sampler found them geometrically impossible for this kinematic chain
    # (the gripper mounts beyond wrist_3, too far down the chain to reach back to them),
    # so checking those specific pairs here would be dead code that never fires. The pairs
    # below are the ones this rig's SRDF actually leaves unconstrained -- i.e. genuinely
    # checked by MoveIt's own collision checker -- that a fine-approach motion could
    # plausibly close: the gripper body/fingertip folding back toward the upper arm or
    # shoulder housing.
    #
    # min_clearance_m values are coarse point-to-point placeholders (this rig's known
    # dimensions: GRIPPER_OFFSET=0.17m tool0->gripper tip for the Robotiq 2F-140, typical
    # UR5e upper_arm/shoulder housing diameter ~0.12-0.15m) -- NOT capsule/mesh distances.
    # Tighten with real collision-mesh geometry (what MoveIt's own checker uses) before
    # trusting this as more than a coarse safety net.
    SELF_COLLISION_PAIRS = [
        ("right_inner_finger_pad", "upper_arm_link", 0.12),
        ("right_inner_finger_pad", "shoulder_link", 0.15),
        ("robotiq_arg2f_base_link", "forearm_link", 0.10),
        ("robotiq_arg2f_base_link", "upper_arm_link", 0.12),
        ("robotiq_arg2f_base_link", "shoulder_link", 0.15),
    ]

    def _get_link_position(self, link):
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link', link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01)
            )
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except Exception:
            return None

    def _is_self_colliding(self):
        """ Coarse arm-vs-arm collision check -- see SELF_COLLISION_PAIRS above. """
        for link_a, link_b, min_dist in self.SELF_COLLISION_PAIRS:
            pos_a = self._get_link_position(link_a)
            pos_b = self._get_link_position(link_b)
            if pos_a is None or pos_b is None:
                continue
            if np.linalg.norm(pos_a - pos_b) < min_dist:
                self.node.get_logger().warn(f"Self-collision risk: {link_a} <-> {link_b}")
                return True
        return False

    def _get_obs(self):
        """ Construct the observation vector with injected Sensor Noise and Explicit Goal """
        # No need to call spin_once() here — the background thread handles all ROS 2 callbacks.
        self._update_ee_pose()
        self._update_target_pose()
        
        # --- Domain Randomization: Sensor Noise ---
        # Add a small amount of Gaussian noise to the joint positions and velocities
        # to prevent the policy from overfitting to perfect simulation sensing.
        
        pos_noise_std = 0.005 # ~0.3 degrees standard deviation
        vel_noise_std = 0.01  # ~0.5 deg/sec standard deviation
        
        noisy_positions = self.current_joint_positions + np.random.normal(0, pos_noise_std, size=6)
        noisy_velocities = self.current_joint_velocities + np.random.normal(0, vel_noise_std, size=6)
        
        # We need the finger joint position from the robot. It should be in the joint states list.
        # However, we only extract the first 6 explicit UR5e joints for current_joint_positions.
        # For the gripper, the simplest heuristic is to assume it is near the commanded action space if it isn't tracked in self.current_joint_positions, 
        # or we can extract it. Since we haven't explicitely subscribed to finger_joint in _joint_state_callback, 
        # let's add it right here or rely on the cached current_gripper_pos if we added it.
        # Actually, let's just make it a dedicated variable we update in callback!
        
        # For safety, if self.current_gripper_pos isn't set, default to 0.0.
        gripper_pos = getattr(self, 'current_gripper_pos', 0.0)

        goal_pose = self._active_goal_pose()
        if self.target_noise_std_m > 0.0:
            goal_pose = goal_pose + np.random.normal(0, self.target_noise_std_m, size=3)

        return np.concatenate((
            noisy_positions,
            noisy_velocities,
            self.current_ee_pose,
            goal_pose,  # lego position (reach-and-grasp only)
            [gripper_pos],     # Expose gripper state!
        ), dtype=np.float32)

    def step(self, action):
        """ 
        Execute one control step in the environment.
        action is an array of 7 values.
        """
        # action[0:6] is a bounded per-step joint delta (see max_joint_delta_rad in
        # __init__), added to the arm's current joint positions -- not clipped further
        # into [-pi, pi] here, since 5 of 6 joints have a real +/-360deg range and wrapping
        # a valid raw reading like shoulder_lift=4.7557 forces a near-2pi unintended sweep
        # to get there (see reset()'s home_action docstring for the identical bug fixed
        # there previously).
        arm_delta = np.clip(action[:6], -self.max_joint_delta_rad, self.max_joint_delta_rad)
        # Clip to each joint's physical position limit -- see joint_pos_min/max in
        # __init__ for why this is required, not just defensive.
        arm_target = np.clip(
            self.current_joint_positions + arm_delta, self.joint_pos_min, self.joint_pos_max
        )
        gripper_val = action[6]

        # Publish the joint trajectory
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.ur3e_joint_names

        point = JointTrajectoryPoint()
        point.positions = arm_target.astype(float).tolist()
        point.time_from_start.sec = int(self.sim_step_time)
        point.time_from_start.nanosec = int((self.sim_step_time - int(self.sim_step_time)) * 1e9)
        
        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)
        
        # Normalized from [-pi, pi] to [0.0, 0.8] (0.8 is fully closed for Robotiq --
        # finger_joint's own real limit is 0.7, so the top of this range clamps there).
        # Fire-and-forget: a 10Hz step() can't block on the action's full result, so this
        # only waits for the server (0.1s) then sends the goal async, same pattern as the
        # original Robotiq-based version of this env.
        gripper_cmd_val = float(((np.clip(gripper_val, -np.pi, np.pi) + np.pi) / (2 * np.pi)) * 0.8)
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = gripper_cmd_val
        goal_msg.command.max_effort = 100.0  # Matches Phase 1's proven operate_gripper() effort
        if self.gripper_client.wait_for_server(timeout_sec=0.1):
            self.gripper_client.send_goal_async(goal_msg)
        
        # Wait for the physical action to be executed in Gazebo.
        # No need for spin_once() — the background thread handles incoming messages.
        time.sleep(self.sim_step_time)
            
        # Collect new state
        state = self._get_obs()

        # Calculate Euclidean distance between End-Effector and the CURRENT phase's goal
        # (lego during 'grasp', bin during 'release' -- see _active_goal_pose). Here we
        # extract x, y, z from the EE pose (first 3 elements)
        ee_pos = state[12:15]
        dist = np.linalg.norm(ee_pos - self._active_goal_pose())

        # Potential-based distance shaping (2026-09-04, replacing the old flat -dist term --
        # see the curriculum-widening implementation spec, section 3). PHI(d) = -d, so this
        # is prev_dist - dist: reward *closing* distance, not accumulating a penalty for
        # merely being far away. This is what made the 2026-08-20 fail-fast reward-hacking
        # exploit possible in the first place (a long honest attempt costs more accumulated
        # penalty than one instant collision) -- SHAPING_FLOOR below capped the exploit's
        # relative cheapness but never removed the underlying incentive. A policy that never
        # closes distance now never earns this term, rather than being penalized more the
        # longer it (honestly) tries. self._prev_dist is set in reset() to the post-handoff
        # distance and updated at the end of this block for the next step.
        reward = float(self._prev_dist - dist)
        self._prev_dist = dist

        # Penalize high arm velocities to prevent severe stuttering or thrashing
        velocity_penalty = 0.05 * float(np.sum(np.abs(state[6:12])))
        reward -= velocity_penalty

        # Cap how much the per-step shaping penalty (distance + velocity, above) can
        # accumulate over an episode, floored at SHAPING_FLOOR. Originally load-bearing
        # against the 2026-08-20 exploit under the old flat -dist term (see
        # [[project-phase2-rl-status]]); with potential-based shaping above, the telescoping
        # sum of (prev_dist - dist) across an episode is already naturally bounded to
        # roughly the episode's total distance closed, well under this floor in normal
        # operation -- kept as a secondary safety net, not expected to do much work anymore.
        SHAPING_FLOOR = -5.0  # matches the smallest fixed termination penalty magnitude
        prospective_sum = self.episode_shaping_sum + reward
        if prospective_sum < SHAPING_FLOOR:
            reward = SHAPING_FLOOR - self.episode_shaping_sum
            self.episode_shaping_sum = SHAPING_FLOOR
        else:
            self.episode_shaping_sum = prospective_sum

        # Gripper normalized closed amount (0 to 1 scaling for conditional checks)
        gripper_closed_amount = ((np.clip(gripper_val, -np.pi, np.pi) + np.pi) / (2 * np.pi))
        
        # Termination conditions
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        
        # Initialize info dict
        info = {}

        # 1. Grasp-success condition (2026-09-04, contact-based -- see the
        # curriculum-widening implementation spec, section 2): grasp-only task, no lift/
        # transport/retention-verify in scope. Success = close distance AND gripper
        # commanded closed AND contact detected -- no MoveIt lift, no ground-truth
        # object-height re-query, no extra episode time. The previous lift-and-verify path
        # (MoveItHandoffBridge.lift_after_grasp(), removed) existed specifically to verify
        # grasp *retention*, which this task no longer requires.
        #
        # Contact signal: the joint-gap fallback from the implementation spec's section 2a,
        # not a Gazebo contact sensor -- deliberately chosen over adding a new SDF
        # <sensor type="contact">/plugin/ros_gz_bridge topic (flagged in the spec as the
        # highest-risk, least-proven item in this pass; this project's own history shows
        # this stack's ros_gz_bridge has already silently dropped data before). The
        # Robotiq finger_joint's *achieved* position (self.current_gripper_pos, real
        # feedback via _joint_state_callback) vs. the *commanded* position
        # (gripper_cmd_val, computed above) already signals contact indirectly: closing on
        # empty air should track the command closely, closing on the lego should stall
        # short of it. CONTACT_GAP_THRESHOLD calibrated live 2026-09-04
        # (contact_threshold_diagnostic.py, 5 genuine-grasp + 5 miss trials at
        # curriculum_level=0.0, ground-truth-labeled via direct MoveIt repositioning, full
        # env.reset() before every trial). Real result: misses consistently gap 0.1000 (NOT
        # 0 -- the finger_joint's own hard limit is 0.7, short of the 0.8 "closed" command,
        # regardless of whether anything is in the way), genuine grasps gapped 0.164-0.171
        # (4/5 trials; 1 outlier discarded as a diagnostic-script measurement artifact, not a
        # real reading). The original 0.05 guess sat *below* the 0.1 no-contact floor,
        # meaning it would have scored every miss as a false-positive "contact detected" --
        # confirmed empirically, not hypothetically. 0.13 sits centered in the real,
        # non-overlapping gap between 0.100 (miss ceiling) and 0.164 (grasp floor).
        CONTACT_GAP_THRESHOLD = 0.13  # meters of finger_joint travel
        achieved_gripper_pos = float(getattr(self, 'current_gripper_pos', 0.0))
        contact_gap = gripper_cmd_val - achieved_gripper_pos
        contact_detected = gripper_closed_amount > 0.5 and contact_gap > CONTACT_GAP_THRESHOLD

        # Note: no bonus for merely being close with the gripper open. Proximity is
        # already rewarded every step via the distance-shaping term above; a flat bonus
        # here used to double up on top of that and specifically rewarded camping close
        # with the gripper open instead of committing to a grasp attempt (non-terminating,
        # so it could be collected every step for an entire episode) -- see
        # [[project-phase2-rl-status]].
        if dist < 0.05 and gripper_closed_amount > 0.5:
            if contact_detected:
                reward += 15.0  # matches the other terminal-success magnitude in this file
                terminated = True
                info['termination_reason'] = 'Full Task Success'
            else:
                reward -= 5.0
                terminated = True
                info['termination_reason'] = 'False Grasp -- No Contact Detected'
                self.node.get_logger().warn(
                    f"Gripper closed near target but no contact detected "
                    f"(commanded={gripper_cmd_val:.4f}, achieved={achieved_gripper_pos:.4f}, "
                    f"gap={contact_gap:.4f} < threshold {CONTACT_GAP_THRESHOLD}) -- closed on "
                    "empty air, not the lego. Failing episode."
                )

        # "Kill Switches" for Safety and Physics Preservation:
        
        # 2. Speed Kill-Switch: Physics Explosion Protection
        # We increase this to 10.0 to prevent false-positives during early random exploration (fixes 0 FPS bug)
        if np.any(np.abs(state[6:12]) > 10.0):
            reward -= 5.0
            terminated = True
            info['termination_reason'] = 'Velocity Kill-Switch Triggered'
            
        # 3. Collision / Out of bounds for the entire Arm
        # Table and self-collision are kept as separate checks/penalties/log fields (not
        # collapsed into one "Collision" signal) so each failure mode stays identifiable
        # in the hardware CSV logs -- see spec section 5's observability rationale. Table
        # collision is checked (and, if true, wins) first: it's the immediate hardware/
        # physics risk, so a state that trips both is recorded as the more urgent one.
        table_collision = ee_pos[2] < 0.02 or self._is_arm_colliding()
        self_collision = False if table_collision else self._is_self_colliding()
        if table_collision:
            # Scaled by remaining distance-to-go (2026-09-04, implementation spec section
            # 3): with curriculum widening the standoff (see MAX_STANDOFF_M above), episodes
            # get longer before reaching the target, so an unscaled flat -5.0 here would
            # make "dive to an early collision from a large standoff" relatively cheaper
            # again as curriculum_level rises -- the same fail-fast exploit class diagnosed
            # 2026-08-20 (see [[project-phase2-rl-status]]), just re-emerging under a
            # different lever. Scaling by dist/MAX_STANDOFF_M penalizes diving from far away
            # more than diving from a near-trivial distance. Multiplier is a first-pass
            # value (see spec's "tune after observing behavior" note), not empirically tuned.
            reward -= 5.0 * (1.0 + float(dist) / self.MAX_STANDOFF_M)
            terminated = True
            info['termination_reason'] = 'Table Collision'
        elif self_collision:
            reward -= 7.0 # Distinct from table collision -- see spec section 5
            terminated = True
            info['termination_reason'] = 'Self Collision'

        # 4. Out-of-Bounds "Fast Fail" Kill-Switch (Lost Lego).
        # UR5 physically reach is ~0.85m. Z < 0.0 means it fell significantly lower than the table.
        #
        # This checks target_pose's distance from base_link's own origin (the robot's
        # reach envelope), NOT `dist` (EE-to-target). `dist` is large by design at the
        # start of every episode -- the safe home/retract pose is intentionally far from
        # the table (arm folded up around world z~1.7m vs. the lego at ~0.83m, ~1.2m+
        # apart) -- so gating on `dist` fast-failed literally every episode at step 1
        # before the policy ever got a chance to move, regardless of action. Checking
        # reachability from the base instead only fires when the object has actually left
        # the workspace (e.g. knocked off the table), which is what this kill-switch was
        # meant to catch. See [[project-phase2-rl-status]].
        #
        # 2026-09-01: unconditional now (was 'grasp'-phase-only) -- reach-and-grasp is the
        # only phase left, see observation_space's comment above. The old block 5
        # (release-phase bin-placement termination) is removed entirely along with it.
        #
        # 2026-09-03: briefly "fixed" this to subtract a true-base world offset, on a wrong
        # diagnosis -- self.target_pose is already base_link-relative (see
        # _update_target_pose's tf_buffer.transform(world_pose, 'base_link', ...) call a few
        # dozen lines up), so that subtraction double-counted the base offset and would have
        # tripped this kill-switch almost immediately. Reverted to the original, correct
        # form. Threshold lowered 1.2 -> 1.0 only to give tighter margin over the new
        # real-hardware-matched lego distance (~0.84m from base) -- not because 1.2 was wrong.
        placement_error_m = None
        target_dist_from_base = float(np.linalg.norm(self.target_pose))
        if target_dist_from_base > 1.0 or self.target_pose[2] < 0.0:
            reward -= 5.0
            terminated = True
            info['termination_reason'] = 'Target Lost (Out of Bounds)'
            self.node.get_logger().warn(f"LEGO LOST! Target Distance from base: {target_dist_from_base:.2f}m, Height: {self.target_pose[2]:.2f}m. Fast-Failing episode!")

        info['is_success'] = terminated and info.get('termination_reason') == 'Full Task Success'
        info['distance_to_target'] = dist
        goal_pose = self._active_goal_pose()

        # --- Write frame to Custom Hardware CSV Logger ---
        self.global_step_count += 1
        try:
            with open(self.hardware_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                row = [
                    self.global_step_count, self.curriculum_level, f"{dist:.4f}", f"{reward:.4f}", terminated,
                    table_collision, self_collision,
                    *[f"{p:.4f}" for p in self.current_joint_positions],
                    *[f"{v:.4f}" for v in self.current_joint_velocities],
                    f"{getattr(self, 'current_gripper_pos', 0.0):.4f}",
                    f"{ee_pos[0]:.4f}", f"{ee_pos[1]:.4f}", f"{ee_pos[2]:.4f}",
                    f"{self.target_pose[0]:.4f}", f"{self.target_pose[1]:.4f}", f"{self.target_pose[2]:.4f}",
                    f"{self.last_handoff_pose[0]:.4f}", f"{self.last_handoff_pose[1]:.4f}", f"{self.last_handoff_pose[2]:.4f}",
                    self.last_handoff_ok,
                    *[f"{a:.4f}" for a in action],
                    self.phase,
                    f"{goal_pose[0]:.4f}", f"{goal_pose[1]:.4f}", f"{goal_pose[2]:.4f}",
                    f"{self.last_transport_pose[0]:.4f}", f"{self.last_transport_pose[1]:.4f}", f"{self.last_transport_pose[2]:.4f}",
                    self.last_transport_ok,
                    "" if placement_error_m is None else f"{placement_error_m:.4f}",
                    f"{self.current_standoff_m:.4f}", f"{self.current_spawn_radius_m:.4f}",
                    contact_detected,
                    f"{self.last_ground_truth_pose[0]:.4f}", f"{self.last_ground_truth_pose[1]:.4f}", f"{self.last_ground_truth_pose[2]:.4f}",
                    "" if self.last_perception_error_m is None else f"{self.last_perception_error_m:.4f}",
                ]
                writer.writerow(row)
        except Exception as e:
            self.node.get_logger().warn(f"Hardware logger failed to write: {e}")
            
        return state, reward, terminated, truncated, info

    def _send_home_trajectory(self, duration_sec):
        """ Publish the home-pose JointTrajectory with the given execution duration. """
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.ur3e_joint_names
        point = JointTrajectoryPoint()
        point.positions = self.home_action.tolist()
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)

    def _wait_for_safe_home(self, max_attempts=3):
        """Ground-truth-verified reset: previously reset() just published the home
        trajectory and blindly slept 1.0s, trusting it arrived (the same class of bug
        Phase 1 hit -- a controller/schedule elapsing is not the same as actually getting
        there). If the arm was left resting on the table by a prior episode's action, 1s
        isn't enough torque/time budget to lift a resting contact back off the table, so
        every subsequent episode started (and immediately re-collided in) practically the
        same drooped posture -- confirmed live: shoulder_pan/shoulder_lift frozen for 30+
        consecutive 1-step episodes while forearm_link sat at z=0.013 in base_link frame,
        just under the 0.02 collision threshold (see project-phase2-rl-status memory).

        This retries with escalating duration/wait and only proceeds once
        _is_arm_colliding() actually reads clear, instead of assuming success."""
        for attempt in range(max_attempts):
            duration = 1.0 + attempt * 2.0  # 1.0s, 3.0s, 5.0s
            self._send_home_trajectory(duration)
            time.sleep(duration + 0.5)
            if not self._is_arm_colliding():
                return True
            self.node.get_logger().warn(
                f"Reset attempt {attempt + 1}/{max_attempts}: arm still colliding after "
                f"{duration}s home trajectory -- retrying with more time."
            )
        self.node.get_logger().error(
            "Reset could not clear a table collision after all retries -- "
            "starting episode anyway from a possibly-unsafe state."
        )
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_shaping_sum = 0.0
        self.phase = 'grasp'
        self.last_transport_pose = [0.0, 0.0, 0.0]
        self.last_transport_ok = False
        self._target_pose_valid = False  # force a fresh ground-truth fetch after this episode's respawn

        # In a real Gazebo setup, you'd either publish a message to a custom reset node,
        # call the /reset_simulation service, or publish a home trajectory here.

        # Verified-safe retract pose -- same physical configuration as Phase 1's
        # home_joints (phase1_tf_pick_and_place.py) and Gazebo's own spawn convention
        # (ur_description/config/initial_positions.yaml), confirmed live via
        # /check_state_validity and an actual Gazebo trajectory execution to sit well
        # clear of the table (all arm links >= 0.89m above the table top).
        #
        # shoulder_lift is given as the raw, unwrapped 4.7557 rad (272.5 deg) rather than
        # its [-pi, pi]-equivalent -1.5275 rad -- despite this env's action_space being
        # bounded to [-pi, pi] (see __init__). That bound only clips step()'s policy
        # output; reset() publishes its own JointTrajectory directly and is never subject
        # to it, so there was no real conflict to "fix" by wrapping. Using the wrapped
        # value here was tried and broke reset(): the robot spawns at shoulder_lift ~=
        # 4.7557, and JointTrajectoryController interpolates raw joint-space (not
        # angle-wrapped), so commanding -1.5275 forced almost a full extra revolution
        # (~2*pi) to get there -- swinging wrist_1_link through the table on literally
        # the very first reset, before any training had even started. See
        # [[project-phase2-rl-status]].
        # Pure-RL home pose (2026-09-09, Vast.ai wrist-camera experimental track): the
        # retract pose above is verified-safe but was never meant to give the wrist camera
        # a view of anything -- it's a fold-up-and-away pose, and confirmed live that RL
        # starting there gets nothing but low-confidence false-positive detections (the
        # camera isn't pointed at the table at all), fast-failing every episode on the
        # "Target Lost" kill-switch before a single meaningful action. When
        # use_classical_handoff is False, start from phase1_joint_pick_and_place.py's
        # joint_pre_pick instead -- the one pose in this codebase already documented as
        # "Above lego_red", i.e. actually oriented toward the workspace.
        #
        # shoulder_lift converted to the SAME raw-unwrapped convention as the retract
        # pose's own value above (-1.589 + 2*pi = 4.694, vs. the wrapped -1.589
        # joint_pre_pick itself uses) -- for the identical reason documented there: this
        # controller interpolates raw joint-space, not angle-wrapped, so commanding a
        # wrapped value far from wherever the raw tracking actually sits risks the same
        # near-full-revolution swing through the table already diagnosed and fixed once
        # for this exact joint. The other 5 joints are used as-is from joint_pre_pick --
        # only shoulder_lift was ever flagged as needing this raw/wrapped care.
        #
        # NOT independently re-verified against this project's current collision geometry
        # (Robotiq-140 + wrist camera additions postdate the script this was borrowed
        # from) -- _wait_for_safe_home()'s own collision-check-and-retry loop below is the
        # actual safety net here, not a substitute for it. First live attempt, not a
        # proven pose yet.
        if self.use_classical_handoff:
            self.home_action = np.array(
                [-0.4513, 4.7557, -0.2256, -1.2844, -0.1041, 1.4753], dtype=np.float32
            )
        else:
            self.home_action = np.array(
                [2.040, 4.6942, 1.991, -1.973, -1.571, 0.0], dtype=np.float32
            )

        # Reset Gripper to Fully Open -- BEFORE the home retreat move, not after. If a
        # prior episode ended with the gripper still closed around the block (timeout/
        # kill-switch fired without ever passing through step()'s grasp-check branch),
        # opening first releases it in place instead of dragging it through the full
        # home-retreat trajectory first -- confirmed live 2026-09-02: lego_red measured
        # climbing to z=1.46m (via the home retreat) before crashing back to its resting
        # height once the gripper finally opened under the old order. See
        # [[project-phase2-rl-status]].
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.0  # Open
        if self.gripper_client.wait_for_server(timeout_sec=0.1):
            self.gripper_client.send_goal_async(goal_msg)
        time.sleep(0.3)  # let the release actually happen before the home move starts

        self._wait_for_safe_home()

        # --- Auto-Respawn Lego ---
        # Bypasses ROS 2 and uses Gazebo's native service to teleport the object.
        # z=0.85 matches this model's own world-file spawn height (pick_and_place_demo.world,
        # a 5.7cm cube). Table's real collision surface is at z=0.80 (top_collision pose
        # z=0.78 + half of its 0.04m thickness); the earlier z=0.82 put the block 8.5mm
        # inside the table on every episode (fixed 2026-08-26, see
        # [[project-phase2-rl-status]]) -- z=0.85 keeps it comfortably above the surface.
        # Uses `ign service`, not `gz service`, and `ignition.msgs.*` types, not `gz.msgs.*`:
        # confirmed live that this stack's `gz` CLI can't reach the running `ign gazebo`
        # (Fortress) server at all, and Fortress's own service providers advertise
        # `ignition.msgs.Pose`/`ignition.msgs.Boolean` (pre-dates the gz.msgs rename in
        # later Gazebo versions) -- `gz.msgs.Pose` fails with "Unable to create request".
        # x/y updated 2026-09-03 to match the world file's real-hardware-matched spawn pose
        # (see pick_and_place_demo.world's lego_red <pose> comment) -- must stay in sync with it.
        # y corrected same day: world y = true_base_y (0.55) + real offset (-0.260) = 0.290,
        # not the raw offset alone (an earlier pass here used -0.260 directly, a bug).
        #
        # 2026-09-04: x/y now perturbed by a curriculum_level-scaled random lateral offset
        # (implementation spec section 1c) -- 0.802/0.290 are still the base point, not
        # replaced. curriculum_level=0 reproduces the exact old fixed spawn (radius 0).
        # Uniform-in-disk sampling (sqrt(u) for radius, not u directly) so points aren't
        # biased toward the center. See MIN/MAX_SPAWN_RADIUS_M in __init__ for the radius
        # bound's derivation and its "not yet live-verified" caveat.
        self.current_spawn_radius_m = (
            self.MIN_SPAWN_RADIUS_M
            + self.curriculum_level * (self.MAX_SPAWN_RADIUS_M - self.MIN_SPAWN_RADIUS_M)
        )
        spawn_r = self.current_spawn_radius_m * np.sqrt(np.random.uniform(0.0, 1.0))
        spawn_theta = np.random.uniform(0.0, 2 * np.pi)
        spawn_x = 0.802 + spawn_r * np.cos(spawn_theta)
        spawn_y = 0.290 + spawn_r * np.sin(spawn_theta)
        respawn_cmd = (
            "ign service -s /world/lego_sorting/set_pose --reqtype ignition.msgs.Pose "
            "--reptype ignition.msgs.Boolean --timeout 500 --req "
            f"'name: \"lego_red\", position: {{x: {spawn_x:.4f}, y: {spawn_y:.4f}, z: 0.85}}, "
            "orientation: {w: 1.0}'"
        )
        try:
            res = subprocess.run(respawn_cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                self.node.get_logger().warn(f"Auto-respawn failed! Gazebo responded: {res.stderr.strip()}")
        except Exception as e:
            self.node.get_logger().warn(f"Failed to execute Gazebo service sub-process: {e}")

        # Brief settle time after the lego teleport (arm is already verified home by
        # _wait_for_safe_home() above -- no spin_once() needed, background thread handles it).
        time.sleep(0.3)

        # Perception-mode freshness guard (2026-09-09, Vast.ai experimental track): without
        # this, the first _update_target_pose() call below could read a detection cached
        # from *before* the respawn above (self._perception_pose is never cleared on
        # respawn, and _perception_pose_stamp was captured but never actually checked
        # anywhere). Bounded wait, not a hard requirement -- background spin thread (same
        # one _wait_for_safe_home() relies on) keeps processing _perception_pose_callback
        # in parallel, so this just gives it a chance to catch up; falls through and uses
        # whatever's available (fresh or not) once the cap is hit rather than blocking
        # the episode indefinitely on a perception pipeline hiccup.
        if self.target_source == 'perception':
            respawn_time = self.node.get_clock().now()
            wait_deadline = time.time() + 1.0
            while time.time() < wait_deadline:
                if self._perception_pose_stamp is not None and self._perception_pose_stamp > respawn_time:
                    break
                time.sleep(0.05)

        # Update to physical lego position instead of theoretical random boxes
        self._update_target_pose()

        # Ground-truth-vs-YOLO diagnostic (2026-09-09): logged, not used to correct the
        # RL-visible target_pose above -- see _fetch_ground_truth_target_pose()'s docstring.
        # Once per episode (not per step) for the same reason the ground-truth path itself
        # only queries Gazebo once per episode -- see _update_target_pose()'s own note on
        # why a subprocess per training step would make training crawl.
        if self.target_source == 'perception':
            gt_pose = self._fetch_ground_truth_target_pose()
            if gt_pose is not None:
                self.last_ground_truth_pose = gt_pose
                self.last_perception_error_m = float(np.linalg.norm(self.target_pose - gt_pose))
                self.node.get_logger().info(
                    f"Perception-vs-ground-truth error this episode: "
                    f"{self.last_perception_error_m:.4f}m "
                    f"(YOLO {self.target_pose.tolist()}, GT {gt_pose.tolist()})"
                )

        # --- Phase A: classical handoff (hybrid planner+RL spec, Recommendation 3) ---
        # MoveIt/OMPL drives tool0 from the safe home pose to a pre-grasp standoff above the
        # target -- deterministic gross reach, already proven across 3 clean Phase 1 runs.
        # RL (Phase B, step() above) then starts every episode at that standoff and only has
        # to learn the final approach + grasp, not full-arm reachability. Logged explicitly
        # per episode (spec section 2) via both this info dict and the
        # Handoff_*/Handoff_OK/Standoff_M/Spawn_Radius_M hardware CSV columns.
        #
        # 2026-09-04: standoff is now curriculum_level-derived (implementation spec section
        # 1) instead of the fixed 5cm this project used for its entire history through
        # today -- see MIN/MAX_STANDOFF_M in __init__. curriculum_level=0 reproduces the
        # exact old fixed 5cm behavior.
        #
        # Pure-RL toggle (2026-09-09): skipped entirely when self.use_classical_handoff is
        # False -- RL then starts every episode at the home pose set above, not a pre-grasp
        # standoff, and controls the full reach itself. last_handoff_pose/last_handoff_ok
        # stay at their __init__ zero-init defaults in this case (CSV logging already
        # handles that gracefully -- these columns just always read the same as a fresh,
        # never-handed-off episode).
        if self.use_classical_handoff:
            self.current_standoff_m = (
                self.MIN_STANDOFF_M
                + self.curriculum_level * (self.MAX_STANDOFF_M - self.MIN_STANDOFF_M)
            )
            handoff_ok, handoff_pose = self.handoff_bridge.handoff_to_pregrasp(
                self.target_pose, standoff_m=self.current_standoff_m
            )
            self.last_handoff_pose = handoff_pose
            self.last_handoff_ok = handoff_ok
            if handoff_ok:
                self.node.get_logger().info(
                    f"Handoff OK -> pre-grasp [{handoff_pose[0]:.3f}, {handoff_pose[1]:.3f}, {handoff_pose[2]:.3f}]"
                )
            else:
                self.node.get_logger().warn(
                    "Classical handoff to pre-grasp did not verify -- starting RL episode from "
                    "wherever the arm actually ended up. step()'s kill-switches will catch a "
                    "genuinely unsafe state on the first step rather than this silently "
                    "pretending the handoff succeeded."
                )

        state = self._get_obs()

        # Potential-based distance shaping (implementation spec section 3) needs a real
        # starting point, not the leftover value from the previous episode -- the post-
        # handoff distance is what RL's first step() call actually shapes against.
        post_handoff_ee_pos = state[12:15]
        self._prev_dist = float(np.linalg.norm(post_handoff_ee_pos - self._active_goal_pose()))

        info = {
            'target_pose': self.target_pose,
            # self.last_handoff_pose/last_handoff_ok, not the local handoff_pose/handoff_ok
            # -- those only exist inside the `if self.use_classical_handoff:` block above,
            # UnboundLocalError'd here the moment that block is skipped (confirmed live).
            # self.last_* already holds the right value either way (zero-init defaults
            # when handoff is off, set from the local vars when it's on).
            'handoff_pose': self.last_handoff_pose,
            'handoff_ok': self.last_handoff_ok,
            'standoff_m': self.current_standoff_m,
            'spawn_radius_m': self.current_spawn_radius_m,
        }

        return state, info

    def render(self):
        pass

    def close(self):
        self._stop_gz_pose_subscriber()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)
