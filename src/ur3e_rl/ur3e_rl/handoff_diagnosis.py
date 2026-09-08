"""Direct root-cause diagnosis for handoff failures (2026-09-05). Reimplements
MoveItHandoffBridge's handoff logic with full instrumentation: prints the REAL MoveGroup
result.error_code (planning vs. execution vs. something else) and the actual TF-measured
position error on every single attempt, success or failure. This is the thing passive
/rosout watching couldn't get us. Not part of the training pipeline -- run manually, only
when nothing else is using MoveGroup (this sends its own goals directly).
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import JointState

# MoveItErrorCodes reference (moveit_msgs/msg/MoveItErrorCodes.msg)
ERROR_CODE_NAMES = {
    1: "SUCCESS", -1: "FAILURE", -2: "PLANNING_FAILED", -3: "INVALID_MOTION_PLAN",
    -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE", -5: "CONTROL_FAILED",
    -6: "UNABLE_TO_AQUIRE_SENSOR_DATA", -7: "TIMED_OUT", -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS", -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS", -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME", -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE", -18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE", -31: "NO_IK_SOLUTION",
}

TARGET = [0.802, -0.260, 0.230]  # matches the real handoff target seen live
N_TRIALS = 15


class DiagNode(Node):
    def __init__(self):
        super().__init__('handoff_diag_node')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_client = ActionClient(self, MoveGroup, 'move_action')
        self.latest_joint_state = None
        self.create_subscription(JointState, 'joint_states', self._js_cb, 10)

    def _js_cb(self, msg):
        self.latest_joint_state = msg


def build_goal(target, joint_state):
    q = [0.0, 1.0, 0.0, 0.0]
    pcm = PositionConstraint()
    pcm.header.frame_id = "base_link"
    pcm.link_name = "tool0"
    pcm.weight = 1.0
    box = BoundingVolume()
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [0.02, 0.02, 0.02]
    box.primitives.append(primitive)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = target
    pose.orientation.w = 1.0
    box.primitive_poses.append(pose)
    pcm.constraint_region = box

    ocm = OrientationConstraint()
    ocm.header.frame_id = "base_link"
    ocm.link_name = "tool0"
    ocm.orientation.x, ocm.orientation.y, ocm.orientation.z, ocm.orientation.w = q
    ocm.absolute_x_axis_tolerance = 0.05
    ocm.absolute_y_axis_tolerance = 0.05
    ocm.absolute_z_axis_tolerance = 3.15
    ocm.weight = 1.0

    constraints = Constraints()
    constraints.position_constraints.append(pcm)
    constraints.orientation_constraints.append(ocm)

    goal = MoveGroup.Goal()
    goal.request.group_name = "arm"
    goal.request.pipeline_id = "ompl"
    goal.request.planner_id = "RRTConnectkConfigDefault"
    goal.request.num_planning_attempts = 30
    goal.request.allowed_planning_time = 10.0
    goal.request.max_velocity_scaling_factor = 0.3
    goal.request.max_acceleration_scaling_factor = 0.3
    goal.request.goal_constraints = [constraints]
    if joint_state is not None:
        goal.request.start_state.joint_state = joint_state
        goal.request.start_state.is_diff = False
    else:
        goal.request.start_state.is_diff = True
    return goal


JOINT_LIMITS = {  # matches ur3e_env.py's joint_pos_min/max (margin already applied there)
    'shoulder_pan_joint': (-6.28, 6.28), 'shoulder_lift_joint': (-6.28, 6.28),
    'elbow_joint': (-3.09, 3.09), 'wrist_1_joint': (-6.28, 6.28),
    'wrist_2_joint': (-6.28, 6.28), 'wrist_3_joint': (-6.28, 6.28),
}


def check_start_state(joint_state):
    if joint_state is None:
        return "NO JOINT STATE AVAILABLE"
    issues = []
    for name, pos in zip(joint_state.name, joint_state.position):
        limits = JOINT_LIMITS.get(name)
        if limits and not (limits[0] <= pos <= limits[1]):
            issues.append(f"{name}={pos:.3f} OUT OF RANGE {limits}")
    return "; ".join(issues) if issues else "within limits"


def run_trial(node, trial_num):
    start_check = check_start_state(node.latest_joint_state)
    if node.latest_joint_state is not None:
        joints_str = ", ".join(f"{n}={p:.3f}" for n, p in
                                zip(node.latest_joint_state.name, node.latest_joint_state.position))
        print(f"Trial {trial_num} start joints: {joints_str}", flush=True)
    print(f"Trial {trial_num} start-state check: {start_check}", flush=True)
    goal = build_goal(TARGET, node.latest_joint_state)
    t0 = time.time()

    if not node.move_client.wait_for_server(timeout_sec=5.0):
        print(f"Trial {trial_num}: MoveGroup server unavailable", flush=True)
        return

    future = node.move_client.send_goal_async(goal)
    while not future.done():
        time.sleep(0.05)
    handle = future.result()
    if not handle.accepted:
        print(f"Trial {trial_num}: GOAL REJECTED (not even accepted for planning)", flush=True)
        return

    res_future = handle.get_result_async()
    while not res_future.done():
        time.sleep(0.05)
    result = res_future.result().result
    code = result.error_code.val
    code_name = ERROR_CODE_NAMES.get(code, f"UNKNOWN({code})")
    plan_time = time.time() - t0

    time.sleep(0.3)
    try:
        t = node.tf_buffer.lookup_transform(
            'base_link', 'tool0', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5))
        actual = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        deltas = [abs(actual[i] - TARGET[i]) for i in range(3)]
        max_delta = max(deltas)
    except Exception as e:
        max_delta = None
        deltas = None

    verify_pass = (max_delta is not None and max_delta <= 0.025)
    print(f"Trial {trial_num}: error_code={code}({code_name}) plan_time={plan_time:.1f}s "
          f"tf_max_delta={max_delta if max_delta is None else round(max_delta,4)} "
          f"deltas={deltas if deltas is None else [round(d,4) for d in deltas]} "
          f"verify_pass={verify_pass}", flush=True)


def main():
    node = DiagNode()
    from rclpy.executors import SingleThreadedExecutor
    import threading
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(2.0)  # let joint_states/tf populate
    for i in range(1, N_TRIALS + 1):
        run_trial(node, i)
        time.sleep(1.0)

    node.destroy_node()


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
