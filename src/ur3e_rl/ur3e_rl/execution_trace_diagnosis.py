"""One-off diagnosis (2026-09-06) for the direct-IK handoff bypass: instead of checking TF
once after a fixed sleep, this polls TF and /joint_states every second throughout a single
commanded move (duration 8s, then 7s extra margin) to see whether the arm is actually
converging toward the commanded joints over time, stalling partway, or not moving at all.
Not part of the training pipeline -- run manually.
"""
import time
import rclpy
from ur3e_rl.ur3e_env import UR3eEnv


def main():
    env = UR3eEnv(env_id=0, namespace='')
    bridge = env.handoff_bridge
    print("Running a real reset() first to get a real target_pose and exercise the "
          "existing (currently-failing) handoff path once...", flush=True)
    env.reset()
    target = list(env.target_pose)
    pregrasp = [target[0], target[1], target[2] + bridge.GRIPPER_OFFSET_M + 0.05]
    print(f"pregrasp target = {pregrasp}", flush=True)

    joints = bridge._compute_ik(pregrasp, "base_link")
    print(f"IK solution = {joints}", flush=True)
    if joints is None:
        print("IK FAILED, aborting")
        env.close()
        return

    js = env.latest_joint_state
    current = dict(zip(js.name, js.position)) if js else {}
    for n in bridge.ARM_JOINT_NAMES:
        print(f"  {n}: current={current.get(n)} target={joints.get(n)} "
              f"delta={abs(joints.get(n,0)-current.get(n,0))}", flush=True)

    duration_sec = 8.0
    print(f"\nPublishing single-waypoint trajectory with duration={duration_sec}s ...", flush=True)
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    traj = JointTrajectory()
    traj.joint_names = bridge.ARM_JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [joints[n] for n in bridge.ARM_JOINT_NAMES]
    point.time_from_start.sec = int(duration_sec)
    point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
    traj.points = [point]
    bridge._trajectory_pub.publish(traj)

    t0 = time.time()
    for i in range(15):
        time.sleep(1.0)
        elapsed = time.time() - t0
        try:
            t = env.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time(),
                                                timeout=rclpy.duration.Duration(seconds=0.3))
            actual = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            deltas = [abs(actual[j] - pregrasp[j]) for j in range(3)]
        except Exception as e:
            actual = None
            deltas = None
        js2 = env.latest_joint_state
        cur2 = dict(zip(js2.name, js2.position)) if js2 else {}
        joint_deltas = {n: round(abs(joints[n] - cur2.get(n, 0)), 4) for n in bridge.ARM_JOINT_NAMES}
        print(f"t={elapsed:5.1f}s  tf_actual={actual}  tf_delta={deltas}  joint_deltas={joint_deltas}",
              flush=True)

    env.close()


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
