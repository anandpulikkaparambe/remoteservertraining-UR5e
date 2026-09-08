"""One-off diagnosis for the direct-IK handoff bypass (2026-09-06): runs a real env.reset()
(so the actual handoff path is exercised, not a hand-rolled substitute), then prints the IK
solution, achieved joint state, and TF delta to find exactly where it's landing wrong. Not
part of the training pipeline -- run manually.
"""
import time
import rclpy
from ur3e_rl.ur3e_env import UR3eEnv


def main():
    env = UR3eEnv(env_id=0, namespace='')
    print("Running a real reset() (exercises the actual handoff path)...", flush=True)
    env.reset()
    print(f"\nlast_handoff_ok={env.last_handoff_ok} last_handoff_pose={env.last_handoff_pose}",
          flush=True)

    bridge = env.handoff_bridge
    target = env.last_handoff_pose

    print(f"\nRecomputing IK for the same target {target} to inspect the solution...",
          flush=True)
    joints = bridge._compute_ik(target, "base_link")
    print(f"IK result: {joints}", flush=True)

    print("\nCurrent joint state right now:", flush=True)
    js = env.latest_joint_state
    current = dict(zip(js.name, js.position)) if js else {}
    for name in bridge.ARM_JOINT_NAMES:
        cmd = joints.get(name) if joints else None
        cur = current.get(name)
        diff = None if (cmd is None or cur is None) else abs(cmd - cur)
        print(f"  {name}: ik_solution={cmd}, actual_current={cur}, diff={diff}", flush=True)

    print("\nTF check against the handoff target:", flush=True)
    try:
        t = env.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time(),
                                            timeout=rclpy.duration.Duration(seconds=0.5))
        actual = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        deltas = [abs(actual[i] - target[i]) for i in range(3)]
        print(f"  target={target}", flush=True)
        print(f"  actual_tool0={actual}", flush=True)
        print(f"  deltas={deltas}", flush=True)
    except Exception as e:
        print(f"  TF lookup failed: {e}", flush=True)

    env.close()


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
