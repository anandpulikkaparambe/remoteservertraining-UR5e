"""Handoff reliability test (2026-09-05): measures handoff_to_pregrasp() success rate over
N full env.reset() cycles, to compare against the known pre-fix baseline (~55-62%, see
project_phase2_rl_status memory) after loosening the wrist-yaw orientation tolerance in
moveit_bridge.py. Not part of the training pipeline -- run manually.
"""
import time
import rclpy
from ur3e_rl.ur3e_env import UR3eEnv

N_TRIALS = 20


def main():
    print("Constructing env...", flush=True)
    env = UR3eEnv(env_id=0, namespace='')

    results = []
    for i in range(N_TRIALS):
        t0 = time.time()
        env.reset()
        elapsed = time.time() - t0
        ok = env.last_handoff_ok
        results.append(ok)
        print(f"Trial {i+1}/{N_TRIALS}: handoff_ok={ok} ({elapsed:.1f}s)", flush=True)

    env.close()
    n_ok = sum(results)
    print(f"\n=== RESULTS ===", flush=True)
    print(f"Handoff success: {n_ok}/{N_TRIALS} = {100*n_ok/N_TRIALS:.1f}%", flush=True)
    print("Pre-fix baseline (yesterday, same method): ~55-62%", flush=True)


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
