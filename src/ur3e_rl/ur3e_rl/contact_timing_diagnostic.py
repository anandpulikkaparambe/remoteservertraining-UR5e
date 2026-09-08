"""Follow-up to contact_threshold_diagnostic.py (2026-09-04): that script gave the gripper
up to 3s (settle-polled) to reach its final position before reading the achieved-vs-commanded
gap. Production step() only waits sim_step_time=0.1s (one RL tick) before checking contact --
if the finger hasn't traveled far enough in that single tick to show real separation between a
grasp and a miss, a genuine grasp closing in the same step dist first crosses under 0.05 could
get misread as "no contact" and killed before it ever had a chance to settle.

This script answers that directly: for each labeled trial, sample achieved_gripper_pos at
every 0.1s tick (matching sim_step_time exactly) for N_TICKS ticks, and report the gap
time-course -- not just a single answer, but how many ticks it actually takes for genuine
grasps and misses to become distinguishable, so a grace-period fix (if needed) can be sized
correctly rather than guessed.

Not part of the RL training pipeline -- run manually.
"""
import time
import random
import rclpy

from ur3e_rl.ur3e_env import UR3eEnv
from control_msgs.action import GripperCommand

N_GRASP_TRIALS = 4
N_MISS_TRIALS = 4
GRASP_STANDOFF_M = 0.02
MISS_STANDOFF_M = 0.09
GRIPPER_CLOSE_CMD = 0.8
GRIPPER_EFFORT = 100.0
TICK_S = 0.1          # matches UR3eEnv.sim_step_time exactly
N_TICKS = 20           # 2.0s total -- well past where the settle-polled diagnostic converged
CURRENT_THRESHOLD = 0.13  # matches ur3e_env.py's just-recalibrated CONTACT_GAP_THRESHOLD


def send_gripper(env, position, effort=GRIPPER_EFFORT):
    goal_msg = GripperCommand.Goal()
    goal_msg.command.position = position
    goal_msg.command.max_effort = effort
    if env.gripper_client.wait_for_server(timeout_sec=1.0):
        env.gripper_client.send_goal_async(goal_msg)
    else:
        print("  [WARN] gripper action server not available", flush=True)


def run_trial(env, label, standoff_m, trial_num, total):
    print(f"Trial {trial_num}/{total} ({label}, standoff={standoff_m}): full reset()...",
          flush=True)
    env.reset()
    reset_ok = env.last_handoff_ok
    ok, pose = env.handoff_bridge.handoff_to_pregrasp(env.target_pose, standoff_m=standoff_m)
    print(f"  reset_handoff_ok={reset_ok} reposition_ok={ok}", flush=True)
    time.sleep(0.5)

    # Command close ONCE (fire-and-forget, exactly like step() does), then sample the
    # achieved position every TICK_S, exactly matching how many real RL steps would elapse
    # if the policy just held the same "closed" action every step.
    send_gripper(env, GRIPPER_CLOSE_CMD)
    gaps = []
    for tick in range(1, N_TICKS + 1):
        time.sleep(TICK_S)
        achieved = float(getattr(env, 'current_gripper_pos', 0.0))
        gap = GRIPPER_CLOSE_CMD - achieved
        gaps.append(gap)
    tick_str = " ".join(f"{g:.3f}" for g in gaps)
    print(f"  [{label}] standoff={standoff_m:.3f} gaps per tick (0.1s each): {tick_str}",
          flush=True)
    # First tick at which gap crosses the *current* threshold and STAYS crossed (no flicker
    # back under) -- this is what a real single-step check would have seen if it fired at
    # that tick.
    first_stable_cross = None
    for i in range(len(gaps)):
        if all(g > CURRENT_THRESHOLD for g in gaps[i:]):
            first_stable_cross = i + 1
            break
    print(f"  [{label}] first tick where gap>{CURRENT_THRESHOLD} and stays there: "
          f"{first_stable_cross}", flush=True)
    return {"label": label, "standoff_m": standoff_m, "gaps": gaps,
            "first_stable_cross": first_stable_cross}


def main():
    print("Constructing env (curriculum_level stays 0.0)...", flush=True)
    env = UR3eEnv(env_id=0, namespace='')

    trials = (["grasp"] * N_GRASP_TRIALS) + (["miss"] * N_MISS_TRIALS)
    random.shuffle(trials)

    results = []
    for i, label in enumerate(trials):
        standoff = GRASP_STANDOFF_M if label == "grasp" else MISS_STANDOFF_M
        results.append(run_trial(env, label, standoff, i + 1, len(trials)))

    env.close()

    print("\n=== RESULTS ===", flush=True)
    print(f"Tick spacing: {TICK_S}s (matches sim_step_time). "
          f"Current CONTACT_GAP_THRESHOLD = {CURRENT_THRESHOLD}", flush=True)
    for r in results:
        print(f"{r['label']:6s} standoff={r['standoff_m']:.3f} "
              f"tick1_gap={r['gaps'][0]:.4f} tick3_gap={r['gaps'][2]:.4f} "
              f"final_gap={r['gaps'][-1]:.4f} "
              f"first_stable_cross_tick={r['first_stable_cross']}", flush=True)

    grasp_tick1 = [r['gaps'][0] for r in results if r['label'] == 'grasp']
    miss_tick1 = [r['gaps'][0] for r in results if r['label'] == 'miss']
    print(f"\nAt tick 1 (0.1s, matches a single real RL step) -- "
          f"grasp gaps: {[round(g,4) for g in grasp_tick1]}, "
          f"miss gaps: {[round(g,4) for g in miss_tick1]}", flush=True)
    grasp_cross_1 = sum(1 for r in results if r['label'] == 'grasp' and r['first_stable_cross'] == 1)
    miss_never_cross = sum(1 for r in results if r['label'] == 'miss' and r['first_stable_cross'] is None)
    print(f"Grasps already correctly detected at tick 1: {grasp_cross_1}/{len(grasp_tick1)}",
          flush=True)
    print(f"Misses that never cross the threshold (correct): {miss_never_cross}/{len(miss_tick1)}",
          flush=True)
    print("\nDone.", flush=True)


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
