"""One-off diagnostic script (2026-09-04, not part of the RL training pipeline): checks
whether CONTACT_GAP_THRESHOLD in ur3e_env.py's step() actually separates genuine grasps from
misses, using ground-truth-labeled trials rather than trusting the constant. Run manually,
not via train_sac.py / launch files. See implementation spec follow-up: "verify with a quick
live check ... before trusting it at the top of a real training run."

Method: at curriculum_level=0.0 (fixed 5cm standoff, trivial reach -- isolates the gripper
mechanism from any reach-accuracy question), reposition tool0 directly via
MoveItHandoffBridge.handoff_to_pregrasp() (bypassing step()'s RL action interface, since we
want precise, ground-truth-labeled positions, not a scripted joint-space approach) to either:
  - a near-zero standoff (genuine grasp: fingers should close around the lego), or
  - a large standoff (miss: fingers close on empty air, clear of the lego)
then command the gripper closed, wait for it to settle, and record the achieved-vs-commanded
joint-gap for each labeled trial.
"""
import sys
import time
import random
import rclpy

from ur3e_rl.ur3e_env import UR3eEnv
from control_msgs.action import GripperCommand

N_GRASP_TRIALS = 5
N_MISS_TRIALS = 5
GRASP_STANDOFF_M = 0.02    # fingertip ~2cm above the lego's ground-truth position -- close
                           # enough that closing fingers should contact it, but not the exact
                           # 0.0 edge case (first diagnostic run's standoff=0.0 trials failed
                           # handoff 3/3 every time -- possibly a genuine near-table/IK-edge
                           # difficulty at that exact height, not conclusively diagnosed;
                           # backing off slightly rather than re-fighting the same value).
MISS_STANDOFF_M = 0.09     # fingertip ~9cm above the lego -- clear air, guaranteed miss
GRIPPER_CLOSE_CMD = 0.8    # matches step()'s fully-closed mapping
GRIPPER_EFFORT = 100.0     # matches current ur3e_env.py step()
SETTLE_POLL_INTERVAL_S = 0.3
SETTLE_MAX_WAIT_S = 3.0
SETTLE_STABLE_DELTA = 0.005  # meters of finger_joint travel considered "stopped moving"


def send_gripper(env, position, effort=GRIPPER_EFFORT):
    goal_msg = GripperCommand.Goal()
    goal_msg.command.position = position
    goal_msg.command.max_effort = effort
    if env.gripper_client.wait_for_server(timeout_sec=1.0):
        env.gripper_client.send_goal_async(goal_msg)
    else:
        print("  [WARN] gripper action server not available")


def wait_for_gripper_settle(env):
    """Poll current_gripper_pos until it stops changing (contact/stall) or timeout."""
    last = getattr(env, 'current_gripper_pos', 0.0)
    elapsed = 0.0
    while elapsed < SETTLE_MAX_WAIT_S:
        time.sleep(SETTLE_POLL_INTERVAL_S)
        elapsed += SETTLE_POLL_INTERVAL_S
        current = getattr(env, 'current_gripper_pos', 0.0)
        if abs(current - last) < SETTLE_STABLE_DELTA:
            return current
        last = current
    return last


def run_trial(env, label, standoff_m, trial_num, total):
    # Full env.reset() between every trial (2026-09-04, third attempt -- the first version
    # chained repositions from wherever the previous trial left the arm with no reset at
    # all; the second version used only _wait_for_safe_home(), which still failed handoff on
    # 8/10 trials). reset() is this project's actual proven-reliable path: full home
    # retreat, a fresh lego respawn, a fresh ground-truth fetch, and its own handoff to the
    # standard 0.05 standoff (curriculum_level is 0.0 throughout this script, so this always
    # reproduces the exact default behavior already exercised cleanly in the earlier smoke
    # test). Only after that succeeds do we take one additional, smaller repositioning step
    # to the actual test standoff -- a small delta from an already-good pose should be an
    # easier planning problem for OMPL than one big jump from the folded-up home
    # configuration directly to a tight target, which is what the second attempt was doing.
    print(f"Trial {trial_num}/{total} ({label}, standoff={standoff_m}): "
          "full reset()...", flush=True)
    env.reset()
    print(f"  reset() done -- default handoff_ok={env.last_handoff_ok}", flush=True)

    ok, pose = env.handoff_bridge.handoff_to_pregrasp(env.target_pose, standoff_m=standoff_m)
    if not ok:
        print(f"  [WARN] repositioning handoff did not verify for this trial "
              f"(standoff={standoff_m}) -- recording anyway, flagged as unverified position",
              flush=True)

    time.sleep(0.5)
    send_gripper(env, GRIPPER_CLOSE_CMD)
    achieved = wait_for_gripper_settle(env)
    gap = GRIPPER_CLOSE_CMD - achieved
    print(f"  [{label}] standoff={standoff_m:.3f} reset_handoff_ok={env.last_handoff_ok} "
          f"reposition_ok={ok} commanded={GRIPPER_CLOSE_CMD:.3f} achieved={achieved:.4f} "
          f"gap={gap:.4f}", flush=True)
    return {"label": label, "standoff_m": standoff_m, "handoff_ok": ok,
            "reset_handoff_ok": env.last_handoff_ok, "achieved": achieved, "gap": gap}


def main():
    print("Constructing env (curriculum_level stays 0.0 -- not touched by this script)...",
          flush=True)
    env = UR3eEnv(env_id=0, namespace='')
    # No separate warm-up reset() here -- trial 1 below calls env.reset() itself as the
    # first step of every trial now (see run_trial()).

    trials = (["grasp"] * N_GRASP_TRIALS) + (["miss"] * N_MISS_TRIALS)
    random.shuffle(trials)

    results = []
    for i, label in enumerate(trials):
        standoff = GRASP_STANDOFF_M if label == "grasp" else MISS_STANDOFF_M
        results.append(run_trial(env, label, standoff, i + 1, len(trials)))

    env.close()

    grasp_gaps = sorted(r["gap"] for r in results if r["label"] == "grasp")
    miss_gaps = sorted(r["gap"] for r in results if r["label"] == "miss")
    print("\n=== RESULTS ===")
    print(f"Genuine-grasp gaps (n={len(grasp_gaps)}): {[round(g, 4) for g in grasp_gaps]}",
          flush=True)
    print(f"Miss gaps         (n={len(miss_gaps)}): {[round(g, 4) for g in miss_gaps]}",
          flush=True)
    if grasp_gaps and miss_gaps:
        print(f"Genuine-grasp gap range: [{min(grasp_gaps):.4f}, {max(grasp_gaps):.4f}]",
              flush=True)
        print(f"Miss gap range:          [{min(miss_gaps):.4f}, {max(miss_gaps):.4f}]",
              flush=True)
        overlap = max(min(grasp_gaps), min(miss_gaps)) <= min(max(grasp_gaps), max(miss_gaps))
        print(f"Distributions overlap: {overlap}", flush=True)
        print(f"Current CONTACT_GAP_THRESHOLD = 0.05 -- "
              f"grasps above it: {sum(g > 0.05 for g in grasp_gaps)}/{len(grasp_gaps)}, "
              f"misses below it: {sum(g <= 0.05 for g in miss_gaps)}/{len(miss_gaps)}",
              flush=True)
    print("\nDone. This script made no training-affecting changes -- constants live in "
          "ur3e_env.py and must be edited there if this data says the threshold is wrong.")


if __name__ == '__main__':
    if not rclpy.ok():
        rclpy.init()
    try:
        main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
