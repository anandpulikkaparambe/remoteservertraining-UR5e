# UR5e RL Training -- Vast.ai / YOLO-integration fork

Standalone fork of the main `ur5e_working_reference` workspace, built 2026-09-08 to run
headless training (SAC/PPO) on a rented Vast.ai GPU instance, in parallel with the local
`ur3e_train` container, and with an opt-in path to source the target position from the
trained YOLO detector instead of Gazebo ground truth.

**Read this whole file before trusting a training run from this fork** -- several pieces
here are new and have not been live-validated end-to-end (marked below).

## What's different from the main repo

- `.gitignore` added -- the main repo has `build/`/`install/`/`log/` committed to git
  (1200+ files, most of its 2.3GB). This fork rebuilds from source via `colcon build`
  instead.
- `src/ur3e_rl/ur3e_rl/ur3e_env.py`: new `UR3E_TARGET_SOURCE` env var (`ground_truth`
  default / `perception`), new `UR3E_TARGET_NOISE_STD_M` env var (default `0.0`, no
  behavior change). See the `_update_target_pose_from_perception` docstring for the known
  gaps in perception mode -- **do not use `TARGET_SOURCE=perception` for a real result
  yet**, only for isolated testing of the plumbing.
- `src/ur_gazebo/launch/ur.gazebo.launch.py`: the `yolo_detector` node launch was fully
  commented out before (the `run_yolo` arg existed but did nothing) -- re-enabled here.
- `run_vastai.sh` / `vastai_train_entrypoint.sh`: new, headless equivalents of the
  original `run_docker.sh` (which assumed a local display). **Not yet run on a real
  Vast.ai instance** -- the 60s readiness wait in the entrypoint is a placeholder, not a
  verified poll. First run is a real test of this script, not a known-working path.

## Known, unresolved blockers (carried over from the main repo, not fixed here)

1. **Handoff reliability sits at ~55-65%**, below the 90% gate curriculum advancement
   needs. Three targeted OMPL fixes and one full IK-bypass rewrite were tried; none
   cleared it. Training under `TARGET_SOURCE=ground_truth` (default) will very likely
   still hit this ceiling.
2. **No hand-eye calibration exists** for perception mode. `yolo_detector.py`'s
   extrinsics assume a fixed overhead camera; the real rig's camera is wrist-mounted. Do
   not deploy `TARGET_SOURCE=perception` results to real hardware without fixing this
   first.
3. **No perception-error characterization** (mean/std of YOLO-estimated vs. true
   position) exists, so `UR3E_TARGET_NOISE_STD_M` has no measured value to set yet --
   leave at `0.0` (ground-truth-equivalent) until that calibration test is run.

See the main repo's `results/rl_training_comparison/THESIS_RESULTS_REPORT.md` for the
full picture (Table D.2 is explicitly pending on these same items).

## Running one instance

```bash
ALGO=sac INSTANCE_ID=0 TARGET_SOURCE=ground_truth CPUS=4 MEMORY=6g ./run_vastai.sh
```

## Running several in parallel

Give each a distinct `INSTANCE_ID` (this also derives a distinct `NAMESPACE`/
`GZ_PARTITION` so they don't cross-talk over ROS/gz-transport):

```bash
for i in 0 1 2 3; do
  INSTANCE_ID=$i CPUS=4 MEMORY=6g ./run_vastai.sh
done
```

Sizing (`CPUS`/`MEMORY` per instance, and how many instances a given listing can hold)
is discussed in this fork's originating conversation, not re-derived here -- summary:
headless, one instance costs roughly 3 CPU cores / 1.2GB RAM for Gazebo+MoveIt alone,
before adding YOLO inference. Check the listing's actual total RAM (not just VRAM)
before committing to an instance count -- that number is usually the binding constraint,
not GPU VRAM, for this workload.

## What this fork does NOT attempt

- It does not fix the handoff-reliability blocker.
- It does not add or validate a real hand-eye calibration.
- It does not run or characterize a perception-error calibration test.

Those are the three items that would actually move this from "designed" to "trustworthy
for a real result" -- flagged here rather than silently assumed done.
