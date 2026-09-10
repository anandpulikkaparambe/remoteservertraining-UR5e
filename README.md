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
  original `run_docker.sh` (which assumed a local display). One `docker run` launches
  `NUM_ENVS` headless Gazebo+MoveIt instances (namespaces `''`, `env1`, `env2`, ...) and
  ONE `train_sac`/`train_ppo --num-envs=NUM_ENVS` process, using this project's
  existing `SubprocVecEnv` support so all instances feed a **single shared policy**,
  not `NUM_ENVS` separate/independent training runs. **Not yet run on a real Vast.ai
  instance** -- the readiness waits in the entrypoint are placeholders, not verified
  polls. First run is a real test of this script, not a known-working path.

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

## Running it

One command starts everything -- `NUM_ENVS` Gazebo instances, all feeding one policy:

```bash
ALGO=sac NUM_ENVS=4 TARGET_SOURCE=ground_truth CPUS=16 MEMORY=10g ./run_vastai.sh
docker logs -f ur3e_vastai_club
```

`NUM_ENVS` instances collecting experience in parallel is what actually shortens
wall-clock time to a given step count -- it does **not** run `NUM_ENVS` separate
policies. If you instead want independent parallel *attempts* (e.g. to compare seeds),
run this script multiple times with `NUM_ENVS=1` and a distinct `CHECKPOINT_DIR` each
time -- that gives you unrelated single-instance runs, on purpose.

Sizing (`CPUS`/`MEMORY` for the whole club, and how many instances a given listing can
hold) is discussed in this fork's originating conversation, not re-derived here --
summary: headless, one Gazebo+MoveIt instance costs roughly 3 CPU cores / 1.2GB RAM,
before adding YOLO inference; the defaults above (`NUM_ENVS=4`, `CPUS=16`,
`MEMORY=10g`) assume a ~20-vCPU host -- adjust all three together if the listing
differs. Check the listing's actual total RAM (not just VRAM) before committing to a
`NUM_ENVS` count -- RAM is usually the binding constraint, not GPU VRAM, for this
workload.

## Renting and setting up a Vast.ai instance (console walkthrough)

A Vast.ai instance **is itself a Docker container** -- you don't get a bare VM you then
run `docker build`/`docker run` inside. That means `run_vastai.sh`'s own `docker build`
and `docker run` calls do **not** work unmodified inside a rented instance (no reliable
Docker-in-Docker across hosts). Instead: rent an instance using this repo's own base
image directly, then run the workspace setup and `vastai_train_entrypoint.sh` steps
straight inside it. Steps below, worked out and verified against the live console on
2026-09-08:

1. **Pick a template that's a plain Docker container, not a "VM"/"Desktop" one.**
   Templates with "(VM)" in the name (e.g. "Ubuntu Desktop (VM)") boot a full virtual
   machine disk and ignore a custom image path entered into their config -- there's no
   "instance type" toggle to fix this from within that template. Start instead from an
   unambiguous Docker template like **"PyTorch (Vast)"** (prefer this over "PyTorch
   (NVIDIA)", which bakes in CUDA-specific docker options this CPU-only project doesn't
   need) and overwrite its image field.
2. **Image**: `osrf/ros:humble-desktop-full` (same as this repo's `Dockerfile`'s `FROM`
   -- avoids re-downloading/rebuilding ROS+MoveIt+Gazebo from scratch). Docker
   repository type: **Public** (it's a public Docker Hub image -- unrelated to this
   GitHub repo being private).
3. **Launch Mode**: "Interactive shell server" (SSH). Not Jupyter, not entrypoint.
4. **Disk space**: >=30GB. Rough budget: base image ~6-8GB, apt installs ~1-2GB, pip
   installs ~1.5GB, `colcon build` output ~1-2GB, repo ~0.1GB, checkpoints/logs buffer
   ~2-5GB.
5. Save the template (private visibility is fine -- that only affects whether other
   Vast.ai users can see it, not functionality).
6. **On the Search/listing page**, filter: rental type **Interruptible** (30-50%+
   cheaper than On-Demand, safe here since `vastai_train_entrypoint.sh` supports
   `--resume` from a checkpoint), **CPU cores >=16**, **RAM >=16GB**, sort by price
   ascending. Read each listing's actual CPU/RAM row before renting (format is
   `offered/total`, e.g. `18.7/56 CPU` means you get 18.7 of that host's 56 threads) --
   don't trust the filter sliders alone, several cheap listings offer far less than
   their host's total. Ignore GPU/VRAM/TFLOPS entirely -- this workload never touches
   the GPU (see `train_sac.py`'s hardcoded `device="cpu"` and the CPU-only torch install
   in `Dockerfile`).
7. Rent, wait for status **"running"** on the Instances tab, then use the
   connect/terminal icon to get the SSH command.
8. **Sanity-check before doing anything else**:
   ```bash
   ls /.dockerenv && echo "container confirmed"
   cat /etc/os-release
   source /opt/ros/humble/setup.bash && ros2 --version
   nproc
   free -h
   df -h /
   ```
9. **Clone this repo via a deploy key** (scoped read-only access, safer than a personal
   token on a rented/ephemeral box):
   ```bash
   ssh-keygen -t ed25519 -C "vastai" -f ~/.ssh/id_ed25519 -N ""
   cat ~/.ssh/id_ed25519.pub
   ```
   Add that public key as a read-only **Deploy key** under this repo's GitHub Settings
   -> Deploy keys, then:
   ```bash
   git clone git@github.com:anandpulikkaparambe/remoteservertraining-UR5e.git /ros2_ws
   cd /ros2_ws
   ```
10. **Build the workspace** (mirrors `Dockerfile`'s `RUN` steps, since there's no nested
    `docker build` here):
    ```bash
    apt-get update && apt-get install -y python3-pip ros-humble-moveit \
        ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
    pip3 install --no-cache-dir -U packaging
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    pip3 install --no-cache-dir --timeout 120 --retries 10 \
        ultralytics opencv-python "numpy<1.24" gymnasium stable_baselines3 tensorboard
    mv /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml \
       /opt/ros/humble/share/moveit_planners_chomp/chomp_interface_plugin_description.xml.disabled
    apt-get update && rosdep update && \
        rosdep install --from-paths src --ignore-src -r -y --skip-keys warehouse_ros_mongo
    source /opt/ros/humble/setup.bash && colcon build
    chmod +x vastai_train_entrypoint.sh
    ```
11. **Launch training directly** -- skip `run_vastai.sh`'s outer `docker run`, you're
    already inside the container Vast.ai gave you:
    ```bash
    mkdir -p rl_logs
    ALGO=sac NUM_ENVS=2 RUN_YOLO=false RESUME= ./vastai_train_entrypoint.sh sac 2 false ""
    ```
    Start with **`NUM_ENVS=2`, not this README's `NUM_ENVS=4` default**, for the very
    first run on a fresh instance -- the whole multi-instance path is unvalidated (see
    "Known, unresolved blockers" above), and a smaller club is cheaper to debug if the
    fixed-sleep readiness wait in `vastai_train_entrypoint.sh` turns out too short.
12. **Watch progress**: `tail -f rl_logs/train_sac_club.log` and
    `rl_logs/gazebo_instance*.log`.
13. **Pull checkpoints back periodically** (from your local machine, `pip install
    vastai` for the CLI) so a preempted Interruptible instance doesn't lose progress:
    ```bash
    vastai copy <instance_id>:/ros2_ws/rl_logs ./rl_logs_vastai
    ```
14. **Destroy the instance from the console when done** -- billed per second while
    running.

## What this fork does NOT attempt

- It does not fix the handoff-reliability blocker.
- It does not add or validate a real hand-eye calibration.
- It does not run or characterize a perception-error calibration test.

Those are the three items that would actually move this from "designed" to "trustworthy
for a real result" -- flagged here rather than silently assumed done.
