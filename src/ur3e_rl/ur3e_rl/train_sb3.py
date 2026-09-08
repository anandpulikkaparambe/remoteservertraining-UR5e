import os
import rclpy
import argparse
import collections
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ur3e_rl.ur3e_env import UR3eEnv

class CurriculumCallback(BaseCallback):
    """
    Custom callback for conditionally increasing the curriculum.
    Ported from train_sac.py's success-triggered version (see
    [[project-phase2-rl-status]]): the original PPO callback advanced
    purely on training progress (num_timesteps / total_timesteps), with
    no regard for whether the policy was actually succeeding.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.success_history = collections.deque(maxlen=20)
        self.curriculum_level = 0.0

    def _on_rollout_start(self) -> None:
        pass  # Now handled in on_step

    def _on_step(self) -> bool:
        # Only record a sample on the step an episode actually ends
        # (`dones[i]`), one outcome per episode instead of one per step --
        # sampling `is_success` on every step (as the original PPO callback
        # did) dilutes the rolling window with mostly-False mid-episode
        # values and makes the >90% advance threshold effectively
        # unreachable. See [[project-phase2-rl-status]] bug #12.
        dones = self.locals.get("dones", [])
        for i, info in enumerate(self.locals.get("infos", [])):
            if i < len(dones) and dones[i] and "is_success" in info:
                self.success_history.append(float(info["is_success"]))

        if len(self.success_history) >= 20:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate > 0.9:
                self.curriculum_level += 0.05
                self.success_history.clear()  # Reset to thoroughly prove the next level

        # Strictly clamp the curriculum for the first 50k steps, matching
        # train_sac.py's Run 6 rule.
        if self.num_timesteps < 50000:
            self.curriculum_level = min(self.curriculum_level, 0.2)
        else:
            self.curriculum_level = min(self.curriculum_level, 1.0)

        self.training_env.env_method("set_curriculum_level", self.curriculum_level)
        return True

def main():
    parser = argparse.ArgumentParser(description="UR5e RL PPO Training Script")
    parser.add_argument("--resume", type=str, default="", help="Path to a .zip model file to resume training")
    # Parallel Gazebo+MoveIt instances (see ur.gazebo.launch.py's namespace/gz_partition
    # args): each instance i>0 must already be running via
    #   ros2 launch ur_gazebo ur.gazebo.launch.py namespace:=envI gz_partition:=envI headless:=true
    # with instance 0 being whatever's running with the default (blank) namespace. This
    # only changes behavior when --num-envs > 1 -- the default (1) is the exact same
    # single-process DummyVecEnv path as before.
    parser.add_argument("--num-envs", type=int, default=1,
                         help="Number of parallel Gazebo instances to collect from (default 1). "
                              "Instance i's namespace/GZ_PARTITION is 'env{i}' for i>0, '' for i=0.")
    args, unknown_args = parser.parse_known_args()

    print(f"Initializing UR5e Gazebo Environment for PPO ({args.num_envs} parallel instance(s))...")
    # rclpy is initialized inside UR3eEnv.__init__() to prevent the Poison Fork crash.

    def _make_env(env_id, namespace):
        def _init():
            # Scopes this worker subprocess's `gz` CLI calls (UR3eEnv's ground-truth
            # object-pose queries and lego respawn) to the matching Gazebo instance's
            # transport bus. Only meaningful with SubprocVecEnv (a real subprocess per
            # env) -- harmless no-op for the num_envs=1/DummyVecEnv path since namespace
            # is '' there, matching GZ_PARTITION's own default.
            if namespace:
                os.environ['GZ_PARTITION'] = namespace
            return UR3eEnv(env_id=env_id, namespace=namespace)
        return _init

    if args.num_envs > 1:
        env_fns = [
            _make_env(i, '' if i == 0 else f'env{i}')
            for i in range(args.num_envs)
        ]
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv([_make_env(0, '')])

    log_dir = "./rl_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # PPO is on-policy (no replay buffer to persist) -- save_freq aligns with
    # n_steps (1024) so checkpoints only land right after a policy update,
    # matching train_sac.py's "only checkpoint after weights actually moved"
    # rationale for its own save_freq/batch_size alignment.
    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path=log_dir,
        name_prefix="ur3e_ppo_model"
    )

    if args.resume:
        print(f"Resuming PPO Agent from checkpoint: {args.resume}")
        # This checkpoint's observation/action space must match the live env's current
        # architecture -- e.g. the 23-D reach-and-grasp-only space (post 2026-09-01
        # revert) vs. the older 24-D two-phase space, or any pre-hybrid-handoff
        # checkpoint using absolute [-pi, pi] joint-angle actions instead of the current
        # bounded joint-delta actions. This only warns (it can't detect which
        # architecture a given .zip was trained under) -- do not --resume from a
        # mismatched checkpoint. See [[project-phase2-rl-status]].
        print("[WARNING] If this checkpoint predates the hybrid planner+RL handoff "
              "(bounded joint-delta actions, post-handoff start states) or the current "
              "23-D observation space, it is NOT a valid resume target -- treat it as a "
              "cold start instead. Every PPO checkpoint on record predates both.")
        if not os.path.exists(args.resume):
            print(f"[ERROR]: Could not find file {args.resume}. Exiting...")
            return
        # custom_objects forces action_space/observation_space to the live env's values
        # instead of trusting the pickled ones in the checkpoint -- confirmed live (for
        # SAC, same stable_baselines3/numpy stack) that .load() can't deserialize those
        # two objects on this numpy/gymnasium version ("Could not deserialize object
        # action_space ... __generator_ctor() takes from 0 to 1 positional arguments but
        # 2 were given"), which otherwise raises before ever reaching the network
        # weights. See [[project-phase2-rl-status]].
        model = PPO.load(
            args.resume, env=env, tensorboard_log=log_dir,
            custom_objects={
                'action_space': env.action_space,
                'observation_space': env.observation_space,
            }
        )
    else:
        print("Initializing new PPO Agent...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            device="cpu"  # GPU unavailable: PyTorch has no CUDA build installed
        )

    print("Starting Training Loop...")

    total_train_steps = 100000
    curriculum_callback = CurriculumCallback(total_timesteps=total_train_steps)

    try:
        # reset_num_timesteps=False on a resume, matching train_sac.py: otherwise even a
        # valid --resume restarts the step counter (and this callback's first-50k-steps
        # curriculum clamp, which gates on self.num_timesteps) from zero every time.
        model.learn(
            total_timesteps=total_train_steps,
            callback=[checkpoint_callback, curriculum_callback],
            tb_log_name="PPO_Reach_Curriculum",
            reset_num_timesteps=not bool(args.resume)
        )

        # Save the final model
        model.save(f"{log_dir}/ur3e_ppo_final")
        print("Training Completed and Model Saved!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save(f"{log_dir}/ur3e_ppo_interrupted")
    finally:
        env.close()

if __name__ == '__main__':
    main()
