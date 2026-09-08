import os
import re
import rclpy
import argparse
import collections
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from ur3e_rl.ur3e_env import UR3eEnv

class CurriculumCallback(BaseCallback):
    """
    Custom callback for conditionally increasing the curriculum.
    Tracks success rates and actively defends against Critic Deterioration.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.success_history = collections.deque(maxlen=20)
        self.curriculum_level = 0.0

    def _on_rollout_start(self) -> None:
        pass # Now handled in on_step

    def _on_step(self) -> bool:
        # Hack to intercept internal SB3 Critic Loss and prevent catastrophic collapse
        if self.model and hasattr(self.model, 'logger') and self.model.logger:
            if "train/critic_loss" in self.model.logger.name_to_value:
                critic_loss = self.model.logger.name_to_value["train/critic_loss"]
                if critic_loss > 1000.0:
                    print(f"\\n[KILL SWITCH] Critic loss exploded to {critic_loss}! Stopping training early to prevent NaN poisoning.")
                    return False
        
        # Track environmental success triggers. `is_success` is set in `info` on every
        # single env step (not just terminal ones -- see ur3e_env.py's step()), so gating
        # only on "is_success" in info sampled every step regardless of episode
        # completion. Since episodes run many steps with is_success=False until (if ever)
        # the one terminal step where it's True, that diluted the rolling window with
        # mostly-False mid-episode samples and made the >90% threshold below essentially
        # unreachable -- curriculum_level was observed stuck at 0.0 for an entire
        # multi-hundred-thousand-step run. Fixed: only record a sample on the step an
        # episode actually ends (`dones[i]`), one outcome per episode instead of one per
        # step. See [[project-phase2-rl-status]].
        dones = self.locals.get("dones", [])
        for i, info in enumerate(self.locals.get("infos", [])):
            if i < len(dones) and dones[i] and "is_success" in info:
                self.success_history.append(float(info["is_success"]))
                
        # Evaluate Success-Triggered Curriculum
        if len(self.success_history) >= 20:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate > 0.9:
                self.curriculum_level += 0.05
                self.success_history.clear() # Reset to thoroughly prove the next level
                
        # Run 6 Rule: Strictly clamp the curriculum for the first 50k steps
        if self.num_timesteps < 50000:
            self.curriculum_level = min(self.curriculum_level, 0.2)
        else:
            self.curriculum_level = min(self.curriculum_level, 1.0)
            
        self.training_env.env_method("set_curriculum_level", self.curriculum_level)
        return True

def main():
    parser = argparse.ArgumentParser(description="UR5e RL SAC Training Script")
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

    print(f"Initializing UR5e Gazebo Environment for SAC ({args.num_envs} parallel instance(s))...")
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
    
    # save_replay_buffer=True: previously only network weights were checkpointed, so a
    # --resume always cold-started the replay buffer regardless of how much off-policy
    # data the prior run had collected (see hybrid planner+RL spec section 6).
    #
    # save_freq raised 256 -> 2000 (2026-09-04, implementation spec section 6): at 256, the
    # ~22MB replay-buffer pickle was written every ~2-3 minutes, which filled disk to 100%
    # once already (2026-08-19) and is the leading suspect for the 2026-09-03 SIGKILL
    # incident on the external Expansion drive (see project_phase2_rl_status memory). 2000
    # steps is roughly 8x less frequent -- still frequent enough that a crash loses at most
    # a few minutes of progress, but meaningfully lighter on disk/write load.
    # name_prefix="justrl" (2026-09-06): checkpoints are named only by step count
    # (e.g. "..._2000_steps.zip") with no run ID, so a fresh cold start's early checkpoints
    # could silently overwrite an older run's files at the same step number -- flagged as a
    # real risk earlier, addressed here by giving this run's checkpoints their own distinct
    # prefix, guaranteed to never collide with "ur3e_sac_model_*" files from any prior run.
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=log_dir,
        name_prefix="justrl",
        save_replay_buffer=True
    )

    if args.resume:
        print(f"Resuming SAC Agent from checkpoint: {args.resume}")
        # Hybrid planner+RL architecture (spec section 4): a checkpoint trained before
        # this change used absolute [-pi, pi] joint-angle actions and full-workspace start
        # states -- both the action semantics and the state distribution are different
        # under the hybrid handoff design, so its network weights are not a valid warm
        # start here. This only warns (it can't detect which architecture a given .zip was
        # trained under) -- do not --resume from a pre-hybrid checkpoint.
        print("[WARNING] If this checkpoint predates the hybrid planner+RL handoff "
              "(bounded joint-delta actions, post-handoff start states), it is NOT a "
              "valid resume target -- treat it as a cold start instead.")
        if not os.path.exists(args.resume):
            print(f"[ERROR]: Could not find file {args.resume}. Exiting...")
            return
        # Load the saved Intuition!
        # custom_objects forces action_space/observation_space to the live env's values
        # instead of trusting the pickled ones in the checkpoint: confirmed live that
        # SAC.load() can't deserialize those two objects at all on this numpy/gymnasium
        # version ("Could not deserialize object action_space ... __generator_ctor()
        # takes from 0 to 1 positional arguments but 2 were given" -- a pickle-format
        # drift from whatever numpy/gymnasium version was installed when this checkpoint
        # was originally saved), which otherwise makes load() raise KeyError before ever
        # reaching the actual network weights. The live env's spaces are the correct
        # ground truth anyway (this is the same architecture the checkpoint was saved
        # under -- 7-dim action, 24-dim observation -- unchanged by this fix).
        model = SAC.load(
            args.resume, env=env, tensorboard_log=log_dir,
            custom_objects={
                'action_space': env.action_space,
                'observation_space': env.observation_space,
            }
        )

        # CheckpointCallback's save_replay_buffer=True writes a sibling
        # <name_prefix>_replay_buffer_<steps>_steps.pkl next to each
        # <name_prefix>_<steps>_steps.zip -- restore it too so a resume keeps its
        # off-policy data instead of relearning from an empty buffer.
        resume_dir, resume_name = os.path.split(args.resume)
        # 2026-09-06: derived via regex on the trailing "_<steps>_steps.zip" pattern
        # instead of a hardcoded prefix string -- the previous version hardcoded
        # "ur3e_sac_model_", which silently broke the moment name_prefix changed (e.g. to
        # "justrl"), always falling through to "resuming with an empty replay buffer"
        # without any error. This form works regardless of name_prefix, so a future rename
        # can't reintroduce the same silent failure. (An earlier version of this bug, fixed
        # 2026-08-31, was a bare .replace() keeping the checkpoint's own ".zip" suffix
        # instead of ".pkl" -- see that date's notes for the original incident.)
        replay_buffer_name = re.sub(r'_(\d+_steps)\.zip$', r'_replay_buffer_\1.pkl', resume_name)
        replay_buffer_path = os.path.join(resume_dir, replay_buffer_name)
        if os.path.exists(replay_buffer_path):
            print(f"Restoring replay buffer from: {replay_buffer_path}")
            try:
                model.load_replay_buffer(replay_buffer_path)
            except Exception as e:
                # Confirmed live 2026-08-31: this .pkl's internal numpy RandomState/
                # Generator was pickled under a different numpy version than whatever is
                # currently installed ("__generator_ctor() takes from 0 to 1 positional
                # arguments but 2 were given") -- a pickle-format break, not a corrupt
                # file. Unlike action_space/observation_space above (where a live
                # replacement object works), the buffer's actual transition data lives
                # behind the same broken unpickle call, so there's no way to recover it
                # here. Not fatal: the network weights loaded fine above and are the
                # valuable part of a resume; losing the replay buffer just means SAC
                # relearns off-policy data from scratch instead of reusing old
                # transitions, not that the resume itself is invalid.
                print(f"[WARNING] Could not load replay buffer ({e}) -- likely a numpy "
                      "pickle-format mismatch between when this was saved and the numpy "
                      "version now installed. Resuming with an empty replay buffer; "
                      "network weights are unaffected.")
        else:
            print(f"[WARNING] No matching replay buffer found at {replay_buffer_path} "
                  "-- resuming with an empty replay buffer.")
    else:
        print("Initializing new SAC Agent...")
        # Instantiate the Soft Actor-Critic agent
        # SAC is generally much more sample-efficient than PPO for physical robotics tasks
        #
        # target_entropy (2026-09-04): SB3's default 'auto' targets -action_dim (-7 here).
        # Diagnostic run against the most recent completed run (SAC_Reach_Curriculum_36,
        # checked via TensorBoard train/ent_coef) showed real early entropy collapse under
        # that default: ent_coef ~1.0 at step 111 -> 0.113 by step 22,305 (within ~7% of
        # that run's total steps) -> ~0.01-0.02 by step 300k+, i.e. the policy went
        # near-deterministic very early in training. That run was entirely under the old
        # fixed-5cm-standoff env (curriculum_level never wired to anything -- see
        # HANDOFF_STANDOFF_M/curriculum wiring above), so the collapse happened before the
        # policy could plausibly have benefited from any curriculum widening even if it had
        # existed then. With curriculum widening now real (see ur3e_env.py), a policy that
        # locks in near-deterministic behavior this early in training risks doing so before
        # curriculum_level has ramped far enough to matter. Per the implementation spec's
        # section 4 ("if entropy did collapse early, increase target_entropy... e.g.
        # -0.5 * action_dim"), set explicitly here rather than left on the default -- gated
        # on this diagnostic result, not applied speculatively.
        target_entropy = -0.5 * env.action_space.shape[0]
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4, # Decreased for Run 6
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            target_entropy=target_entropy,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu"  # GPU unavailable: PyTorch has no CUDA build installed
        )

    print("Starting Training Loop...")
    
    total_train_steps = 100000  # reduced from 200000 on 2026-09-03 (thesis deadline pressure + re-verifying the newly real-hardware-matched env geometry)
    curriculum_callback = CurriculumCallback(total_timesteps=total_train_steps)
    
    try:
        # reset_num_timesteps=False on a resume: previously this was omitted, so even a
        # valid --resume restarted the SAC step counter (and the Run 6 curriculum clamp
        # in CurriculumCallback, which gates on self.num_timesteps < 50000) from zero
        # every time, silently re-running the first-50k-steps clamp on top of an
        # already-trained model (see hybrid planner+RL spec section 6).
        model.learn(
            total_timesteps=total_train_steps,
            callback=[checkpoint_callback, curriculum_callback],
            tb_log_name="justrl",
            reset_num_timesteps=not bool(args.resume)
        )

        # Save the final model
        model.save(f"{log_dir}/justrl_final")
        print("Training Completed and Model Saved!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save(f"{log_dir}/justrl_interrupted")
    finally:
        env.close()

if __name__ == '__main__':
    main()
