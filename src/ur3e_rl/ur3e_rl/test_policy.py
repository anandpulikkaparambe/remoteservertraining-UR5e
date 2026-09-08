import argparse
import os
import rclpy
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from ur3e_rl.ur3e_env import UR3eEnv

def _create_env():
    return UR3eEnv(env_id=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC checkpoint")
    parser.add_argument("--model-path", type=str, default="./rl_logs/ur3e_sac_final.zip",
                         help="Path to the .zip checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, default=10,
                         help="Number of evaluation episodes to run")
    args, _ = parser.parse_known_args()

    print("Initializing UR5e Gazebo Environment for Evaluation...")

    # Initialize ROS 2
    if not rclpy.ok():
        rclpy.init()

    # Create the environment exactly as it was during training
    env = DummyVecEnv([_create_env])

    model_path = args.model_path

    # Check if a model actually exists yet
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}!")
        print("Please wait for your training to finish saving, or change the filename to point to a checkpoint in ./rl_logs/ if you stopped it early.")
        env.close()
        return

    print(f"Loading trained SAC model from {model_path}...")
    model = SAC.load(model_path, env=env)

    # --- Systematic Generalization Test ---
    import json
    import time
    benchmark_dir = os.path.expanduser("~/.local/share/ur3e_sorting/benchmarks/")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # We will run 10 test episodes and measure the success rate
    num_episodes = args.episodes
    successes = 0
    
    print("\n--- Starting Sim-to-Real Evaluation ---")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        run_start_time = time.time()
        
        while not done:
            # The agent predicts the next best action given the current observation
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute the action in Gazebo (or on the real robot if connected to its hardware interface!)
            obs, reward, done, info = env.step(action)
            step += 1
            
            # The 'done' flag is triggered by either success (< 5cm) or collision (< 0cm Z-height)
            if done:
                # DummyVecEnv wraps the info dict in a list
                env_info = info[0]
                is_success = bool(env_info.get('is_success', False))
                dist = env_info.get('distance_to_target', 0.0)
                
                # --- Create Benchmark JSON ---
                run_duration = time.time() - run_start_time
                benchmark = {
                    "planner_id": "RL_SAC",
                    "pipeline_id": "stable_baselines3",
                    "moves": [
                        {
                            "move": "rl_episode",
                            "duration_s": run_duration,
                            "attempts": 1,
                            "used_cache": False,
                            "planning_time_s": 0.0,
                            "planner_used": "RL_SAC",
                            "num_waypoints": step,
                            "path_length_rad": 0.0, # Not explicitly tracked here
                            "verify_error": float(dist),
                            "joint_angles_rad": {},
                            "failed": not is_success
                        }
                    ],
                    "run_duration_s": run_duration,
                    "total_retries": 0,
                    "placement_error_m": float(dist),
                    "success": is_success
                }
                
                fname = os.path.join(benchmark_dir, f"run_{int(time.time())}.json")
                with open(fname, 'w') as f:
                    json.dump(benchmark, f, indent=2)
                
                if is_success:
                    successes += 1
                    print(f"Episode {episode + 1}: SUCCESS! Reached target in {step} steps. Saved to {os.path.basename(fname)}.")
                else:
                    print(f"Episode {episode + 1}: FAILED. Distance: {dist:.3f}m. Saved to {os.path.basename(fname)}.")
                    
    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {successes}/{num_episodes} ({(successes/num_episodes)*100}%)")
    print("If Success Rate is high, this policy is robust enough for physical Sim-to-Real transfer!")
    
    env.close()

if __name__ == '__main__':
    main()
