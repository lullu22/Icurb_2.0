import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# Importa i componenti custom
from road_drawer_env import RoadDrawerEnv
from rl_model import CustomCnnExtractor 

# =================================================================
#                            CONFIGURATION RL
# =================================================================

LOG_DIR = "./logs/ppo_road_drawer_v1_reversed"
CHECKPOINT_DIR = "./checkpoints_rl_reversed"
TOTAL_TIMESTEPS = 10_000_000  
EVAL_FREQ_EPISODES = 5  
ENABLE_REVERSE_LEARNING = False

# Hyperparameters PPO (Standard) 
PPO_PARAMS = {
    "n_steps": 8192,
    "batch_size": 512,
    "learning_rate": 1e-4, 
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
}

# POLICY PARAMETERS (CNN custom)
POLICY_KWARTS = {
    "features_extractor_class": CustomCnnExtractor,
    "features_extractor_kwargs": {"features_dim": 256}, 
    "normalize_images": False, 
}


# =================================================================
#                 CALLBACK FOR RENDERING AND SAVING
# =================================================================

"""
Creation of an extended version of BaseCallback (from stable_baselines3.common.callbacks) to perform evaluation 
"""

class RenderCallback(BaseCallback):
    """
    Computes an entire episode every N episodes, saves the image and the model.
    """
    def __init__(self, eval_env, render_freq, save_path, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.save_path = save_path
        self.episode_counter = 0

        self.last_obs = self.eval_env.reset() # is a tuple (obs, info)

    def _on_step(self) -> bool: # default function of BaseCallback, we set it to always return True, in order to not stop training
        return True 

    def _on_rollout_end(self) -> None:
        
        self.episode_counter += 1

        if self.episode_counter % self.render_freq == 0:
            
            # --- take environment info ---

            # We work with DummyVecEnv, so we access the inner env with envs[0], standard SB3 practice
            inner_env = self.eval_env.envs[0]
            # Get current map name
            current_map = inner_env.current_map_name
            # Recover the ID from the mission_cache that we saved in the reset
            current_path_id = inner_env.mission_cache.get('path_id', '?')
            # Recover total number of paths available (if exists)
            raw_total = getattr(inner_env, 'original_total_paths', len(inner_env.available_paths_cache))

            if  isinstance(raw_total, int):
                max_index = raw_total - 1
            else:
                max_index = '?'

            # ----------------------------
            
            # --- print start-info -------------

            print(f"\n[VALID START] Map: {current_map} | Path ID: {current_path_id}/{max_index}")

            

            # --- Start test and render ---

            obs = self.last_obs
            if isinstance(obs, tuple): 
                obs = obs[0]
            
            done = False
            final_reward = 0.0
            max_steps_limit = 600  # Safety limit to avoid infinite loops in testing
            steps_taken = 0
            
            while not done and steps_taken < max_steps_limit:
                action, _states = self.model.predict(obs, deterministic=True)
                step_output = self.eval_env.step(action)
                
                # compatibility with different Gym versions
                # --- Gym >=0.26 returns 5 values (obs, reward, done, truncated, info) ---
                if len(step_output) == 5:
                    obs, reward, done, truncated, info = step_output
                # --- Gym <0.26 returns 4 values (obs, reward, done, info) ---
                else:
                    obs, reward, done, info = step_output
                    truncated = False 

                self.last_obs = obs # save last obs for next step

                # Unwrap from list if needed (from DummyVecEnv)
                if isinstance(done, (list, np.ndarray)): 
                    done = done[0]
                if isinstance(reward, (list, np.ndarray)):
                    final_reward += reward[0]
                else: 
                    final_reward += reward
                
                steps_taken += 1

            # --- Save render image and model checkpoint ---

            env_instance = self.eval_env.envs[0] #### Possibile modifica, gia dichiarato sopra (riga 74)
            file_name = f"render_{self.num_timesteps}_ID{current_path_id}_R{final_reward:.1f}.png"
            env_instance.render_frame(save_path=os.path.join(self.save_path, file_name), final_steps=steps_taken)

            # --- print end-info ------------
            print(f"[VALID END]   Reward: {final_reward:.2f} | Steps: {steps_taken}")
            
            # model checkpoint saving
            ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_model_reversed.zip")
            self.model.save(ckpt_path)

            #  Reset episode counter
            self.episode_counter = 0

            # ----------------------------

# =================================================================
#                         MAIN FUNCTION
# =================================================================

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- 1. creation environments ---
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
   # Train
    train_env_instance = RoadDrawerEnv(split='train', device=device_name, enable_reverse_learning= ENABLE_REVERSE_LEARNING)
    train_env = DummyVecEnv([lambda: train_env_instance])

    # Validation
    eval_env_instance = RoadDrawerEnv(split='valid', device=device_name, enable_reverse_learning= ENABLE_REVERSE_LEARNING)
    eval_env = DummyVecEnv([lambda: eval_env_instance])

    # --- 2. configuration PPO ---
    model = PPO(
        ActorCriticCnnPolicy, 
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device_name,
        **PPO_PARAMS, 
        policy_kwargs=POLICY_KWARTS,
    )

    # --- 3. configuration callbacks and training ---

    os.makedirs(os.path.join(LOG_DIR, "renders_reversed"), exist_ok=True)
    
    render_callback = RenderCallback(
        eval_env=eval_env,
        render_freq=EVAL_FREQ_EPISODES,
        save_path=os.path.join(LOG_DIR, "renders_reversed")
    )
    
    print(f"\nstart training of PPO on {device_name}...")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=render_callback
    )

    model.save(os.path.join(CHECKPOINT_DIR, "rl_drawer_final_reversed.zip"))
    print("\nTraining completed. Model saved.")


if __name__ == "__main__":
    main()