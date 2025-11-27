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
#                           CONFIGURAZIONE RL
# =================================================================

LOG_DIR = "./logs/ppo_road_drawer_v1"
CHECKPOINT_DIR = "./checkpoints_rl"
TOTAL_TIMESTEPS = 2_000_000  
EVAL_FREQ_EPISODES = 3    

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

# Parametri Policy (CNN custom)
POLICY_KWARTS = {
    "features_extractor_class": CustomCnnExtractor,
    "features_extractor_kwargs": {"features_dim": 256}, 
    "normalize_images": False, 
}


# =================================================================
#                 CALLBACK PER VISUALIZZAZIONE E SALVATAGGIO
# =================================================================

class RenderCallback(BaseCallback):
    """
    Esegue un episodio completo ogni N episodi, salva l'immagine e il modello.
    """
    def __init__(self, eval_env, render_freq, save_path, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.save_path = save_path
        self.episode_counter = 0

        self.last_obs = self.eval_env.reset()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        
        self.episode_counter += 1

        if self.episode_counter % self.render_freq == 0:
            
            # --- RECUPERO DATI AMBIENTE ---
            # Andiamo a "spiare" dentro l'ambiente per vedere quale mappa è caricata ORA.
            # eval_env è un VecEnv, quindi l'ambiente vero è dentro .envs[0]
            inner_env = self.eval_env.envs[0]
            
            current_map = inner_env.current_map_name
            # Recuperiamo l'ID dal mission_cache che abbiamo salvato nel reset
            current_path_id = inner_env.mission_cache.get('path_id', '?')
            raw_total = getattr(inner_env, 'original_total_paths', len(inner_env.available_paths_cache))

            if  isinstance(raw_total, int):
                max_index = raw_total - 1
            else:
                max_index = '?'
            
            # --- STAMPA SINCRONIZZATA ---
            # Stampiamo ORA, esattamente prima di iniziare il test
            print(f"\n[VALID START] Map: {current_map} | Path ID: {current_path_id}/{max_index}")

            # --- INIZIO TEST (uguale a prima) ---
            obs = self.last_obs
            if isinstance(obs, tuple): obs = obs[0]
                
            done = False
            final_reward = 0.0
            max_steps_limit = 600 
            steps_taken = 0
            
            while not done and steps_taken < max_steps_limit:
                action, _states = self.model.predict(obs, deterministic=True)
                step_output = self.eval_env.step(action)
                
                if len(step_output) == 5:
                    obs, reward, done, truncated, info = step_output
                else:
                    obs, reward, done, info = step_output
                    truncated = False 
                
                self.last_obs = obs # Aggiorniamo per il futuro
                
                if isinstance(done, (list, np.ndarray)): done = done[0]
                if isinstance(reward, (list, np.ndarray)): final_reward += reward[0]
                else: final_reward += reward
                
                steps_taken += 1

            # --- FINE TEST E RENDER ---
            env_instance = self.eval_env.envs[0] 
            file_name = f"render_{self.num_timesteps}_ID{current_path_id}_R{final_reward:.1f}.png"
            env_instance.render_frame(save_path=os.path.join(self.save_path, file_name), final_steps=steps_taken)
            
            # Stampa risultato subito dopo lo Start
            print(f"[VALID END]   Reward: {final_reward:.2f} | Steps: {steps_taken}")
            
            ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_model.zip")
            self.model.save(ckpt_path)
            self.episode_counter = 0


# =================================================================
#                           FUNZIONE MAIN
# =================================================================

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # --- 1. CREAZIONE AMBIENTI ---
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_env_instance = RoadDrawerEnv(split='train', device=device_name)
    train_env = DummyVecEnv([lambda: train_env_instance]) 
    
    # L'ambiente di validazione pesca dallo split 'valid' (se esiste nel JSON)
    eval_env_instance = RoadDrawerEnv(split='valid', device=device_name)
    eval_env = DummyVecEnv([lambda: eval_env_instance]) 

    # --- 2. CONFIGURAZIONE ALGORITMO PPO ---
    model = PPO(
        ActorCriticCnnPolicy, 
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device_name,
        **PPO_PARAMS, 
        policy_kwargs=POLICY_KWARTS,
    )
    
    # --- 3. CONFIGURAZIONE CALLBACK E TRAINING ---
    
    os.makedirs(os.path.join(LOG_DIR, "renders"), exist_ok=True)
    
    render_callback = RenderCallback(
        eval_env=eval_env,
        render_freq=EVAL_FREQ_EPISODES,
        save_path=os.path.join(LOG_DIR, "renders")
    )
    
    print(f"\nInizio addestramento Agente RL (PPO) su {device_name}...")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=render_callback
    )

    model.save(os.path.join(CHECKPOINT_DIR, "rl_drawer_final.zip"))
    print("\nAddestramento completato. Modello salvato.")


if __name__ == "__main__":
    main()