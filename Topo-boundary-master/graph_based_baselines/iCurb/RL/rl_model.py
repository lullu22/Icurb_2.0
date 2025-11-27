import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- CONFIGURATION ---
# Deve coincidere con quello nell'env, ma qui è dinamico
OUT_FEATURE_SIZE = 256 
# ---------------------

class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    CNN Ottimizzata per il tracciamento di linee (Road Tracing).
    Input atteso: (2, 64, 64) -> (Canali, Altezza, Larghezza)
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = OUT_FEATURE_SIZE):
        super(CustomCnnExtractor, self).__init__(observation_space, features_dim)
        
        # Ora abbiamo 2 canali (Strada + Target)
        n_input_channels = 2
        
        # Definizione della CNN "High-Res"
        self.cnn = nn.Sequential(
            # Layer 1: Input (B, 2, 64, 64)
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 32, 32, 32)

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 64, 16, 16)

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 128, 8, 8)

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # --- FIX DEL CRASH: Calcolo dimensione dinamica ---
        with th.no_grad():
            # observation_space.sample() restituisce (2, 64, 64)
            # Aggiungiamo UNA SOLA dimensione batch: [None] -> (1, 2, 64, 64)
            sample = observation_space.sample()
            dummy_input = th.as_tensor(sample[None]).float()
            
            # Passiamo nella CNN per vedere quanto esce lungo il vettore
            output = self.cnn(dummy_input)
            n_flatten = output.shape[1]
        
        # Linear Layer Finale
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # SB3 passa già i tensori nel formato corretto (Batch, Canali, H, W)
        # Non serve fare unsqueeze manuali strani qui se l'environment è configurato bene.
        return self.linear(self.cnn(observations))