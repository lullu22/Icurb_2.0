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
    Custom CNN feature extractor for Stable Baselines 3.
    Processes observations with 2 channels (road heatmap + target/compass).
    Produces a feature vector of size `features_dim`.
    1. Input: (B, 2, 128, 128)
    2. CNN Layers:
       - Conv2d -> ReLU -> MaxPool2d
       - Conv2d -> ReLU -> MaxPool2d
       - Conv2d -> ReLU -> MaxPool2d
       - Conv2d -> ReLU
    3. Flatten
    4. Linear Layer -> ReLU
    5. Output: (B, features_dim)
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = OUT_FEATURE_SIZE):
        super(CustomCnnExtractor, self).__init__(observation_space, features_dim)
        
        # number of input channels (2: road heatmap + target/compass)
        n_input_channels = 2
        
        # Define the CNN architecture
        self.cnn = nn.Sequential(
            # Layer 1: Input (B, 2, 128, 128)
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 32, 64, 64)

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 64, 32, 32)

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (B, 128, 16, 16)

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        with th.no_grad():
            # compute shape of the output of the CNN
            sample = observation_space.sample()
            dummy_input = th.as_tensor(sample[None]).float()
            # Pass through CNN, in this way we can infer the output size
            output = self.cnn(dummy_input)
            n_flatten = output.shape[1] # in this way we get the flattened size dynamically respecting the crop size (input size)
        
        # Define the final linear layer to get desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Forward pass through CNN and Linear layers
        return self.linear(self.cnn(observations))