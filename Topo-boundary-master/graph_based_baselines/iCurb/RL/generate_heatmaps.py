import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as tvf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# --- PYTHON PATH FIX ---
import sys
# Adds the project root directory to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -----------------------

# IMPORT THE MODEL CLASS DEFINITION
from models.models_encoder import FPN 

# =================================================================
#                         CONFIGURATION
# =================================================================

# --- PATHS ---
DATA_ROOT = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"          # Folder containing TIFF images (Relative to RL/)
JSON_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/data_split.json"      
MODEL_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/checkpoints/seg_pretrain_manhattan_efficentnet_1.6_v2.pth" 
# Output paths are relative to the Current Working Directory (CWD), which is RL/
OUTPUT_DIR_TRAIN_HEATMAPS = "train/heatmaps"
OUTPUT_DIR_VAL_HEATMAPS = "valid/heatmaps"
OUTPUT_DIR_VIZ_TRAIN = "train/viz" # New visualization folder
OUTPUT_DIR_VIZ_VAL = "valid/viz"     # New visualization folder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CHANNELS = 4 
N_CLASSES = 1

# =================================================================
#                         HELPER FUNCTIONS
# =================================================================

def load_model(model_path):
    """Loads the FPN model weights and sets it to evaluation mode."""
    print(f"Loading model on {DEVICE} from: {model_path}")
    
    model = FPN(n_channels=N_CHANNELS, n_classes=N_CLASSES) 
    
    try:
        if DEVICE.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"ERROR: Could not load FPN model weights: {e}")
        return None
    
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image_to_tensor(img_pil):
    """Converts PIL image (assumed 4-channel input) to normalized tensor."""
    img_tensor = tvf.to_tensor(img_pil)
    
    # Ensure the tensor has exactly N_CHANNELS (4)
    if img_tensor.shape[0] > N_CHANNELS:
         img_tensor = img_tensor[:N_CHANNELS, :, :]
    elif img_tensor.shape[0] < N_CHANNELS:
        # Pad if necessary (e.g., RGB image loaded, but model expects 4 channels)
        padding = torch.zeros(N_CHANNELS - img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = torch.cat([img_tensor, padding], dim=0)

    # Add the batch dimension [1, C, H, W]
    return img_tensor.unsqueeze(0)

def save_visualization(img_rgb, heatmap_np, output_path, name):
    """Overlays the heatmap on the original RGB image and saves the result."""
    
    plt.figure(figsize=(10, 10))
    
    # 1. Display the RGB background image (zorder=0)
    plt.imshow(img_rgb)
    
    # 2. Overlay the Heatmap (zorder=1)
    # Use 'jet' or 'viridis' colormap, 0.5 transparency.
    plt.imshow(heatmap_np, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    
    # Add a color bar to show the probability scale
    plt.colorbar(fraction=0.046, pad=0.04) 
    
    plt.title(f"Heatmap Overlay: {name}")
    plt.axis('off')
    
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"WARNING: Could not save visualization to {output_path}: {e}")
    finally:
        plt.close()


def generate_and_save_heatmaps(net, split_data, output_dir_npy, output_dir_viz, split_name):
    """Generates heatmaps, saves .npy, and saves visualizations."""
    
    # Fix 1: Ensure output folders exist without redundant nesting
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_viz, exist_ok=True)
    
    print(f"\nStarting Heatmap generation for split: {split_name}")

    with torch.no_grad():
        for image_name in tqdm(split_data, desc=f"Processing {split_name}"):
            
            image_path = os.path.join(DATA_ROOT, f"{image_name}.tiff")
            
            if not os.path.exists(image_path):
                 print(f"WARNING: Image not found: {image_name}.tiff. Skipping.")
                 continue

            img_pil = Image.open(image_path)
            
            # 1. Preprocessing for Model (Tensor)
            input_tensor = preprocess_image_to_tensor(img_pil).to(DEVICE)

            # 2. FPN Inference
            logits, _ = net(input_tensor) 
            
            # 3. Sigmoid and Conversion NumPy
            heatmap_tensor = torch.sigmoid(logits)
            heatmap_np = heatmap_tensor.squeeze().cpu().numpy() 

            # 4. Save .npy for RL input
            np.save(os.path.join(output_dir_npy, f"{image_name}.npy"), heatmap_np)
            
            # 5. Save Visualization (Fix 2: Overlay)
            img_rgb = img_pil.convert('RGB')
            save_visualization(img_rgb, heatmap_np, 
                               os.path.join(output_dir_viz, f"{image_name}.png"), image_name)


def main():
    # 0. Load Dataset Split Info
    try:
        with open(JSON_PATH, 'r') as f:
            dataset_info = json.load(f)
            train_data = dataset_info['train']
            val_data = dataset_info['valid']
    except Exception as e:
        print(f"Error loading dataset JSON from {JSON_PATH}: {e}")
        return

    # 1. Load Model
    net = load_model(MODEL_PATH)
    if net is None: return

    # 2. Generate and Save Heatmaps (Train Set)
    generate_and_save_heatmaps(net, train_data, OUTPUT_DIR_TRAIN_HEATMAPS, OUTPUT_DIR_VIZ_TRAIN, 'train')

    # 3. Generate and Save Heatmaps (Validation Set)
    generate_and_save_heatmaps(net, val_data, OUTPUT_DIR_VAL_HEATMAPS, OUTPUT_DIR_VIZ_VAL, 'valid')

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()