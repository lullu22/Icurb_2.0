import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv 
from torch_geometric.data import DataLoader, Dataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os
import numpy as np
from tqdm import tqdm
import pickle

# --- Import per Visualizzazione ---
import matplotlib
matplotlib.use('Agg') # Backend non-interattivo
import matplotlib.pyplot as plt
from PIL import Image
# --------------------------------

# --- CONFIGURAZIONE GLOBALE ---
OUTPUT_DATASET_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/GNN/gnn_dataset_cleaner_20" 
GT_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/gt_graphs_2"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"

# Output
LOG_DIR = './runs/gnn_cleaner_5feat_experiment_v2' # Cartella nuova per non mischiare log
CHECKPOINT_SAVE_PATH = './checkpoints/gnn_cleaner_gat.pth'

# Parametri
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1 
VALIDATION_SPLIT = 0.20
NUM_CLASSES = 2 
IMG_SIZE = 1000.0
EPOCHS = 250
# ------------------------------------------------------------------

#
# --- 1. Dataset Loader ---
#
class GNNLoadingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(GNNLoadingDataset, self).__init__(root_dir, transform)
        self.root = root_dir
        self.pt_files = sorted([f for f in os.listdir(self.root) if f.endswith('.pt')])
        if not self.pt_files:
            raise FileNotFoundError(f"Nessun file .pt trovato in {self.root}.")
        print(f"Dataset Loader: Trovati {len(self.pt_files)} campioni.")

    def len(self):
        return len(self.pt_files)
        
    def get(self, idx):
        path = os.path.join(self.root, self.pt_files[idx])
        data = torch.load(path)
        return data

#
# --- 2. Modello: GATCleaner (GATv2) ---
#
class GATCleaner(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(GATCleaner, self).__init__()
        # Usiamo GATv2Conv per attenzione dinamica migliore
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        
        self.node_cls_head = torch.nn.Linear(hidden_channels, NUM_CLASSES) 

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        node_logits = self.node_cls_head(x)
        return node_logits

#
# --- 3. Evaluate (con Metriche Avanzate) ---
#
def evaluate(model, data_loader, epoch, criterion):
    model.eval() 
    total_loss = 0; total_samples = 0
    
    # Liste per accumulare tutto il dataset di validazione
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Validation E{epoch:03d}"):
            data = data.to(DEVICE)
            logits = model(data.x, data.edge_index)
            loss = criterion(logits, data.y)
            
            total_loss += loss.item()
            total_samples += 1
            
            # Accumula predizioni per calcolo F1 globale
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
            
    avg_loss = total_loss / max(total_samples, 1)

    # Calcolo Metriche sklearn
    # Class 0 = Keep (Strade), Class 1 = Discard (Rumore)
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'recall_roads': recall[0],   # Quanto siamo bravi a NON cancellare strade (Cruciale)
        'f1_roads': f1[0],           # Bilanciamento su Strade
        'f1_noise': f1[1],           # Bilanciamento su Rumore
        'precision_noise': precision[1] # Quando diciamo "butta", è vero?
    }
    return metrics

#
# --- 4. Train Loop ---
#
def train_with_logging(writer, model, optimizer, data_loader, global_step_start, criterion):
    model.train()
    total_loss = 0; total_samples = 0
    global_step = global_step_start
    
    for data in tqdm(data_loader, desc="Training GNN"):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        node_pred = model(data.x, data.edge_index)
        loss = criterion(node_pred, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += 1
        
        writer.add_scalar('Loss/Step_Train', loss.item(), global_step)
        global_step += 1
        
    writer.add_scalar('Loss/Epoch_Avg_Train', total_loss / max(total_samples, 1), global_step_start)
    return total_loss / max(total_samples, 1)

#
# --- 5. Logging Visuale ---
#
def log_validation_image(writer, model, val_dataset, epoch_step):
    # print(f"Generating validation image...")
    model.eval()
    idx = torch.randint(0, len(val_dataset), (1,)).item()
    data = val_dataset[idx]
    
    try:
        file_name = data.file_name
        tiff_path = os.path.join(IMAGE_DIR, file_name.replace('.pickle', '.tiff'))
        gt_graph_path = os.path.join(GT_GRAPH_DIR, file_name)
        
        rgb_img = np.array(Image.open(tiff_path).convert('RGB'))
        with open(gt_graph_path, 'rb') as f:
            gt_graph = pickle.load(f)
            gt_nodes = np.array(gt_graph['vertices'])
    except Exception as e:
        print(f"Viz Error: {e}")
        return

    data_gpu = data.to(DEVICE)
    with torch.no_grad():
        node_logits = model(data_gpu.x, data_gpu.edge_index)
    
    node_preds = torch.argmax(node_logits, dim=1).cpu().numpy()
    pred_nodes = data.x.cpu().numpy()[:, :2] * IMG_SIZE
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"GNN Cleaner Result - {file_name}", fontsize=16)

    # Pannello 1: GNN Output
    ax1.imshow(rgb_img, zorder=0)
    kept = pred_nodes[node_preds == 0]
    discarded = pred_nodes[node_preds == 1]
    if len(kept) > 0: ax1.scatter(kept[:, 1], kept[:, 0], c='lime', s=15, label='Kept', zorder=2)
    if len(discarded) > 0: ax1.scatter(discarded[:, 1], discarded[:, 0], c='red', s=15, label='Discarded', zorder=3)
    ax1.set_title("GNN Output")
    ax1.legend()
    ax1.axis('off')

    # Pannello 2: GT
    ax2.imshow(rgb_img, zorder=0)
    if len(gt_nodes) > 0: ax2.scatter(gt_nodes[:, 1], gt_nodes[:, 0], c='cyan', s=15, label='GT Nodes', zorder=2)
    ax2.set_title("Ground Truth")
    ax2.legend()
    ax2.axis('off')

    writer.add_figure('Val/Cleaning_Result', fig, global_step=epoch_step)
    plt.close(fig)

#
# --- 6. Main ---
#
def main():
    writer = SummaryWriter(LOG_DIR)
    print(f"Logs saved to: {LOG_DIR}")
    
    try:
        full_dataset = GNNLoadingDataset(root_dir=OUTPUT_DATASET_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); return

    # Verifica feature
    if len(full_dataset) > 0:
        if full_dataset.get(0).x.size(1) != 5:
            print("ATTENZIONE CRITICA: Attese 5 features.")
    
    IN_CHANNELS = 5 

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Pesi Manuali ---
    # Class 0 = Keep, Class 1 = Discard
    # Peso 2.5 su Discard significa che pulire è importante, ma non deve distruggere tutto.
    weights = torch.tensor([1.0, 2.5], dtype=torch.float).to(DEVICE)
    print(f"Weights -> Keep: {weights[0]:.2f}, Discard: {weights[1]:.2f}")
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    # --- Model Setup ---
    HIDDEN_CHANNELS = 128
    HEADS = 16
    LR = 0.00005 # Start low for BatchSize=1
    
    model = GATCleaner(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, heads=HEADS)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    
    # --- Scheduler ---
    # Se la Recall o la Loss non migliorano per 15 epoche, dimezza il LR
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    
    GLOBAL_STEP = 0
    
    print(f"Starting Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_with_logging(writer, model, optimizer, train_loader, GLOBAL_STEP, criterion)
        GLOBAL_STEP += len(train_loader)
        
        # Evaluate returns dictionary
        metrics = evaluate(model, val_loader, epoch, criterion)
        
        val_loss = metrics['loss']
        
        # Step scheduler based on Validation Loss
        scheduler.step(val_loss)
        
        # Logging Metriche Avanzate
        writer.add_scalar('Loss/Epoch_Val', val_loss, GLOBAL_STEP)
        writer.add_scalar('Metrics/Recall_Roads', metrics['recall_roads'], GLOBAL_STEP) # <--- WATCH THIS
        writer.add_scalar('Metrics/F1_Noise', metrics['f1_noise'], GLOBAL_STEP)
        writer.add_scalar('Hyperparams/LearningRate', optimizer.param_groups[0]['lr'], GLOBAL_STEP)

        print(f"Ep {epoch:03d} | Loss: {val_loss:.4f} | Recall Roads: {metrics['recall_roads']:.4f} | F1 Noise: {metrics['f1_noise']:.4f}")
        
        if epoch % 5 == 0 or epoch == 1:
            log_validation_image(writer, model, val_dataset, GLOBAL_STEP)
    
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
    writer.close()
    print(f"Training Done. Model saved to {CHECKPOINT_SAVE_PATH}")

if __name__ == '__main__':
    main()