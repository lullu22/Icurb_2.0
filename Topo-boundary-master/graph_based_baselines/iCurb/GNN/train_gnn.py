import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv 
from torch_geometric.data import DataLoader, Dataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
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
# Input
OUTPUT_DATASET_DIR = "./gnn_dataset_20_80"  
PRED_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/pred_graph_2"
GT_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/gt_graphs_2"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"

# Output
LOG_DIR = './runs/gnn_refiner_GAT_experiment_1'
CHECKPOINT_SAVE_PATH = './checkpoints/gnn_refiner_gat.pth'

# Parametri
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.20
NUM_CLASSES = 2 
IMG_SIZE = 1000.0
EPOCHS = 200
# ------------------------------------------------------------------

#
# --- 1. Classe di Caricamento Dati (Corretta) ---
# (Immutata)
#
class GNNLoadingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(GNNLoadingDataset, self).__init__(root_dir, transform)
        self.root = root_dir
        self.pt_files = sorted([f for f in os.listdir(self.root) if f.endswith('.pt')])
        if not self.pt_files:
            raise FileNotFoundError(f"Nessun file .pt trovato in {self.root}.")
        print(f"Dataset Loader: Trovati {len(self.pt_files)} campioni di dati (.pt) elaborati.")

    def len(self):
        return len(self.pt_files)
        
    def get(self, idx):
        path = os.path.join(self.root, self.pt_files[idx])
        data = torch.load(path)
        return data

#
# --- 2. Definizione del Modello: GATRefiner (immutato) ---
#
class GATRefiner(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(GATRefiner, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.node_cls_head = torch.nn.Linear(hidden_channels, NUM_CLASSES) 
        self.link_pred_head = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        node_output = self.node_cls_head(x); link_embedding = x
        return node_output, link_embedding

#
# --- 3. Funzione di Validazione (EVALUATE) ---
#
def evaluate(model, data_loader, epoch, node_criterion, link_criterion):
    model.eval() 
    total_loss = 0; total_samples = 0
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Validation E{epoch:03d}"):
            data = data.to(DEVICE)
            node_pred, link_emb = model(data.x, data.edge_index)
            
            loss_node = node_criterion(node_pred, data.y)
            
            src, dst = data.edge_label_index
            if src.numel() > 0:
                emb_src = link_emb[src]; emb_dst = link_emb[dst]
                emb_pair = torch.cat([emb_src, emb_dst], dim=1) 
                link_pred_score = model.link_pred_head(emb_pair).squeeze(-1) 
                link_target = data.edge_label.float()
                loss_link = link_criterion(link_pred_score, link_target)
            else:
                loss_link = torch.tensor(0.0, device=DEVICE)
                
            total_combined_loss = loss_node + loss_link
            total_loss += total_combined_loss.item()
            total_samples += 1
            
    return total_loss / max(total_samples, 1)

#
# --- 4. Funzione di Training con Logging ---
#
def train_with_logging(writer, model, optimizer, data_loader, global_step_start, node_criterion, link_criterion):
    model.train()
    total_loss = 0; total_samples = 0
    global_step = global_step_start
    
    for data in tqdm(data_loader, desc="Training GNN"):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        node_pred, link_emb = model(data.x, data.edge_index)
        
        loss_node = node_criterion(node_pred, data.y)
        
        src, dst = data.edge_label_index
        if src.numel() > 0:
            emb_src = link_emb[src]; emb_dst = link_emb[dst]
            emb_pair = torch.cat([emb_src, emb_dst], dim=1) 
            link_pred_score = model.link_pred_head(emb_pair).squeeze(-1) 
            link_target = data.edge_label.float()
            loss_link = link_criterion(link_pred_score, link_target)
        else:
            loss_link = torch.tensor(0.0, device=DEVICE)
            
        total_combined_loss = loss_node + loss_link
        
        total_combined_loss.backward()
        optimizer.step()
        
        total_loss += total_combined_loss.item()
        total_samples += 1
        
        writer.add_scalar('Loss/Step_Total', total_combined_loss.item(), global_step)
        global_step += 1
        
    writer.add_scalar('Loss/Epoch_Average_Train', total_loss / max(total_samples, 1), global_step_start)
    return total_loss / max(total_samples, 1)

#
# --- 5. Funzione di Logging Immagini (immutata) ---
#
def log_validation_image(writer, model, val_dataset, epoch_step):
    print(f"Generating validation image for epoch (step {epoch_step})...")
    model.eval()
    
    idx = torch.randint(0, len(val_dataset), (1,)).item()
    data = val_dataset[idx]
    
    try:
        file_name = data.file_name
    except AttributeError:
        print("ERRORE: 'file_name' non trovato nel file .pt. Riesegui create_gnn_dataset.py.")
        return

    tiff_path = os.path.join(IMAGE_DIR, file_name.replace('.pickle', '.tiff'))
    gt_graph_path = os.path.join(GT_GRAPH_DIR, file_name)
    
    try:
        rgb_img = np.array(Image.open(tiff_path).convert('RGB'))
        with open(gt_graph_path, 'rb') as f:
            gt_graph = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load data for visualization: {e}")
        return

    data_gpu = data.to(DEVICE)
    with torch.no_grad():
        node_pred_logits, link_emb = model(data_gpu.x, data_gpu.edge_index)
    
    node_pred_labels = torch.argmax(node_pred_logits, dim=1).cpu().numpy()
    nodes_to_keep_mask = (node_pred_labels == 0)
    
    src, dst = data.edge_label_index
    if src.numel() > 0:
        emb_src = link_emb[src]; emb_dst = link_emb[dst]
        emb_pair = torch.cat([emb_src, emb_dst], dim=1) 
        link_pred_score = model.link_pred_head(emb_pair).squeeze(-1) 
        links_to_add_mask = (torch.sigmoid(link_pred_score) > 0.5).cpu().numpy()
        candidate_pairs = data.edge_label_index.t().cpu().numpy()
        links_to_add = candidate_pairs[links_to_add_mask]
    else:
        links_to_add = []

    all_pred_nodes = data.x.cpu().numpy()[:, :2] * IMG_SIZE 
    gt_nodes = np.array(gt_graph['vertices'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"Validation @ Step {epoch_step} - {data.file_name}", fontsize=16)

    # Pannello 1: Grafo Sporco (Input)
    ax1.imshow(rgb_img)
    if len(all_pred_nodes) > 0:
        ax1.scatter(all_pred_nodes[:, 1], all_pred_nodes[:, 0], c='red', s=5, label='Nodi Sporchi')
    original_edges = data.edge_index.t().cpu().numpy()
    for (n1_idx, n2_idx) in original_edges:
        if n1_idx < len(all_pred_nodes) and n2_idx < len(all_pred_nodes):
            n1, n2 = all_pred_nodes[n1_idx], all_pred_nodes[n2_idx]
            ax1.plot([n1[1], n2[1]], [n1[0], n2[0]], color='cyan', linewidth=1.0, alpha=0.5) 
    ax1.set_title("Input: Grafo Sporco (da FPN)")
    ax1.axis('off')

    # Pannello 2: Grafo Raffinato (Output GNN)
    ax2.imshow(rgb_img)
    if len(all_pred_nodes) > 0:
        nodes_kept = all_pred_nodes[nodes_to_keep_mask]
        nodes_discarded = all_pred_nodes[~nodes_to_keep_mask]
        ax2.scatter(nodes_discarded[:, 1], nodes_discarded[:, 0], c='red', s=5, label='Nodi Scartati (Label 1)')
        ax2.scatter(nodes_kept[:, 1], nodes_kept[:, 0], c='lime', s=5, label='Nodi Mantenuti (Label 0)')
        for (n1_idx, n2_idx) in original_edges:
            if nodes_to_keep_mask[n1_idx] and nodes_to_keep_mask[n2_idx]:
                n1, n2 = all_pred_nodes[n1_idx], all_pred_nodes[n2_idx]
                ax2.plot([n1[1], n2[1]], [n1[0], n2[0]], color='lime', linewidth=1.0)
    for (n1_idx, n2_idx) in links_to_add:
        if n1_idx < len(all_pred_nodes) and n2_idx < len(all_pred_nodes) and nodes_to_keep_mask[n1_idx] and nodes_to_keep_mask[n2_idx]:
            n1, n2 = all_pred_nodes[n1_idx], all_pred_nodes[n2_idx]
            ax2.plot([n1[1], n2[1]], [n1[0], n2[0]], color='magenta', linewidth=2, linestyle='--')
    ax2.set_title("Output: Grafo Raffinato (dalla GNN)")
    ax2.axis('off')

    # Pannello 3: Grafo GT (Target)
    ax3.imshow(rgb_img)
    if len(gt_nodes) > 0:
        ax3.scatter(gt_nodes[:, 1], gt_nodes[:, 0], c='cyan', s=5, label='Nodi GT')
    gt_adj = gt_graph['adj']
    for i in range(len(gt_nodes)):
        for j in range(i + 1, len(gt_nodes)):
            if gt_adj[i, j] != np.inf:
                n1, n2 = gt_nodes[i], gt_nodes[j]
                ax3.plot([n1[1], n2[1]], [n1[0], n2[0]], color='cyan', linewidth=1.0)
    ax3.set_title("Target: Grafo Ground Truth")
    ax3.axis('off')

    writer.add_figure('Validation/Epoch_Visualization', fig, global_step=epoch_step)
    plt.close(fig)
    print(f"Immagine di validazione per lo step {epoch_step} salvata su TensorBoard.")

#
# --- 6. Main Execution (MODIFICATA CON BOOST E PARAMETRI) ---
#
def main():
    writer = SummaryWriter(LOG_DIR)
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")
    print(f"Using device: {DEVICE}")
    
    try:
        full_dataset = GNNLoadingDataset(root_dir=OUTPUT_DATASET_DIR)
    except FileNotFoundError as e:
        print(f"ERRORE FATALE: {e}")
        writer.close()
        return

    in_channels = full_dataset.get(0).x.size(1) 
    if in_channels == 2:
        print(f"ATTENZIONE: Trovate 2 feature. Il modello si baserà solo su [y, x].")
    elif in_channels == 4:
        print(f"INFO: Trovate 4 feature [y, x, degree, angle]. Corretto.")
    else:
        print(f"ATTENZIONE: Trovate {in_channels} feature. Previste 2 o 4.")


    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices) 
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Dataset diviso: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Calcolo dei Pesi per la Loss (Class Weighting) ---
    print("Calcolo dei pesi per lo sbilanciamento delle classi...")
    all_node_labels = []
    all_link_labels = []
    
    for i in tqdm(range(len(train_dataset)), desc="Analisi Dataset"):
        data = train_dataset[i]
        all_node_labels.append(data.y)
        all_link_labels.append(data.edge_label)
    
    all_node_labels = torch.cat(all_node_labels)
    all_link_labels = torch.cat(all_link_labels)

    # 1. Pesi per i Nodi (0=Buono, 1=Spazzatura)
    num_nodes_total = all_node_labels.size(0)
    num_nodes_0 = (all_node_labels == 0).sum().item()
    num_nodes_1 = (all_node_labels == 1).sum().item()
    
    if num_nodes_0 == 0 or num_nodes_1 == 0:
        print("ATTENZIONE: Il dataset di training contiene solo una classe di nodi.")
        node_weights = torch.tensor([1.0, 1.0]).to(DEVICE)
    else:
        weight_class_0 = num_nodes_total / num_nodes_0
        weight_class_1 = num_nodes_total / num_nodes_1
        node_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float).to(DEVICE)

    print(f"Nodi: 'Buoni' (0): {num_nodes_0}, 'Spazzatura' (1): {num_nodes_1}")
    print(f"Pesi Loss Nodi: {node_weights.cpu().numpy()}")

    # 2. Pesi per i Link (0=Non Connettere, 1=Connetti)
    num_links_total = all_link_labels.size(0)
    num_links_1 = (all_link_labels == 1).sum().item()
    num_links_0 = num_links_total - num_links_1
    
    if num_links_0 == 0 or num_links_1 == 0:
        print("ATTENZIONE: Il dataset di training contiene solo una classe di link.")
        link_pos_weight = torch.tensor(1.0).to(DEVICE)
    else:
        # --- MODIFICA: Aggiungi BOOST ---
        calculated_weight = num_links_0 / num_links_1
        MANUAL_BOOST_FACTOR = 5.0 # <-- Aumenta questo se vedi ancora pochi link
        
        link_pos_weight = torch.tensor(calculated_weight * MANUAL_BOOST_FACTOR, dtype=torch.float).to(DEVICE)
        print(f"Link: 'Non Connettere' (0): {num_links_0}, 'Connetti' (1): {num_links_1}")
        print(f"Peso Positivo Loss Link (con boost {MANUAL_BOOST_FACTOR}x): {link_pos_weight.cpu().item():.2f}")
        # --- FINE MODIFICA ---
    
    # Inizializzazione Criteri di Loss (con i pesi)
    node_criterion = torch.nn.CrossEntropyLoss(weight=node_weights)
    link_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=link_pos_weight)
    
    # --- MODIFICA: Parametri Modello ---
    HIDDEN_CHANNELS = 64 # <-- Aumentato
    NUM_HEADS = 16        # <-- Aumentato
    LEARNING_RATE = 0.0005 # <-- Abbassato
    WEIGHT_DECAY = 5e-4
    # --- FINE MODIFICA ---
    
    model = GATRefiner(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, heads=NUM_HEADS)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    EPOCHS = 200
    GLOBAL_STEP = 0

    print(f"Inizio training GNN (GAT) per {EPOCHS} epoche...")
    
    for epoch in range(1, EPOCHS + 1):
        # Passa i criteri pesati alle funzioni
        train_loss = train_with_logging(writer, model, optimizer, train_loader, GLOBAL_STEP, node_criterion, link_criterion) 
        GLOBAL_STEP += len(train_loader) 
        val_loss = evaluate(model, val_loader, epoch, node_criterion, link_criterion)
        
        writer.add_scalar('Loss/Epoch_Average_Train', train_loss, GLOBAL_STEP)
        writer.add_scalar('Loss/Epoch_Validation', val_loss, GLOBAL_STEP)
        
        print(f"Epoca {epoch:03d} | Step: {GLOBAL_STEP} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        log_validation_image(writer, model, val_dataset, GLOBAL_STEP)
             
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
    writer.close()
    print(f"\nTraining completo. Modello salvato in: {CHECKPOINT_SAVE_PATH}")

if __name__ == '__main__':
    main()