import sys
import os
# Aggiunge la cartella padre (iCurb) al path di sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import torchvision.transforms.functional as tvf
from tqdm import tqdm
import skimage.io as io
from skimage import morphology, util
from PIL import Image
import pickle

# Import GNN
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # <--- IMPORTANTE: GATv2
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Import pulizia immagini
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importa FPN
from models.models_encoder import FPN 

# --- CONFIGURAZIONE ---
# Percorsi
FPN_CHECKPOINT_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/checkpoints/seg_pretrain_manhattan_efficentnet_1.6_v2.pth"
GNN_CHECKPOINT_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/GNN/checkpoints/gnn_cleaner_gat.pth" # Il modello appena addestrato

DATA_SPLIT_JSON = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/data_split.json"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff" 

# Cartelle di Output
OUTPUT_RL_READY_DIR = "./records/rl_ready_graphs"    # Qui andranno i file per l'RL
OUTPUT_VISUALIZATION_DIR = "./records/final_refinement_viz" # Qui le immagini per noi

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parametri Segmentazione
THRESHOLD = 0.2
CLEANING_MIN_SIZE = 25 
SAMPLING_DISTANCE = 30
IMG_SIZE = 1000.0

# Parametri GNN (DEVONO ESSERE UGUALI A TRAIN_GNN.PY)
IN_CHANNELS = 5
HIDDEN_CHANNELS = 128
HEADS = 16
NUM_CLASSES = 2
# ----------------------

# --- 1. Classi Grafo (Standard) ---
class Vertex():
    def __init__(self,v):
        self.coord = v; self.index = v[0] * 1000 + v[1]
        self.neighbors = []; self.unprocessed_neighbors = []
        self.processed_neighbors = []; self.sampled_neighbors = []
        self.key_vertex = False
    def compare(self,v):
        return self.coord[0] == v[0] and self.coord[1] == v[1]
    def next(self,previous):
        neighbors = self.neighbors.copy()
        if previous in neighbors: neighbors.remove(previous)
        if not neighbors: return None
        return neighbors[0]
    def distance(self,v):
        return pow(pow(self.coord[0]-v.coord[0],2)+pow(self.coord[1]-v.coord[1],2),0.5)

class Graph():
    def __init__(self):
        self.vertices = []; self.key_vertices = []; self.sampled_vertices = []
    def find_vertex(self,index):
        for v in self.vertices:
            if index == v.index: return v
        return None
    def add_v(self,v,neighbors):
        self.vertices.append(v)
        for n in neighbors:
            index = n[0] * 1000 + n[1]
            u = self.find_vertex(index)
            if u is not None:
                u.neighbors.append(v); v.neighbors.append(u)
                u.unprocessed_neighbors.append(v); v.unprocessed_neighbors.append(u)
    def find_key_vertices(self):
        for v in self.vertices:
            v.pixel_degree = len(v.neighbors) 
            if v.pixel_degree != 2:
                v.key_vertex = True; self.key_vertices.append(v); self.sampled_vertices.append(v)

# --- 2. Generazione Grafo Sporco (5 Feature) ---
def generate_graph(skeleton, pred_mask):
    def find_neighbors(v,img,remove=False):
        output_v = []; H, W = img.shape
        def get_pixel_value(u):
            if u[0] < 0 or u[0] >= H or u[1] < 0 or u[1] >= W: return
            if img[u[0],u[1]]: output_v.append(u)
        get_pixel_value([v[0]+1,v[1]]); get_pixel_value([v[0]-1,v[1]])
        get_pixel_value([v[0],v[1]-1]); get_pixel_value([v[0],v[1]+1])
        get_pixel_value([v[0]+1,v[1]-1]); get_pixel_value([v[0]+1,v[1]+1])
        get_pixel_value([v[0]-1,v[1]-1]); get_pixel_value([v[0]-1,v[1]+1])
        if remove: img[v[0],v[1]] = 0
        return output_v

    graph = Graph(); img = skeleton.copy() 
    if np.sum(img) == 0: return {'vertices':[], 'adj':np.array([]), 'features':[]}
    
    pre_points = np.where(img!=0)
    pre_points = [[pre_points[0][i],pre_points[1][i]] for i in range(len(pre_points[0]))]
    
    for point in pre_points:
        v = Vertex(point); graph.add_v(v,find_neighbors(point,img))
    
    graph.find_key_vertices() 
    
    # Sampling logic
    for key_vertex in graph.key_vertices:
        if len(key_vertex.unprocessed_neighbors):
            for neighbor in key_vertex.unprocessed_neighbors.copy(): 
                if neighbor not in key_vertex.unprocessed_neighbors: continue
                key_vertex.unprocessed_neighbors.remove(neighbor)
                curr_v = neighbor; pre_v = key_vertex; sampled_v = key_vertex; counter = 1
                while(not curr_v.key_vertex):
                    if counter % SAMPLING_DISTANCE == 0:
                        sampled_v.sampled_neighbors.append(curr_v)
                        curr_v.sampled_neighbors.append(sampled_v)
                        sampled_v = curr_v
                        if not sampled_v.key_vertex: graph.sampled_vertices.append(sampled_v)
                    next_v = curr_v.next(pre_v)
                    if next_v is None: break 
                    pre_v = curr_v; curr_v = next_v; counter += 1
                sampled_v.sampled_neighbors.append(curr_v); curr_v.sampled_neighbors.append(sampled_v)
                if pre_v in curr_v.unprocessed_neighbors: curr_v.unprocessed_neighbors.remove(pre_v)

    vertices = []; features = [] 
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        vertices.append([int(v.coord[0]), int(v.coord[1])])
        
        cy, cx = float(v.coord[0]), float(v.coord[1])
        degree = float(v.pixel_degree)
        angle = 0.0; num = len(v.sampled_neighbors)
        if num == 1:
            n1 = v.sampled_neighbors[0]
            angle = np.arctan2(n1.coord[0]-cy, n1.coord[1]-cx)
        elif num == 2:
            n1, n2 = v.sampled_neighbors[0], v.sampled_neighbors[1]
            angle = np.arctan2(n2.coord[0]-n1.coord[0], n2.coord[1]-n1.coord[1])
        intensity = float(pred_mask[int(cy), int(cx)])
        
        features.append([cy, cx, degree, angle, intensity])

    if not graph.sampled_vertices: return {'vertices':[], 'adj':np.array([]), 'features':[]}
    
    adjacent = np.ones((len(graph.sampled_vertices),len(graph.sampled_vertices))) * np.inf
    for v in graph.sampled_vertices:
        for u in v.sampled_neighbors: 
            if u in graph.sampled_vertices: 
                dist = v.distance(u)
                adjacent[v.index,u.index] = dist; adjacent[u.index,v.index] = dist
                
    return {'vertices':vertices, 'adj':adjacent, 'features': features}

# --- 3. Definizione Modello GNN (GATv2Aggiornata) ---
class GATCleaner(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(GATCleaner, self).__init__()
        # Deve essere identica a quella usata in train_gnn.py
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.node_cls_head = torch.nn.Linear(hidden_channels, NUM_CLASSES) 

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return self.node_cls_head(x)

def _get_edge_index(adj_matrix):
    if adj_matrix.size == 0: return torch.tensor([], dtype=torch.long).reshape(2, 0)
    row, col = np.where((adj_matrix != np.inf) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
    return to_undirected(edge_index)

# --- 4. Crea Grafo Pulito (Elimina Nodi) ---
def create_clean_graph(dirty_graph_data, keep_mask):
    old_vertices = dirty_graph_data['vertices']
    old_adj = dirty_graph_data['adj']
    old_features = dirty_graph_data['features']
    
    keep_indices = np.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        return {'vertices': [], 'adj': np.array([]), 'features': []}
    
    new_vertices = [old_vertices[i] for i in keep_indices]
    new_features = [old_features[i] for i in keep_indices] 
    
    # Sotto-matrice adiacenza
    new_adj = old_adj[np.ix_(keep_indices, keep_indices)]
    
    return {'vertices': new_vertices, 'adj': new_adj, 'features': new_features}

# --- 5. Visualizzazione ---
def save_final_visualization(rgb_img, dirty_data, clean_data, mask, output_path, name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"Pipeline Refinement: {name}", fontsize=16)
    
    dirty_nodes = np.array(dirty_data['vertices'])
    
    # 1. Input
    ax1.imshow(rgb_img)
    if len(dirty_nodes) > 0:
        ax1.scatter(dirty_nodes[:, 1], dirty_nodes[:, 0], c='red', s=10, label='Input Nodes')
    ax1.set_title("1. Input Sporco (FPN)")
    
    # 2. GNN Decision
    ax2.imshow(rgb_img)
    if len(dirty_nodes) > 0:
        kept = dirty_nodes[mask]
        discarded = dirty_nodes[~mask]
        ax2.scatter(discarded[:, 1], discarded[:, 0], c='red', s=10, label='Rumore')
        ax2.scatter(kept[:, 1], kept[:, 0], c='lime', s=10, label='Strada')
    ax2.set_title("2. Pulizia GNN")
    ax2.legend()
    
    # 3. Output
    ax3.imshow(rgb_img)
    clean_nodes = np.array(clean_data['vertices'])
    if len(clean_nodes) > 0:
        ax3.scatter(clean_nodes[:, 1], clean_nodes[:, 0], c='lime', s=15, label='Nodi Puliti')
        # Mostra collegamenti rimasti
        adj = clean_data['adj']
        r, c = np.where((adj != np.inf) & (np.triu(np.ones_like(adj), k=1).astype(bool)))
        for i in range(len(r)):
            p1, p2 = clean_nodes[r[i]], clean_nodes[c[i]]
            ax3.plot([p1[1], p2[1]], [p1[0], p2[0]], 'lime', lw=1.0)
    ax3.set_title("3. Output Finale")
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# --- Main ---
def main():
    print(f"Using device: {DEVICE}")
    
    # Load FPN
    fpn = FPN(n_channels=4, n_classes=1).to(DEVICE)
    fpn.load_state_dict(torch.load(FPN_CHECKPOINT_PATH, map_location=DEVICE))
    fpn.eval()

    # Load GNN
    print(f"Loading GNN from {GNN_CHECKPOINT_PATH}...")
    gnn = GATCleaner(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, heads=HEADS).to(DEVICE)
    gnn.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, map_location=DEVICE))
    gnn.eval()

    os.makedirs(OUTPUT_RL_READY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True)
    
    with open(DATA_SPLIT_JSON, 'r') as f:
        test_list = json.load(f)['test']

    print(f"Inizio pulizia su {len(test_list)} immagini di Test...")
    
    with torch.no_grad():
        for name in tqdm(test_list):
            try:
                # 1. Immagine -> Heatmap
                img_path = os.path.join(IMAGE_DIR, f"{name}.tiff")
                img_pil = Image.open(img_path)
                img_rgb = np.array(img_pil.convert('RGB'))
                t_in = tvf.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
                preds, _ = fpn(t_in)
                heatmap = torch.sigmoid(preds).squeeze().cpu().numpy()
                
                # 2. Heatmap -> Grafo Sporco
                mask = (heatmap > THRESHOLD)
                mask = remove_small_objects(mask, min_size=CLEANING_MIN_SIZE)
                mask = binary_fill_holes(mask)
                skel = util.img_as_ubyte(morphology.skeletonize(mask, method='lee'))
                
                graph_dirty = generate_graph(skel, heatmap)
                if not graph_dirty['vertices']: continue

                # 3. Grafo Sporco -> GNN
                feats = graph_dirty['features']
                x_tensor = torch.tensor(feats, dtype=torch.float)
                x_tensor[:, 0] /= IMG_SIZE; x_tensor[:, 1] /= IMG_SIZE
                edge_index = _get_edge_index(graph_dirty['adj'])
                data_gnn = Data(x=x_tensor, edge_index=edge_index).to(DEVICE)
                
                # 4. GNN -> Maschera Pulizia
                logits = gnn(data_gnn.x, data_gnn.edge_index)
                keep_mask = (torch.argmax(logits, dim=1) == 0).cpu().numpy() # 0 = Keep
                
                # 5. Maschera -> Grafo Pulito
                graph_clean = create_clean_graph(graph_dirty, keep_mask)
                
                # 6. Salvataggio
                with open(os.path.join(OUTPUT_RL_READY_DIR, f"{name}.pickle"), 'wb') as f:
                    pickle.dump(graph_clean, f)
                    
                # Visualizza solo ogni tanto per velocità (o togli l'if per vederle tutte)
                save_final_visualization(img_rgb, graph_dirty, graph_clean, keep_mask, 
                                         os.path.join(OUTPUT_VISUALIZATION_DIR, f"{name}.png"), name)
                
            except Exception as e:
                print(f"Error {name}: {e}")

    print(f"\nFinito! I grafi puliti sono in: {OUTPUT_RL_READY_DIR}")

if __name__ == "__main__":
    main()