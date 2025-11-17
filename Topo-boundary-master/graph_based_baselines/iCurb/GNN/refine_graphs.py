import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
import torchvision.transforms.functional as tvf
from tqdm import tqdm
import timm 
import skimage.io as io
from skimage import morphology, util
from PIL import Image
import pickle

# Import GNN e helper
import torch.nn.functional as F
from torch_geometric.nn import GATConv 
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra

# Import per la pulizia
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importa le definizioni dei modelli
from models.models_encoder import FPN 

# --- CONFIGURAZIONE ---
FPN_CHECKPOINT_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/checkpoints/seg_pretrain_manhattan_efficentnet_1.6_v2.pth"
GNN_CHECKPOINT_PATH = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/GNN/checkpoints/gnn_refiner_gat.pth"
DATA_SPLIT_JSON = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/data_split.json"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff" 
OUTPUT_VISUALIZATION_DIR = "./records/final_refined_graphs_3_panel" # Nuova cartella di Output

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.2
CLEANING_MIN_SIZE = 25 
LINK_CANDIDATE_DISTANCE = 80.0
SAMPLING_DISTANCE = 30
IMG_SIZE = 1000.0
NUM_CLASSES = 2
# ----------------------

#
# --- (Classi Vertex, Graph, GATRefiner e funzioni generate_graph, _get_edge_index - omesse per brevità) ---
# (Sono identiche al codice precedente)
#
class Vertex():
    def __init__(self,v):
        self.coord = v; self.index = v[0] * 1000 + v[1]
        self.neighbors = []; self.unprocessed_neighbors = []
        self.processed_neighbors = []; self.sampled_neighbors = []
        self.key_vertex = False
    def compare(self,v):
        if self.coord[0] == v[0] and self.coord[1] == v[1]: return True
        return False
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

def generate_graph(skeleton, file_name, graph_dir):
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
                if pre_v in curr_v.unprocessed_neighbors:
                     curr_v.unprocessed_neighbors.remove(pre_v)
    vertices = []; features = [] 
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        vertices.append([int(v.coord[0]), int(v.coord[1])])
        degree = v.pixel_degree 
        angle = 0.0; num_sampled_neighbors = len(v.sampled_neighbors)
        if num_sampled_neighbors == 1:
            n1 = v.sampled_neighbors[0]; dy = n1.coord[0] - v.coord[0]; dx = n1.coord[1] - v.coord[1]
            angle = np.arctan2(dy, dx)
        elif num_sampled_neighbors == 2:
            n1 = v.sampled_neighbors[0]; n2 = v.sampled_neighbors[1]
            dy = n2.coord[0] - n1.coord[0]; dx = n2.coord[1] - n1.coord[1]
            angle = np.arctan2(dy, dx)
        features.append([degree, angle])
    if not graph.sampled_vertices: return {'vertices':[], 'adj':np.array([]), 'features':[]}
    adjacent = np.ones((len(graph.sampled_vertices),len(graph.sampled_vertices))) * np.inf
    for v in graph.sampled_vertices:
        for u in v.sampled_neighbors:
            if u in graph.sampled_vertices: 
                dist = v.distance(u); adjacent[v.index,u.index] = dist; adjacent[u.index,v.index] = dist
    graph_data = {'vertices':vertices, 'adj':adjacent, 'features': features}
    return graph_data

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

def _get_edge_index(adj_matrix):
    if adj_matrix.size == 0:
        return torch.tensor([], dtype=torch.long).reshape(2, 0)
    row, col = np.where((adj_matrix != np.inf) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index

#
# --- Componente 5: Funzione di Visualizzazione Finale (MODIFICATA A 3 PANNELLI) ---
#
def save_final_visualization(original_rgb_img, 
                             all_pred_nodes, original_edges, 
                             nodes_to_keep_mask, links_to_add, 
                             output_path, name):
    
    # Crea la figura con 3 pannelli
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"Pipeline di Raffinamento per {name}", fontsize=16)
    
    all_pred_nodes_np = np.array(all_pred_nodes)
    nodes_kept = all_pred_nodes_np[nodes_to_keep_mask]
    nodes_discarded = all_pred_nodes_np[~nodes_to_keep_mask]

    # --- Pannello 1: Grafo Sporco (Input GNN) ---
    ax1.imshow(original_rgb_img)
    if len(all_pred_nodes_np) > 0:
        ax1.scatter(all_pred_nodes_np[:, 1], all_pred_nodes_np[:, 0], c='red', s=5, label='Nodi Sporchi')
    for (n1_idx, n2_idx) in original_edges:
        if n1_idx < len(all_pred_nodes_np) and n2_idx < len(all_pred_nodes_np):
            n1, n2 = all_pred_nodes_np[n1_idx], all_pred_nodes_np[n2_idx]
            ax1.plot([n1[1], n2[1]], [n1[0], n2[0]], color='cyan', linewidth=1.0, alpha=0.5) 
    ax1.set_title("1. Input GNN: Grafo Sporco (con Rumore)")
    ax1.axis('off')

    # --- Pannello 2: Predizioni GNN (Debug) ---
    ax2.imshow(original_rgb_img)
    if len(all_pred_nodes_np) > 0:
        # Mostra TUTTI i nodi, colorati per decisione
        ax2.scatter(nodes_discarded[:, 1], nodes_discarded[:, 0], c='red', s=5, label='Nodi Scartati')
        ax2.scatter(nodes_kept[:, 1], nodes_kept[:, 0], c='lime', s=5, label='Nodi Mantenuti')
        # Mostra i link che la GNN vuole aggiungere
        for (n1_idx, n2_idx) in links_to_add:
             if n1_idx < len(all_pred_nodes_np) and n2_idx < len(all_pred_nodes_np):
                n1, n2 = all_pred_nodes_np[n1_idx], all_pred_nodes_np[n2_idx]
                ax2.plot([n1[1], n2[1]], [n1[0], n2[0]], color='magenta', linewidth=2, linestyle='--', label='Link Riparati')
    ax2.set_title("2. Predizioni GNN (Debug)")
    ax2.axis('off')

    # --- Pannello 3: Grafo Finale (Output) ---
    ax3.imshow(original_rgb_img)
    if len(all_pred_nodes_np) > 0:
        # Mostra SOLO i nodi Mantenuti
        ax3.scatter(nodes_kept[:, 1], nodes_kept[:, 0], c='lime', s=10, label='Nodi Raffinati')
        
        # Mostra SOLO gli archi Mantenuti
        for (n1_idx, n2_idx) in original_edges:
            if nodes_to_keep_mask[n1_idx] and nodes_to_keep_mask[n2_idx]:
                n1, n2 = all_pred_nodes_np[n1_idx], all_pred_nodes_np[n2_idx]
                ax3.plot([n1[1], n2[1]], [n1[0], n2[0]], color='lime', linewidth=1.5)
        
        # Mostra SOLO gli archi Riparati
        for (n1_idx, n2_idx) in links_to_add:
            if n1_idx < len(all_pred_nodes_np) and n2_idx < len(all_pred_nodes_np) and nodes_to_keep_mask[n1_idx] and nodes_to_keep_mask[n2_idx]:
                n1, n2 = all_pred_nodes_np[n1_idx], all_pred_nodes_np[n2_idx]
                ax3.plot([n1[1], n2[1]], [n1[0], n2[0]], color='magenta', linewidth=2, linestyle='--')
                
    ax3.set_title("3. Output: Grafo Finale Raffinato")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close(fig)

#
# --- Componente 6: Main (La Pipeline di Inferenza) ---
#
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Carica Modello FPN
    print(f"Loading FPN model from: {FPN_CHECKPOINT_PATH}")
    fpn_model = FPN(n_channels=4, n_classes=1) 
    if torch.cuda.is_available():
        fpn_model.load_state_dict(torch.load(FPN_CHECKPOINT_PATH))
    else:
        fpn_model.load_state_dict(torch.load(FPN_CHECKPOINT_PATH, map_location=torch.device('cpu')))
    fpn_model.to(DEVICE); fpn_model.eval() 
    print("FPN model ready.")

    # 2. Carica Modello GNN
    print(f"Loading GNN model from: {GNN_CHECKPOINT_PATH}")
    IN_CHANNELS = 4 
    HIDDEN_CHANNELS = 16
    NUM_HEADS = 4
    gnn_model = GATRefiner(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, heads=NUM_HEADS)
    if torch.cuda.is_available():
        gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH))
    else:
        gnn_model.load_state_dict(torch.load(GNN_CHECKPOINT_PATH, map_location=torch.device('cpu')))
    gnn_model.to(DEVICE); gnn_model.eval()
    print("GNN model ready.")

    # 3. Leggi la lista di Test
    print(f"Reading test list from: {DATA_SPLIT_JSON}")
    try:
        with open(DATA_SPLIT_JSON, 'r') as f:
            test_list = json.load(f)['test'] 
    except FileNotFoundError:
        print(f"ERROR: Data split JSON not found at {DATA_SPLIT_JSON}"); return
        
    print(f"Found {len(test_list)} test images.")
    os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True) 

    print(f"Starting inference pipeline...")
    
    with torch.no_grad(): 
        for name in tqdm(test_list, desc="Refining Test Graphs"):
            try:
                # --- PASSO 1: Carica Immagine ---
                img_path = os.path.join(IMAGE_DIR, f"{name}.tiff")
                img_full_tiff = Image.open(img_path) 
                original_rgb_array = np.array(img_full_tiff.convert('RGB'))
                tiff_for_model = tvf.to_tensor(img_full_tiff).to(DEVICE)
                tiff_batch = tiff_for_model.unsqueeze(0) 
                
                # --- PASSO 2: Esegui FPN (Segmentazione) ---
                predictions, _ = fpn_model(tiff_batch)
                pred_mask = torch.sigmoid(predictions)
                pred_mask_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
                
                # --- PASSO 3: Crea Scheletro Sporco ---
                binary_mask_dirty = (pred_mask_np > THRESHOLD)
                cleaned_mask = remove_small_objects(binary_mask_dirty, min_size=CLEANING_MIN_SIZE)
                filled_mask = binary_fill_holes(cleaned_mask)
                skeleton = morphology.skeletonize(filled_mask, method='lee')
                skeleton_salvabile = util.img_as_ubyte(skeleton)
                
                # --- PASSO 4: Crea Grafo Sporco (con 4 feature) ---
                graph_data_dirty = generate_graph(skeleton_salvabile, "", None) 
                
                all_pred_nodes = graph_data_dirty.get('vertices', [])
                all_pred_features = graph_data_dirty.get('features', [])
                all_pred_adj = graph_data_dirty.get('adj', np.array([]))

                if len(all_pred_nodes) == 0:
                    tqdm.write(f"  -> Img: {name} | Grafo vuoto. Salto.")
                    continue
                
                # --- PASSO 5: Prepara i Dati per la GNN ---
                x_features = torch.cat([
                    torch.tensor(all_pred_nodes, dtype=torch.float) / IMG_SIZE, 
                    torch.tensor(all_pred_features, dtype=torch.float)         
                ], dim=1)
                edge_index = _get_edge_index(all_pred_adj)
                
                data = Data(x=x_features, edge_index=edge_index).to(DEVICE)
                
                # --- PASSO 6: Esegui GNN (Raffinamento) ---
                node_pred_logits, link_emb = gnn_model(data.x, data.edge_index)
                
                # --- PASSO 7: Decodifica Predizioni GNN ---
                node_pred_labels = torch.argmax(node_pred_logits, dim=1).cpu().numpy()
                nodes_to_keep_mask = (node_pred_labels == 0)
                
                endpoints = []
                for i in range(len(all_pred_nodes)):
                    if all_pred_features[i][0] <= 1: # Grado 0 o 1
                        endpoints.append(i)
                
                links_to_add = []
                if len(endpoints) >= 2:
                    endpoint_coords = np.array([all_pred_nodes[i] for i in endpoints])
                    dist_matrix = squareform(pdist(endpoint_coords))
                    dist_matrix[dist_matrix == 0] = np.inf 
                    
                    candidate_pairs_set = set()
                    map_g1_to_all = {node_idx: i for i, node_idx in enumerate(endpoints)}
                    for i, node_i_idx in enumerate(endpoints):
                        j = np.argmin(dist_matrix[i, :])
                        dist = dist_matrix[i, j]
                        if dist < LINK_CANDIDATE_DISTANCE:
                            node_j_idx = endpoints[j]
                            pair = tuple(sorted((node_i_idx, node_j_idx)))
                            candidate_pairs_set.add(pair)
                    
                    candidate_pairs = list(candidate_pairs_set)
                    
                    if candidate_pairs:
                        link_label_index = torch.tensor(candidate_pairs, dtype=torch.long).t().contiguous().to(DEVICE)
                        src, dst = link_label_index
                        emb_src = link_emb[src]; emb_dst = link_emb[dst]
                        emb_pair = torch.cat([emb_src, emb_dst], dim=1) 
                        link_pred_score = gnn_model.link_pred_head(emb_pair).squeeze(-1) 
                        links_to_add_mask = (torch.sigmoid(link_pred_score) > 0.5).cpu().numpy()
                        links_to_add = np.array(candidate_pairs)[links_to_add_mask]

                # --- PASSO 8: Salva Visualizzazione Finale ---
                png_name = f"{name}.png"
                output_path_viz = os.path.join(OUTPUT_VISUALIZATION_DIR, png_name)
                original_edges = data.edge_index.t().cpu().numpy()
                
                save_final_visualization(
                    original_rgb_array, 
                    all_pred_nodes, 
                    original_edges,
                    nodes_to_keep_mask, 
                    links_to_add, 
                    output_path_viz, 
                    name
                )
                
                tqdm.write(f"  -> Img: {name} | Nodi Raffinati: {np.sum(nodes_to_keep_mask)}/{len(all_pred_nodes)} | Link Aggiunti: {len(links_to_add)}")
                
            except FileNotFoundError:
                print(f"Warning: Image not found, skipped: {img_path}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\nInference pipeline complete.")
    print(f"Final refined visualizations saved to: {OUTPUT_VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()