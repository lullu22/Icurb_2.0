import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- Imports for Visualization ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
# -------------------------------

# --- Configuration ---
PRED_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/pred_graph_2" 
GT_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/gt_graphs_2"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"

OUTPUT_DATASET_DIR = "./gnn_dataset_cleaner_20" 
OUTPUT_NODE_VIZ_DIR = "./gnn_dataset_node_viz" 

# --- FLAG DI VISUALIZZAZIONE ---
CREATE_VISUALIZATIONS = False  # Imposta a True per vedere le linee di debug
# --------------------------------

NODE_MATCH_THRESHOLD = 20.0 
IMG_SIZE = 1000.0
# ---------------------

#
# --- Visualization Function (DEBUG LOGIC) ---
#
def save_debug_visualization(rgb_img, pred_nodes, gt_nodes, matches, distances, labels, output_path, name):
    """
    Mostra i nodi predetti collegati al loro nearest neighbor nel GT.
    Verde = Entro la soglia (Keep).
    Rosso = Fuori soglia (Discard).
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img, zorder=0)
    
    pred_nodes_np = np.array(pred_nodes)
    gt_nodes_np = np.array(gt_nodes)
    labels_np = np.array(labels)

    # 1. Disegna i nodi GT (Cyan) per riferimento
    if len(gt_nodes_np) > 0:
        plt.scatter(gt_nodes_np[:, 1], gt_nodes_np[:, 0], c='cyan', s=15, label='GT Nodes', zorder=1, alpha=0.6)

    if len(pred_nodes_np) > 0:
        # 2. Disegna le linee di connessione e i nodi predetti
        for i in range(len(pred_nodes_np)):
            pred_node = pred_nodes_np[i]
            gt_idx = matches[i]
            dist = distances[i]
            label = labels_np[i]
            
            # Se c'è un match valido (il KDTree potrebbe ritornare indici strani se vuoto, ma qui controlliamo prima)
            if gt_idx < len(gt_nodes_np):
                gt_node = gt_nodes_np[gt_idx]
                
                if label == 0: # KEEP (Buono)
                    color = 'lime'
                    linestyle = '-' # Linea solida per match forti
                    zorder = 3
                else: # DISCARD (Spazzatura)
                    color = 'red'
                    linestyle = ':' # Linea tratteggiata per match falliti (troppo lontani)
                    zorder = 2
                
                # Disegna il nodo predetto
                plt.scatter(pred_node[1], pred_node[0], c=color, s=15, zorder=zorder)
                
                # Disegna la linea della distanza verso il GT
                plt.plot([pred_node[1], gt_node[1]], [pred_node[0], gt_node[0]], 
                         color=color, linewidth=0.8, linestyle=linestyle, zorder=zorder-1)

    # Legenda custom
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='cyan', marker='o', lw=0),
                    Line2D([0], [0], color='lime', marker='o', lw=1),
                    Line2D([0], [0], color='red', marker='o', linestyle=':', lw=1)]
    plt.legend(custom_lines, ['GT Node', 'Pred (Keep - <20px)', 'Pred (Discard - >20px)'])
    
    plt.title(f"Debug Matching for {name}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()

# -----------------------------------------------

class RoadGraphDataset(Dataset):
    def __init__(self, pred_dir, gt_dir, output_dir, img_dir, node_viz_dir, transform=None):
        super(RoadGraphDataset, self).__init__(None, transform)
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.img_dir = img_dir
        self.node_viz_dir = node_viz_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        if CREATE_VISUALIZATIONS:
            os.makedirs(self.node_viz_dir, exist_ok=True)

        pred_files = set(f for f in os.listdir(pred_dir) if f.endswith('.pickle'))
        gt_files = set(f for f in os.listdir(gt_dir) if f.endswith('.pickle'))
        self.file_names = sorted(list(pred_files.intersection(gt_files)))
        print(f"Found {len(self.file_names)} matching graph pairs.")

    def len(self):
        return len(self.file_names)

    def _get_edge_index(self, adj_matrix):
        if adj_matrix.size == 0:
            return torch.tensor([], dtype=torch.long).reshape(2, 0)
        row, col = np.where((adj_matrix != np.inf) & (np.eye(adj_matrix.shape[0]) == 0))
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
        edge_index = to_undirected(edge_index)
        return edge_index

    def _get_node_labels(self, pred_nodes, gt_nodes):
        """
        Restituisce Labels, Distanze e Indici GT.
        """
        if len(gt_nodes) == 0:
            labels = torch.ones(len(pred_nodes), dtype=torch.long)
            # Se non c'è GT, distanze infinite
            distances = np.full(len(pred_nodes), np.inf)
            gt_indices = np.zeros(len(pred_nodes), dtype=int)
            return labels, distances, gt_indices
            
        if len(pred_nodes) == 0:
            return torch.tensor([], dtype=torch.long), np.array([]), np.array([])
            
        gt_tree = cKDTree(gt_nodes)
        # Query per trovare distanza e indice del vicino GT più prossimo
        distances, gt_indices = gt_tree.query(pred_nodes)
        
        # Label 0 (Keep) se dist < soglia, altrimenti 1 (Discard)
        labels = (distances > NODE_MATCH_THRESHOLD).astype(np.int64)
        return torch.tensor(labels, dtype=torch.long), distances, gt_indices

    def get(self, idx):
        file_name = self.file_names[idx]
        pred_path = os.path.join(self.pred_dir, file_name)
        gt_path = os.path.join(self.gt_dir, file_name)

        try:
            with open(pred_path, 'rb') as f:
                pred_graph = pickle.load(f)
            with open(gt_path, 'rb') as f:
                gt_graph = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return None

        # --- 1. Estrazione Feature (Input X) ---
        pred_features = pred_graph.get('features', [])
        
        if len(pred_features) == 0:
             return None
             
        x_tensor = torch.tensor(pred_features, dtype=torch.float)
        
        # Normalizzazione Coordinate (colonne 0 e 1)
        if x_tensor.size(1) == 5:
            x_tensor[:, 0] /= IMG_SIZE # Y
            x_tensor[:, 1] /= IMG_SIZE # X
        else:
            print(f"ERROR: Feature dimension mismatch in {file_name}. Expected 5.")
            return None
        
        # --- 2. Adiacenza ---
        edge_index = self._get_edge_index(pred_graph.get('adj', np.array([])))

        # --- 3. Etichette Nodi e Debug Info ---
        pred_nodes_raw = pred_graph.get('vertices', []) 
        gt_nodes_raw = gt_graph.get('vertices', [])
        
        y_node_labels, node_distances, gt_indices = self._get_node_labels(pred_nodes_raw, gt_nodes_raw)

        # --- Visualization (Con logica Debug) ---
        if CREATE_VISUALIZATIONS:
            tiff_path = os.path.join(self.img_dir, file_name.replace('.pickle', '.tiff'))
            viz_path = os.path.join(self.node_viz_dir, file_name.replace('.pickle', '.png'))
            
            try:
                original_tiff = Image.open(tiff_path)
                original_rgb_array = np.array(original_tiff.convert('RGB'))
                
                save_debug_visualization(
                    original_rgb_array, 
                    pred_nodes_raw, 
                    gt_nodes_raw, 
                    gt_indices, 
                    node_distances, 
                    y_node_labels,
                    viz_path, 
                    file_name
                )
            except Exception as e:
                print(f"Warning: Viz failed for {file_name}: {e}")

        # --- Creazione Oggetto Data ---
        data = Data(
            x=x_tensor,       # [N, 5]
            edge_index=edge_index,
            y=y_node_labels,  # [N]
            file_name=file_name
        )
        
        save_path = os.path.join(self.output_dir, f'data_{idx}.pt')
        torch.save(data, save_path)
        
        return data

def main():
    print("Starting GNN Cleaner Dataset creation (with Debug Logic)...")
    print(f"Input (X): {PRED_GRAPH_DIR}")
    print(f"Input (Y): {GT_GRAPH_DIR}")
    print(f"Output (Data): {OUTPUT_DATASET_DIR}")
    
    if CREATE_VISUALIZATIONS:
        print(f"Output (Viz): {OUTPUT_NODE_VIZ_DIR}")
        print("Visualizations ENABLED.")
    else:
        print("Visualizations DISABLED.")
    
    dataset = RoadGraphDataset(
        pred_dir=PRED_GRAPH_DIR,
        gt_dir=GT_GRAPH_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        img_dir=IMAGE_DIR, 
        node_viz_dir=OUTPUT_NODE_VIZ_DIR
    )
    
    print("\nProcessing and saving graph pairs...")
    for i in tqdm(range(len(dataset))):
        dataset.get(i) 
        
    print(f"\nProcessing complete.")
    print(f"Dataset saved to: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    main()