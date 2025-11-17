import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

# --- Imports for Visualization ---
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
# -------------------------------


# --- Configuration (UPDATED) ---
PRED_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/pred_graph_2"
GT_GRAPH_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/records/gt/gt_graphs_2"
IMAGE_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"

OUTPUT_DATASET_DIR = "./gnn_dataset_20_80"
OUTPUT_NODE_VIZ_DIR = "./gnn_dataset_node_viz_20_80" # Viz 1
OUTPUT_LINK_VIZ_DIR = "./gnn_dataset_link_viz_20_80" # Viz 2
OUTPUT_DEBUG_VIZ_DIR = "./gnn_dataset_debug_viz_20_80" # Viz 3

# --- FLAG DI VISUALIZZAZIONE  ---
CREATE_VISUALIZATIONS = False 
# --------------------------------

NODE_MATCH_THRESHOLD = 20.0 
LINK_CANDIDATE_DISTANCE = 80.0 
IMG_SIZE = 1000.0
PARALLEL_ANGLE_THRESHOLD = np.pi / 4 
# ---------------------

#
#
def save_node_label_visualization(rgb_img, nodes, edges, labels, output_path, name):
    plt.figure(figsize=(10, 10)); plt.imshow(rgb_img, zorder=0)
    nodes_np = np.array(nodes); labels_np = np.array(labels)
    good_nodes = nodes_np[labels_np == 0]; garbage_nodes = nodes_np[labels_np == 1]
    if len(good_nodes) > 0: plt.scatter(good_nodes[:, 1], good_nodes[:, 0], c='lime', s=10, label='Good Nodes (Label 0)', zorder=2)
    if len(garbage_nodes) > 0: plt.scatter(garbage_nodes[:, 1], garbage_nodes[:, 0], c='magenta', s=10, label='Garbage Nodes (Label 1)', zorder=3)
    if len(nodes_np) > 0 and edges.numel() > 0:
        for (src, dst) in edges.t().tolist():
            if src < len(nodes_np) and dst < len(nodes_np):
                n1, n2 = nodes_np[src], nodes_np[dst]; plt.plot([n1[1], n2[1]], [n1[0], n2[0]], color='cyan', linewidth=1.0, alpha=0.5, zorder=1)
    plt.title(f"Node Classification Labels for {name}"); plt.legend(); plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200); plt.close()

def save_link_label_visualization(rgb_img, nodes, edges, node_labels, candidate_pairs, link_labels, output_path, name):
    plt.figure(figsize=(10, 10)); plt.imshow(rgb_img, zorder=0)
    nodes_np = np.array(nodes); node_labels_np = np.array(node_labels)
    good_nodes = nodes_np[node_labels_np == 0]; garbage_nodes = nodes_np[node_labels_np == 1]
    if len(good_nodes) > 0: plt.scatter(good_nodes[:, 1], good_nodes[:, 0], c='lime', s=10, label='Nodi Buoni (Label 0)', zorder=2)
    if len(garbage_nodes) > 0: plt.scatter(garbage_nodes[:, 1], garbage_nodes[:, 0], c='magenta', s=10, label='Nodi Spazzatura (Label 1)', zorder=3)
    if len(nodes_np) > 0 and edges.numel() > 0:
        for (src, dst) in edges.t().tolist():
            if src < len(nodes_np) and dst < len(nodes_np):
                if node_labels_np[src] == 0 and node_labels_np[dst] == 0:
                    n1, n2 = nodes_np[src], nodes_np[dst]; plt.plot([n1[1], n2[1]], [n1[0], n2[0]], color='blue', linewidth=1.0, zorder=1, label='Arco Esistente (Buono)')
    if len(link_labels) > 0 and len(candidate_pairs) > 0:
        link_labels_np = np.array(link_labels); candidate_pairs_np = np.array(candidate_pairs)
        connect_links_label_1 = candidate_pairs_np[link_labels_np == 1]
        no_connect_links_label_0 = candidate_pairs_np[link_labels_np == 0]
    else:
        connect_links_label_1 = []; no_connect_links_label_0 = []
    for (src, dst) in connect_links_label_1:
        n1, n2 = nodes_np[src], nodes_np[dst]; plt.plot([n1[1], n2[1]], [n1[0], n2[0]], color='lime', linewidth=1.5, linestyle='--', zorder=1, label='Connetti (Label 1)')
    for (src, dst) in no_connect_links_label_0:
        n1, n2 = nodes_np[src], nodes_np[dst]; plt.plot([n1[1], n2[1]], [n1[0], n2[0]], color='red', linewidth=1.5, linestyle=':', zorder=1, label='Non Connettere (Label 0)')
    plt.title(f"Link Prediction Labels for {name}"); handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)); plt.legend(by_label.values(), by_label.keys()); plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200); plt.close()

def save_debug_matching_visualization(rgb_img, pred_nodes, pred_features, gt_nodes, matches, distances, threshold, output_path, name):
    plt.figure(figsize=(10, 10)); plt.imshow(rgb_img, zorder=0)
    if len(gt_nodes) > 0:
        gt_nodes_np = np.array(gt_nodes)
        plt.scatter(gt_nodes_np[:, 1], gt_nodes_np[:, 0], c='cyan', s=15, label='GT Nodes (Y)', zorder=1, alpha=0.5)
    if len(pred_nodes) == 0:
        plt.title(f"Debug Matching for {name} (No Pred Nodes)"); plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200); plt.close()
        return
    for i in range(len(pred_nodes)):
        pred_node = pred_nodes[i]; gt_node_idx = matches[i]
        gt_node = gt_nodes[gt_node_idx]; dist = distances[i]
        degree = pred_features[i][0]; angle_rad = pred_features[i][1]; angle_deg = np.degrees(angle_rad)
        if dist <= threshold:
            color = 'lime'; label = 'Match (Good)'
        else:
            color = 'red'; label = 'No Match (Garbage)'
        plt.scatter(pred_node[1], pred_node[0], c=color, s=20, zorder=3, label=label)
        plt.plot([pred_node[1], gt_node[1]], [pred_node[0], gt_node[0]], color=color, linewidth=0.8, linestyle=':', zorder=2)
    plt.title(f"Debug Nearest Neighbor Matching for {name}\nThreshold = {threshold}px")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)); plt.legend(by_label.values(), by_label.keys()); plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200); plt.close()

# -----------------------------------------------

class RoadGraphDataset(Dataset):
    def __init__(self, pred_dir, gt_dir, output_dir, img_dir, node_viz_dir, link_viz_dir, debug_viz_dir, transform=None):
        super(RoadGraphDataset, self).__init__(None, transform)
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.img_dir = img_dir
        self.node_viz_dir = node_viz_dir
        self.link_viz_dir = link_viz_dir
        self.debug_viz_dir = debug_viz_dir 
        os.makedirs(self.output_dir, exist_ok=True)
        # --- MODIFICA: Controlla la flag ---
        if CREATE_VISUALIZATIONS:
            os.makedirs(self.node_viz_dir, exist_ok=True)
            os.makedirs(self.link_viz_dir, exist_ok=True)
            os.makedirs(self.debug_viz_dir, exist_ok=True)
        # ---------------------------------
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
        if len(gt_nodes) == 0:
            labels = torch.ones(len(pred_nodes), dtype=torch.long)
            distances = np.full(len(pred_nodes), np.inf)
            gt_indices = np.zeros(len(pred_nodes), dtype=int)
            return labels, distances, gt_indices
        if len(pred_nodes) == 0:
            return torch.tensor([], dtype=torch.long), np.array([]), np.array([])
        gt_tree = cKDTree(gt_nodes)
        distances, gt_indices = gt_tree.query(pred_nodes)
        labels = (distances > NODE_MATCH_THRESHOLD).astype(np.int64)
        return torch.tensor(labels, dtype=torch.long), distances, gt_indices

    def _get_link_labels(self, pred_graph, gt_graph):
        pred_nodes = pred_graph.get('vertices', [])
        pred_adj = pred_graph.get('adj', np.array([]))
        pred_features = pred_graph.get('features', []) 
        gt_nodes = gt_graph.get('vertices', [])
        gt_adj = gt_graph.get('adj', np.array([]))
        
        if len(pred_nodes) == 0 or len(gt_nodes) == 0 or len(pred_features) == 0:
            return torch.tensor([], dtype=torch.long).reshape(2, 0), torch.tensor([], dtype=torch.long)

        endpoints_g1 = []
        nodes_g0 = []
        for i in range(len(pred_nodes)):
            degree = pred_features[i][0] 
            if degree == 1:
                endpoints_g1.append(i)
            elif degree == 0:
                nodes_g0.append(i)
        
        all_loose_nodes = endpoints_g1 + nodes_g0 
        
        if len(all_loose_nodes) < 2:
            return torch.tensor([], dtype=torch.long).reshape(2, 0), torch.tensor([], dtype=torch.long)
            
        loose_coords = np.array([pred_nodes[i] for i in all_loose_nodes])
        dist_matrix = squareform(pdist(loose_coords))
        dist_matrix[dist_matrix == 0] = np.inf 
        
        gt_tree = cKDTree(gt_nodes)
        candidate_pairs_set = set()
        
        map_g1_to_all = {node_idx: i for i, node_idx in enumerate(all_loose_nodes) if node_idx in endpoints_g1}
        for node_i_idx in endpoints_g1:
            i_global = map_g1_to_all[node_i_idx]
            j_global = np.argmin(dist_matrix[i_global, :])
            dist = dist_matrix[i_global, j_global]
            if dist < LINK_CANDIDATE_DISTANCE:
                node_j_idx = all_loose_nodes[j_global]
                pair = tuple(sorted((node_i_idx, node_j_idx)))
                candidate_pairs_set.add(pair)

        map_g0_to_all = {node_idx: i for i, node_idx in enumerate(all_loose_nodes) if node_idx in nodes_g0}
        for node_i_idx in nodes_g0:
            i_global = map_g0_to_all[node_i_idx]
            all_dists = dist_matrix[i_global, :]
            k_nearest_indices = np.argpartition(all_dists, 2)[:2]
            for j_global in k_nearest_indices:
                dist = all_dists[j_global]
                if dist < LINK_CANDIDATE_DISTANCE:
                    node_j_idx = all_loose_nodes[j_global]
                    pair = tuple(sorted((node_i_idx, node_j_idx)))
                    candidate_pairs_set.add(pair)

        if not candidate_pairs_set:
            return torch.tensor([], dtype=torch.long).reshape(2, 0), torch.tensor([], dtype=torch.long)
            
        edge_labels_dict = {}
        candidate_pairs_global_idx = list(candidate_pairs_set)

        for pair in candidate_pairs_global_idx: 
            node_i_idx, node_j_idx = pair
            
            if pred_adj[node_i_idx, node_j_idx] != np.inf:
                edge_labels_dict[pair] = 0 
                continue
            
            feat_i = pred_features[node_i_idx]; feat_j = pred_features[node_j_idx]
            angle_i = feat_i[1]; angle_j = feat_j[1]
            
            is_parallel = False
            if feat_i[0] == 1 and feat_j[0] == 1: 
                delta_angle = abs(angle_i - angle_j)
                if delta_angle > np.pi: delta_angle = 2 * np.pi - delta_angle
                if delta_angle < PARALLEL_ANGLE_THRESHOLD:
                    is_parallel = True
            
            dist_i, gt_i_idx = gt_tree.query(pred_nodes[node_i_idx])
            dist_j, gt_j_idx = gt_tree.query(pred_nodes[node_j_idx])
            label = 0 
            
            if (not is_parallel and 
                dist_i < NODE_MATCH_THRESHOLD and 
                dist_j < NODE_MATCH_THRESHOLD and 
                gt_i_idx != gt_j_idx):
                
                if gt_adj.shape[0] > gt_i_idx and gt_adj.shape[0] > gt_j_idx:
                    path_dist, _ = dijkstra(csgraph=gt_adj, directed=False, indices=gt_i_idx, return_predecessors=True)
                    if path_dist[gt_j_idx] != np.inf:
                        label = 1 
            
            edge_labels_dict[pair] = label

        edge_labels = [edge_labels_dict[pair] for pair in candidate_pairs_global_idx]
        edge_label_index = torch.tensor(candidate_pairs_global_idx, dtype=torch.long).t().contiguous()
        edge_label = torch.tensor(edge_labels, dtype=torch.long)
        
        return edge_label_index, edge_label


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

        pred_nodes = pred_graph.get('vertices', [])
        pred_features = pred_graph.get('features', []) 
        
        if len(pred_nodes) == 0 or (len(pred_features) != len(pred_nodes)):
            print(f"Warning: Skipping {file_name}, feature/node mismatch or empty graph.")
            return None 

        x_features = torch.cat([
            torch.tensor(pred_nodes, dtype=torch.float) / IMG_SIZE, 
            torch.tensor(pred_features, dtype=torch.float)         
        ], dim=1) 
        
        edge_index = self._get_edge_index(pred_graph.get('adj', np.array([])))

        gt_nodes = gt_graph.get('vertices', [])
        y_node_labels, node_distances, node_gt_indices = self._get_node_labels(pred_nodes, gt_nodes)
        y_link_label_index, y_link_labels = self._get_link_labels(pred_graph, gt_graph)

        # --- MODIFICA: Controlla la flag ---
        if CREATE_VISUALIZATIONS:
            tiff_path = os.path.join(self.img_dir, file_name.replace('.pickle', '.tiff'))
            node_viz_path = os.path.join(self.node_viz_dir, file_name.replace('.pickle', '.png'))
            link_viz_path = os.path.join(self.link_viz_dir, file_name.replace('.pickle', '.png'))
            debug_viz_path = os.path.join(self.debug_viz_dir, file_name.replace('.pickle', '.png'))
            
            try:
                original_tiff = Image.open(tiff_path)
                original_rgb_array = np.array(original_tiff.convert('RGB'))
                
                save_node_label_visualization(
                    original_rgb_array, pred_nodes, edge_index, y_node_labels,
                    node_viz_path, file_name
                )
                
                save_link_label_visualization(
                    original_rgb_array, pred_nodes, edge_index, 
                    y_node_labels,
                    y_link_label_index.t().tolist(), y_link_labels.tolist(),
                    link_viz_path, file_name
                )
                
                save_debug_matching_visualization(
                    original_rgb_array, pred_nodes, pred_features, gt_nodes,
                    node_gt_indices, node_distances, NODE_MATCH_THRESHOLD,
                    debug_viz_path, file_name
                )
                
            except FileNotFoundError:
                 print(f"Warning: Could not find TIFF {tiff_path} for visualization.")
            except Exception as e:
                print(f"Warning: Failed to create visualization for {file_name}: {e}")
        # --- FINE MODIFICA ---

        data = Data(
            x=x_features,
            edge_index=edge_index,
            y=y_node_labels,
            edge_label_index=y_link_label_index,
            edge_label=y_link_labels,
            file_name=file_name
        )
        
        save_path = os.path.join(self.output_dir, f'data_{idx}.pt')
        torch.save(data, save_path)
        
        return data

def main():
    print("Starting GNN Dataset creation...")
    print(f"Input (X): {PRED_GRAPH_DIR}")
    print(f"Input (Y): {GT_GRAPH_DIR}")
    print(f"Input (RGB): {IMAGE_DIR}")
    print(f"Output (Data): {OUTPUT_DATASET_DIR}")
    
    if CREATE_VISUALIZATIONS:
        print(f"Output (Node Viz): {OUTPUT_NODE_VIZ_DIR}")
        print(f"Output (Link Viz): {OUTPUT_LINK_VIZ_DIR}")
        print(f"Output (Debug Viz): {OUTPUT_DEBUG_VIZ_DIR}")
    else:
        print("Visualizations are DISABLED for speed.")
    
    dataset = RoadGraphDataset(
        pred_dir=PRED_GRAPH_DIR,
        gt_dir=GT_GRAPH_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        img_dir=IMAGE_DIR, 
        node_viz_dir=OUTPUT_NODE_VIZ_DIR,
        link_viz_dir=OUTPUT_LINK_VIZ_DIR,
        debug_viz_dir=OUTPUT_DEBUG_VIZ_DIR
    )
    
    print("\nProcessing and saving graph pairs...")
    for i in tqdm(range(len(dataset))):
        dataset.get(i) 
        
    print(f"\nProcessing complete.")
    print(f"Processed dataset saved to: {OUTPUT_DATASET_DIR}")

if __name__ == "__main__":
    main()