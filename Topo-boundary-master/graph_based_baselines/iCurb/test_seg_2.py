import os
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

# --- Imports for Cleaning and Repair ---
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
# ----------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import your model definition
from models.models_encoder import FPN 

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./checkpoints/seg_pretrain_manhattan_efficentnet_1.6_v2.pth"
DATA_SPLIT_JSON = "./dataset_manhattan/data_split.json"
IMAGE_DIR = "./dataset_manhattan/cropped_tiff" 

OUTPUT_SKELETON_DIR = "./records/gt/pred_skeleton_2"
OUTPUT_GRAPH_DIR = "./records/gt/pred_graph_2"
OUTPUT_VISUALIZATION_DIR = "./records/gt/pred_visualization_2" 

THRESHOLD = 0.2
CLEANING_MIN_SIZE = 25 
REPAIR_MAX_DISTANCE = 20 
SAMPLING_DISTANCE = 30 
# ----------------------

#
# --- Vertex, Graph (Immutate) ---
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
        if previous in neighbors:
            neighbors.remove(previous)
        if not neighbors: 
            return None
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
                v.key_vertex = True
                self.key_vertices.append(v)
                self.sampled_vertices.append(v) 


#
# --- generate_graph  (5 FEATURES) ---
#
def generate_graph(skeleton, pred_mask, file_name, graph_dir):
    def find_neighbors(v,img,remove=False):
        output_v = []
        H, W = img.shape
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
    
    if np.sum(img) == 0:
        graph_data = {'vertices':[], 'adj':np.array([]), 'features':[]}
        pickle_name = file_name[:-3] + 'pickle'
        with open(os.path.join(graph_dir, pickle_name),'wb') as jf:
            pickle.dump(graph_data, jf)
        return graph_data
        
    pre_points = np.where(img!=0)
    pre_points = [[pre_points[0][i],pre_points[1][i]] for i in range(len(pre_points[0]))]
    
    for point in pre_points:
        v = Vertex(point); graph.add_v(v,find_neighbors(point,img))
    
    graph.find_key_vertices() 
    
    # --- LOGICA DI CAMPIONAMENTO ---
    for key_vertex in graph.key_vertices:
        for neighbor in key_vertex.unprocessed_neighbors.copy():
            if neighbor not in key_vertex.unprocessed_neighbors:
                continue
            key_vertex.unprocessed_neighbors.remove(neighbor)
            
            curr_v = neighbor
            pre_v = key_vertex
            sampled_v = key_vertex 
            counter = 1
            
            while(not curr_v.key_vertex):
                if counter % SAMPLING_DISTANCE == 0:
                    sampled_v.sampled_neighbors.append(curr_v)
                    curr_v.sampled_neighbors.append(sampled_v)
                    sampled_v = curr_v
                    if not sampled_v.key_vertex: 
                        graph.sampled_vertices.append(sampled_v)
                
                next_v = curr_v.next(pre_v)
                if next_v is None: break
                
                if pre_v in curr_v.unprocessed_neighbors:
                    curr_v.unprocessed_neighbors.remove(pre_v)
                
                pre_v = curr_v; curr_v = next_v
                counter += 1
            
            sampled_v.sampled_neighbors.append(curr_v)
            curr_v.sampled_neighbors.append(sampled_v)
            
            if pre_v in curr_v.unprocessed_neighbors:
                 curr_v.unprocessed_neighbors.remove(pre_v)
    # --- FINE CAMPIONAMENTO ---

    # --- CALCOLO FEATURE (5 FEATURES) ---
    vertices = []
    features = [] 
    
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        # Salviamo le coordinate anche in vertices per compatibilità con vecchi script di visualizzazione
        vertices.append([int(v.coord[0]), int(v.coord[1])])
        
        # 1. Coordinate Y (Feature spaziale)
        coord_y = float(v.coord[0])
        # 2. Coordinate X (Feature spaziale)
        coord_x = float(v.coord[1])
        
        # 3. Grado (Feature topologica)
        degree = float(v.pixel_degree)
        
        # 4. Angolo (Feature geometrica/direzionale)
        angle = 0.0
        num_sampled_neighbors = len(v.sampled_neighbors)
        if num_sampled_neighbors == 1:
            n1 = v.sampled_neighbors[0]
            dy = n1.coord[0] - v.coord[0]
            dx = n1.coord[1] - v.coord[1]
            angle = np.arctan2(dy, dx)
        elif num_sampled_neighbors == 2:
            n1 = v.sampled_neighbors[0]
            n2 = v.sampled_neighbors[1]
            dy = n2.coord[0] - n1.coord[0]
            dx = n2.coord[1] - n1.coord[1]
            angle = np.arctan2(dy, dx)
        
        # 5. Intensity (Feature semantica dalla Heatmap)
        intensity = float(pred_mask[int(v.coord[0]), int(v.coord[1])])
        
        # VETTORE FINALE A 5 DIMENSIONI
        features.append([coord_y, coord_x, degree, angle, intensity])

    if not graph.sampled_vertices:
        graph_data = {'vertices':[], 'adj':np.array([]), 'features':[]}
        pickle_name = file_name[:-3] + 'pickle'
        with open(os.path.join(graph_dir, pickle_name),'wb') as jf:
            pickle.dump(graph_data, jf)
        return graph_data
        
    # --- Costruisci ADJ ---
    adjacent = np.ones((len(graph.sampled_vertices),len(graph.sampled_vertices))) * np.inf
    for v in graph.sampled_vertices:
        for u in v.sampled_neighbors: 
            if u in graph.sampled_vertices: 
                dist = v.distance(u)
                adjacent[v.index,u.index] = dist
                adjacent[u.index,v.index] = dist
    
    graph_data = {'vertices':vertices, 'adj':adjacent, 'features': features}
    
    pickle_name = file_name[:-3] + 'pickle' 
    with open(os.path.join(graph_dir, pickle_name),'wb') as jf:
        pickle.dump(graph_data, jf)
        
    return graph_data
# --- FINE generate_graph ---


def repair_graph(graph_data, max_distance):
    vertices = graph_data.get('vertices', [])
    adj_matrix = graph_data.get('adj', np.array([]))
    features = graph_data.get('features', []) 
    
    if not vertices or adj_matrix.size == 0: 
        return graph_data 
        
    num_nodes = len(vertices)
    endpoints = []
    
    for i in range(num_nodes):
        neighbors = np.sum(adj_matrix[i, :] != np.inf) 
        if neighbors <= 1: 
             endpoints.append(i)
        
    if len(endpoints) < 2: 
        return graph_data 

    endpoint_coords = np.array([vertices[i] for i in endpoints])
    dist_matrix = squareform(pdist(endpoint_coords))
    potential_pairs_indices = np.where((dist_matrix > 0) & (dist_matrix < max_distance))
    
    repaired_adj = adj_matrix.copy() 
    num_connections = 0
    
    for idx_i, idx_j in zip(*potential_pairs_indices):
        node_i = endpoints[idx_i]; node_j = endpoints[idx_j]
        if repaired_adj[node_i, node_j] != np.inf: continue
        
        dist_path, _ = dijkstra(csgraph=repaired_adj, directed=False, indices=node_i, return_predecessors=True)
        if dist_path[node_j] == np.inf:
            dist_euclidean = dist_matrix[idx_i, idx_j]
            repaired_adj[node_i, node_j] = dist_euclidean
            repaired_adj[node_j, node_i] = dist_euclidean
            num_connections += 1
            
    if num_connections > 0: tqdm.write(f"  -> Graph repair: Added {num_connections} connections.")
    
    return {'vertices': vertices, 'adj': repaired_adj, 'features': features} 

def save_visualization(original_rgb_img, skeleton_img_array, graph_data, output_path, name):
    vertices = graph_data.get('vertices', [])
    adj_matrix = graph_data.get('adj', np.array([]))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_rgb_img) 
    
    if vertices:
        vertices_np = np.array(vertices)
        plt.scatter(vertices_np[:, 1], vertices_np[:, 0], c='red', s=10, zorder=5, label='Nodes') 
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if adj_matrix[i, j] != np.inf:
                    v1 = vertices[i]
                    v2 = vertices[j]
                    plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'cyan', linewidth=1.5) 
    
    plt.title(f"Graph on RGB for {name}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

#
# --- Funzione Main ---
#
def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {CHECKPOINT_PATH}")
    net = FPN(n_channels=4, n_classes=1) 
    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(CHECKPOINT_PATH))
    else:
        net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu')))
    net.to(DEVICE)
    net.eval() 
    print("Model ready.")

    print(f"Reading file list from: {DATA_SPLIT_JSON}")
    try:
        with open(DATA_SPLIT_JSON, 'r') as f:
            json_data = json.load(f)
            test_list = json_data['train'] 
    except FileNotFoundError:
        print(f"ERROR: Data split JSON not found at {DATA_SPLIT_JSON}")
        return
        
    print(f"Found {len(test_list)} images.")
    os.makedirs(OUTPUT_SKELETON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True) 

    print(f"Starting pipeline (Sampling {SAMPLING_DISTANCE}px) -> Generating 5 Features...")
    
    with torch.no_grad(): 
        for name in tqdm(test_list, desc="Processing Predicted Graphs"):
            try:
                img_path = os.path.join(IMAGE_DIR, f"{name}.tiff")
                img_full_tiff = Image.open(img_path) 
                
                original_rgb_array = np.array(img_full_tiff.convert('RGB'))
                
                tiff_for_model = tvf.to_tensor(img_full_tiff).to(DEVICE)
                tiff_batch = tiff_for_model.unsqueeze(0) 
                
                predictions, _ = net(tiff_batch)
                pred_mask = torch.sigmoid(predictions)
                pred_mask_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy() 
                
                binary_mask_dirty = (pred_mask_np > THRESHOLD)
                cleaned_mask = remove_small_objects(binary_mask_dirty, min_size=CLEANING_MIN_SIZE)
                filled_mask = binary_fill_holes(cleaned_mask)
                
                skeleton = morphology.skeletonize(filled_mask, method='lee')
                skeleton_salvabile = util.img_as_ubyte(skeleton)
                
                png_name = f"{name}.png"
                output_path_png = os.path.join(OUTPUT_SKELETON_DIR, png_name)
                io.imsave(output_path_png, skeleton_salvabile)
                
                # Passiamo pred_mask_np (heatmap)
                graph_data_raw = generate_graph(skeleton_salvabile, pred_mask_np, png_name, OUTPUT_GRAPH_DIR)
                
                graph_data_repaired = repair_graph(graph_data_raw, max_distance=REPAIR_MAX_DISTANCE)
                
                output_path_viz = os.path.join(OUTPUT_VISUALIZATION_DIR, png_name)
                save_visualization(original_rgb_array, skeleton_salvabile, graph_data_repaired, output_path_viz, name)
                
                vertices = graph_data_repaired.get('vertices', [])
                adj_matrix = graph_data_repaired.get('adj', np.array([]))
                total_vertices = len(vertices)
                num_endpoints = 0
                if total_vertices > 0:
                    for i in range(total_vertices):
                        neighbors = np.sum(adj_matrix[i, :] != np.inf) 
                        if neighbors <= 1: num_endpoints += 1
                            
                tqdm.write(f"  -> Img: {name} | Vertici: {total_vertices} | Endpoint: {num_endpoints}")
                
            except FileNotFoundError:
                print(f"Warning: Image not found, skipped: {img_path}")
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\nProcessing complete.")
    print(f"Final graphs with 5 FEATURES saved to: {OUTPUT_GRAPH_DIR}")

if __name__ == "__main__":
    main()