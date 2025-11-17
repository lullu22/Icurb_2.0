import os
import json
import numpy as np
import skimage.io as io
from skimage import morphology, util
from tqdm import tqdm # Importa tqdm per tqdm.write
import pickle
from PIL import Image 

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- Import per la logica del grafo ---
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
# ----------------------------------------------------

# --- Configuration ---
DATA_SPLIT_JSON = "./dataset_manhattan/data_split.json"
GT_SKELETON_DIR = "./dataset_manhattan/labels/binary_map" 
IMAGE_DIR = "./dataset_manhattan/cropped_tiff" 
OUTPUT_GT_GRAPH_DIR = "./records/gt/gt_graphs_2"        
OUTPUT_GT_VIZ_DIR = "./records/gt/gt_visualizations_2"  
SAMPLING_DISTANCE = 30 
# ----------------------

#
# --- Vertex, Graph (COPIATE DA test_seg.py) ---
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
        if not neighbors: # Se è un vicolo cieco
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
            # Salva il grado del pixel originale per dopo
            v.pixel_degree = len(v.neighbors) 
            if v.pixel_degree != 2:
                v.key_vertex = True; self.key_vertices.append(v); self.sampled_vertices.append(v)

#
# --- generate_graph (COPIATO DA test_seg.py) ---
#
def generate_graph(skeleton, file_name, graph_dir):
    def find_neighbors(v,img,remove=False):
        output_v = []
        # Definisci i limiti dell'immagine
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
    
    for key_vertex in graph.key_vertices:
        if len(key_vertex.unprocessed_neighbors):
            for neighbor in key_vertex.unprocessed_neighbors.copy(): 
                if neighbor not in key_vertex.unprocessed_neighbors:
                    continue
                key_vertex.unprocessed_neighbors.remove(neighbor)
                
                curr_v = neighbor; pre_v = key_vertex; sampled_v = key_vertex; counter = 1
                while(not curr_v.key_vertex):
                    if counter % SAMPLING_DISTANCE == 0:
                        sampled_v.sampled_neighbors.append(curr_v)
                        curr_v.sampled_neighbors.append(sampled_v)
                        sampled_v = curr_v
                        if not sampled_v.key_vertex: 
                            graph.sampled_vertices.append(sampled_v)
                    
                    next_v = curr_v.next(pre_v)
                    if next_v is None: break 
                    pre_v = curr_v; curr_v = next_v; counter += 1
                
                sampled_v.sampled_neighbors.append(curr_v); curr_v.sampled_neighbors.append(sampled_v)
                if pre_v in curr_v.unprocessed_neighbors:
                     curr_v.unprocessed_neighbors.remove(pre_v)

    # --- CALCOLO FEATURE ---
    vertices = []
    features = [] 
    
    for ii, v in enumerate(graph.sampled_vertices):
        v.index = ii
        vertices.append([int(v.coord[0]), int(v.coord[1])])
        
        degree = v.pixel_degree 
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
        
        features.append([degree, angle])
    # --- FINE CALCOLO FEATURE ---

    if not graph.sampled_vertices:
        graph_data = {'vertices':[], 'adj':np.array([]), 'features':[]}
        pickle_name = file_name[:-3] + 'pickle'
        with open(os.path.join(graph_dir, pickle_name),'wb') as jf:
            pickle.dump(graph_data, jf)
        return graph_data
        
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


#
# --- save_visualization (con colori GT) ---
#
def save_visualization(original_rgb_img, skeleton_img_array, graph_data, output_path, name):
    vertices = graph_data.get('vertices', []) 
    adj_matrix = graph_data.get('adj', np.array([]))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_rgb_img, zorder=0) 

    if vertices:
        vertices_np = np.array(vertices)
        
        plt.scatter(vertices_np[:, 1], vertices_np[:, 0], c='magenta', s=10, zorder=2, label='Nodes')
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if adj_matrix[i, j] != np.inf:
                    v1 = vertices[i] 
                    v2 = vertices[j] 
                    plt.plot([v1[1], v2[1]], [v1[0], v2[0]], color='lime', linewidth=1.5, zorder=1)
    
    plt.title(f"GT Graph on RGB for {name}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()

#
# --- Funzione Main (aggiornata con Stampa) ---
#
def main():
    print(f"Starting Ground Truth Graph generation (using SAMPLING)...")
    print(f"Graph output directory: {OUTPUT_GT_GRAPH_DIR}")
    print(f"Visualization output directory: {OUTPUT_GT_VIZ_DIR}")

    try:
        with open(DATA_SPLIT_JSON, 'r') as f:
            file_list = json.load(f)['train'] 
    except FileNotFoundError:
        print(f"ERROR: Data split JSON not found at {DATA_SPLIT_JSON}")
        return
        
    os.makedirs(OUTPUT_GT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GT_VIZ_DIR, exist_ok=True) 
    
    for name in tqdm(file_list, desc="Generating GT Graphs"):
        try:
            png_name = f"{name}.png"
            skeleton_path = os.path.join(GT_SKELETON_DIR, png_name)
            tiff_path = os.path.join(IMAGE_DIR, f"{name}.tiff") 
            
            gt_skeleton_img = io.imread(skeleton_path, as_gray=True)
            original_tiff = Image.open(tiff_path)
            original_rgb_array = np.array(original_tiff.convert('RGB'))
            
            # (Non serve pulizia qui, assumiamo che il GT sia pulito)
            
            graph_data_gt = generate_graph(gt_skeleton_img, png_name, OUTPUT_GT_GRAPH_DIR)
            
            output_path_viz = os.path.join(OUTPUT_GT_VIZ_DIR, png_name)
            save_visualization(original_rgb_array, gt_skeleton_img, graph_data_gt, output_path_viz, name)
            
            # --- NUOVO: Stampa Statistiche Grafo ---
            vertices = graph_data_gt.get('vertices', [])
            adj_matrix = graph_data_gt.get('adj', np.array([]))
            
            total_vertices = len(vertices)
            num_endpoints = 0
            if total_vertices > 0:
                for i in range(total_vertices):
                    # Calcola il grado dalla matrice di adiacenza finale
                    neighbors = np.sum(adj_matrix[i, :] != np.inf) 
                    if neighbors <= 1: # 0 (isolato) o 1 (endpoint)
                        num_endpoints += 1
                        
            tqdm.write(f"  -> Img: {name} | Vertici Totali: {total_vertices} | Endpoint: {num_endpoints}")
            # --- FINE NUOVA SEZIONE ---
            
        except FileNotFoundError as e:
            print(f"Warning: File not found, skipped: {e.filename}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nGround Truth graphs (Y) generation complete.")
    print(f"Graphs saved to: {OUTPUT_GT_GRAPH_DIR}")
    print(f"Visualizations saved to: {OUTPUT_GT_VIZ_DIR}")

if __name__ == '__main__':
    main()