import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
import matplotlib.cm as cm

# =================CONFIGURATION=================
BASE_PROJECT_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/"
GT_GRAPH_DIR = os.path.join(BASE_PROJECT_DIR, "records", "gt", "gt_graphs_2_RDP")
HEATMAP_DIR = os.path.join(BASE_PROJECT_DIR, "RL", "train", "heatmaps") 

TARGET_IMAGE_LIST = ["005250_04", "002247_02","000227_10","000250_13","002242_22", "980200_40"]  # Change this to visualize a different image

output_folder = "debug_paths_grids"
os.makedirs(output_folder, exist_ok=True)
# ===============================================

def _reconstruct_path_indices(preds, start, end):
    path = []
    curr = end
    max_iter = len(preds) * 2
    count = 0
    while curr != -9999 and curr != start and count < max_iter:
        path.append(curr)
        curr = preds[curr]
        if curr < 0: break
        count += 1
    path.append(start)
    return path[::-1]

def visualize_all_paths_grid(image_name):

    output_folder = "debug_paths_grids"
    os.makedirs(output_folder, exist_ok=True)
    OUTPUT_IMAGE_PATH = os.path.join(output_folder, f"{image_name}_all_paths_grid.png")
    print(f"--- Generazione Griglia Percorsi per: {image_name} ---")
    

    heatmap_path = os.path.join(HEATMAP_DIR, f"{image_name}.npy")
    graph_path = os.path.join(GT_GRAPH_DIR, f"{image_name}.pickle")

    if not os.path.exists(heatmap_path) or not os.path.exists(graph_path):
        print(f"Errore: File non trovati.")
        return

    heatmap = np.load(heatmap_path)
    with open(graph_path, 'rb') as f:
        gt_data = pickle.load(f)

    gt_vertices = np.array(gt_data['vertices'])
    gt_adj = gt_data['adj']
    H, W = heatmap.shape
    margin = 20
    search_radius = 5
    # Identificazione Border Indices
    border_indices = []
    for i, (y, x) in enumerate(gt_vertices):
        if (y < margin or y > H - margin or x < margin or x > W - margin):
            yc, xc = int(y), int(x)
            yc_min = max(0, yc - search_radius)
            yc_max = min(H, yc + search_radius + 1)
            xc_min = max(0, xc - search_radius)
            xc_max = min(W, xc + search_radius + 1)
            local_patch = heatmap[yc_min:yc_max, xc_min:xc_max]
            if np.max(local_patch) > 0.2:
                border_indices.append(i)

    # Trova percorsi
    all_possible_paths = []
    processed_pairs = set()

    print("Calcolo percorsi...")
    for start_idx in border_indices:
        dist_matrix, predecessors = dijkstra(gt_adj, directed=False, indices=start_idx, return_predecessors=True)

        for end_idx in border_indices:
            if end_idx == start_idx: continue
            
            # NOTA: Rimuoviamo il controllo 'processed_pairs' qui per vedere TUTTO, 
            # anche i duplicati inversi o simili, così capiamo esattamente cosa vede l'agente.
            # pair_key = tuple(sorted((start_idx, end_idx)))
            # if pair_key in processed_pairs: continue

            if dist_matrix[end_idx] == np.inf: continue
            if np.linalg.norm(gt_vertices[end_idx] - gt_vertices[start_idx]) <= 300: continue

            path_indices = _reconstruct_path_indices(predecessors, start_idx, end_idx)
            path_pixels = gt_vertices[path_indices]

            # Aggiungiamo alla lista
            all_possible_paths.append({
                'path': path_pixels,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            # processed_pairs.add(pair_key)

    num_paths = len(all_possible_paths)
    print(f"Totale percorsi trovati: {num_paths}")

    if num_paths == 0:
        print("Nessun percorso trovato.")
        return

    # --- CONFIGURAZIONE GRIGLIA (SUBPLOTS) ---
    # Calcoliamo quante righe e colonne servono
    cols = 3
    rows = math.ceil(num_paths / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() # Appiattiamo l'array per iterarci facilmente

    # Colori distinti
    colors = cm.get_cmap('jet', num_paths)

    for i in range(len(axes)):
        ax = axes[i]
        
        # Se abbiamo finito i percorsi ma ci sono ancora riquadri vuoti, spegnili
        if i >= num_paths:
            ax.axis('off')
            continue

        path_data = all_possible_paths[i]
        path = path_data['path']
        py, px = zip(*path)
        color = colors(i)

        # 1. Sfondo Heatmap
        ax.imshow(heatmap, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        
        # 2. Linea Percorso
        ax.plot(px, py, color=color, linewidth=3, alpha=0.9)

        # 3. Start e End
        start_y, start_x = py[0], px[0]
        end_y, end_x = py[-1], px[-1]

        # Cerchio Start
        ax.scatter(start_x, start_y, color=color, s=150, edgecolors='black', zorder=5)
        # X End
        ax.scatter(end_x, end_y, color='red', marker='x', s=100, linewidth=2, zorder=5)

        # Titolo del Riquadro
        ax.set_title(f"Path ID: {i}\nLen: {len(path)} px", fontsize=10, fontweight='bold', color='black')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Griglia salvata in: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    for img in TARGET_IMAGE_LIST:
        visualize_all_paths_grid(img)

