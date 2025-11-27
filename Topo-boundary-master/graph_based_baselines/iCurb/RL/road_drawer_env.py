import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import pickle
import json
import math
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from skimage.draw import line
import matplotlib.pyplot as plt 

# --- CONFIGURATION ---
BASE_PROJECT_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/"
GT_GRAPH_DIR = os.path.join(BASE_PROJECT_DIR, "records", "gt", "gt_graphs_2") 
GT_JSON_PATH = os.path.join(BASE_PROJECT_DIR, "dataset_manhattan", "data_split.json")     

IMAGE_SIZE = 1000
CROP_SIZE = 128           
SAFE_THRESHOLD = 0.1    
BORDER_WIDTH = 10        

ANGULAR_ACTIONS_DEGREES = np.array([-45, -30, -15,-10,-5, 0, 5, 10, 15, 30, 45], dtype=np.float32) 
stop_action_index= len(ANGULAR_ACTIONS_DEGREES)
SEGMENT_LENGTH = 4 
WAYPOINT_MIN_DIST = 120

class RoadDrawerEnv(gym.Env):
    def __init__(self, split='train', device='cpu'):
        super().__init__()
        self.split = split
        self.device = device
        self.heatmap_dir = os.path.join(BASE_PROJECT_DIR, "RL", self.split, "heatmaps")
        self.gt_graph_dir = GT_GRAPH_DIR 

        # --- CARICAMENTO DATASET ---
        try:
            with open(GT_JSON_PATH, 'r') as f:
                dataset_info = json.load(f)
                split_key = split if split in dataset_info else 'train'
                self.file_list = dataset_info[split_key]
                print(f"--- ENV '{split}' CARICATO: {len(self.file_list)} immagini ---")
        except Exception as e:
            print(f"ERRORE JSON: {e}")
            self.file_list = []

        enable_overfit = True  

        target_image = "005250_04" # we can select the desired image for testing a single image 

        if enable_overfit: 
            self.file_list = [target_image]
        
        self.initial_file_list = self.file_list.copy()

        #self.action_space = spaces.Discrete(len(ANGULAR_ACTIONS_DEGREES) + 1) 
        self.action_space = spaces.Discrete(len(ANGULAR_ACTIONS_DEGREES)) #without stop action
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        
        self.current_pos = None
        self.current_heading = 0.0
        self.current_segment = [] 
        self.mission_path = []       
        self.current_wp_index = 0    
        self.current_target = None   
        
        self.heatmap = None
        self.gt_data = None
        self.steps_in_episode = 0
        self.max_steps = 600 
        self.drawn_nodes = set()
        self.current_map_name = None
        
        # --- SNAPSHOT  Memory ---
        self.last_finished_segment = [] 
        self.last_mission_path = []
        self.last_target = None
        self.last_heatmap = None 


        #new variable for persistency 
        self.mission_cache = None 
        self.last_run_success = False
        self.consecutive_failures = 0 
        if self.split == 'train':
            self.max_failures = 100 
        else: 
            self.max_failures = 0    #################################################################

        # conuter of roads drawn
        self.off_road_counter = 0
        self.steps_penalty = -0.005

        self.global_episode_count = 0

    def _find_all_valid_paths(self):
        """
        Trova TUTTI i percorsi validi sulla mappa corrente e li restituisce in una lista.
        """
        if self.gt_data is None or self.heatmap is None:
            return []

        gt_vertices = np.array(self.gt_data['vertices'])
        gt_adj = self.gt_data['adj']
        H, W = self.heatmap.shape
        margin = 20
        search_radius = 5

        # 1. Trova border indices
        border_indices = []
        for i, (y, x) in enumerate(gt_vertices):
            if (y < margin or y > H-margin or x < margin or x > W-margin):
                yc, xc =  int(y), int(x)
                yc_min = max(0, yc - search_radius)
                yc_max = min(H, yc + search_radius + 1)
                xc_min = max(0, xc - search_radius)
                xc_max = min(W, xc + search_radius + 1)
                local_patch = self.heatmap[yc_min:yc_max, xc_min:xc_max]
                if np.max(local_patch) > 0.2:
                    border_indices.append(i)

        possible_missions = []
        
        # 2. Dijkstra per trovare tutti i collegamenti
        for start_idx in border_indices:
            dist_matrix, predecessors = dijkstra(gt_adj, directed=False, indices=start_idx, return_predecessors=True)
            
            for end_idx in border_indices:
                if end_idx == start_idx: continue
                
                # Validazione distanza
                if dist_matrix[end_idx] == np.inf: continue
                if np.linalg.norm(gt_vertices[end_idx] - gt_vertices[start_idx]) <= 300: continue
                
                # Ricostruzione percorso
                full_path = self._reconstruct_path(predecessors, start_idx, end_idx)
                
                # Aggiungi alla lista
                possible_missions.append({
                    'path_indices': full_path, # Indici dei nodi
                    'start_coord': gt_vertices[start_idx],
                    'end_coord': gt_vertices[end_idx]
                })

        return possible_missions

    def reset(self, seed=None, options=None):
        # 1. Snapshot Memory
        if hasattr(self, 'current_segment') and len(self.current_segment) > 1:
            self.last_finished_segment = list(self.current_segment)
            self.last_mission_path = list(self.mission_path)
            self.last_target = self.current_target
            if self.heatmap is not None:
                self.last_heatmap = self.heatmap.copy()

        super().reset(seed=seed)
        
        # --- GESTIONE PERSISTENZA (RETRY DOPO FALLIMENTO) ---
        restor_old_mission = False
        if self.mission_cache is not None: 
            if (not self.last_run_success) and (self.consecutive_failures < self.max_failures): 
                restor_old_mission = True 
                self.consecutive_failures += 1
            else : 
                self.consecutive_failures = 0 
                self.last_run_success = False 

        if restor_old_mission: 
            data = self.mission_cache
            if self.current_map_name != data['map_name'] or self.heatmap is None: 
                self.file_list = [data['map_name']]
                self._load_new_map_data()
            
            self.mission_path = list(data['path'])
            self.current_pos = np.array(data['start_pos'])
            self.current_heading = data['start_heading']
            self.current_wp_index = 1 if len(self.mission_path) > 1 else 0 
            self.current_target = self.mission_path[self.current_wp_index]
            cy, cx = int(self.current_pos[0]), int(self.current_pos[1])
            self.current_segment = [(cy, cx)] 
            self.drawn_nodes = set([(cy, cx)])
            self.steps_in_episode = 0
            self.visited_pixels = set()          
            self.visited_pixels.add((cy, cx))    
            
            return self._get_observation(), {}

        # -------------------------------------------------------
        # NUOVA MISSIONE (Solo se ha avuto successo o ha superato i retry)
        # -------------------------------------------------------
        
        # 2. Carica Nuova Mappa (popola self.available_paths_cache)
        self._load_new_map_data()
        self.global_episode_count += 1
        
        path_selected_idx = -1 
        
        if self.available_paths_cache:
            # --- SELEZIONE NUOVO PATH ---
            path_selected_idx = np.random.randint(len(self.available_paths_cache))
            mission_data = self.available_paths_cache[path_selected_idx]
            
            path_indices = mission_data['path_indices']
            gt_vertices = np.array(self.gt_data['vertices'])
            
            # Campionamento
            sparse_indices = self._sample_path_by_distance(path_indices, gt_vertices, WAYPOINT_MIN_DIST)
            self.mission_path = [gt_vertices[i] for i in sparse_indices]
            
            self.current_pos = self.mission_path[0]
            self.current_wp_index = 1 
            self.current_target = self.mission_path[self.current_wp_index]
            
            # --- QUESTA PRINT RIMANE: TI AVVISA SOLO AL CAMBIO ---
            real_path_id = mission_data['global_id']

            if self.split == 'train':
                print(f"[{self.split.upper()}] Ep: {self.global_episode_count} | Map: {self.current_map_name} | Path ID: {real_path_id}/{(self.original_total_paths)-1}")
            
            # Calcolo Angolo
            dy = self.current_target[0] - self.current_pos[0]
            dx = self.current_target[1] - self.current_pos[1]
            base_angle = math.atan2(dy, dx)
            
        else:
            # FALLBACK
            print(f"[{self.split.upper()}] WARNING: Nessun percorso trovato! Usando Random Fallback.")
            self.current_pos = self._select_start_point_fallback()
            self.mission_path = [self.current_pos]
            self.current_target = self.current_pos
            self.current_wp_index = 0
            base_angle = np.random.uniform(0, 6.28)

        # Init Stato
        self.current_heading = base_angle + np.random.uniform(-0.3, 0.3)
        
        self.mission_cache = {
            'map_name': self.current_map_name,
            'start_pos': self.current_pos,
            'start_heading': self.current_heading,
            'path': list(self.mission_path),
            'path_id': path_selected_idx
        }

        self.visited_pixels = set()
        cy, cx = int(self.current_pos[0]), int(self.current_pos[1])
        self.visited_pixels.add((cy, cx))
        self.current_segment = [(cy, cx)] 
        self.drawn_nodes = set()
        self.drawn_nodes.add((cy, cx))
        self.steps_in_episode = 0
        
        return self._get_observation(), {}

        

    def step(self, action):
        
        done = False
        truncated = False
        
        reward = self.steps_penalty  # small penalty for each step taken 0,05

        
        y, x = self.current_pos
        H, W = self.heatmap.shape
        dist_to_wp = np.linalg.norm(self.current_pos - self.current_target)

        """
        # ---------------------------------------------------------
        # 1.  (STOP ACTION) - "FORCED EXPLORATION"
        # ---------------------------------------------------------
        if action == stop_action_index:
            # A. CHECK VITTORIA (Target FINALE raggiunto?)
            
            is_last_wp = (self.current_wp_index == len(self.mission_path) - 1)
            
            if is_last_wp and dist_to_wp < 15.0:
                reward += 100.0 # JACKPOT
                done = True
                self.last_run_success = True 
                return self._get_observation(), reward, done, truncated, {}

            # B. CHECK TEMPO MINIMO (Es. 400 Step)
            MIN_STEPS_TO_STOP = 400
            
            if self.steps_in_episode < MIN_STEPS_TO_STOP:
                # TENTATIVO DI STOP PREMATURO -> VIETATO!
                reward += -0.05  # Penalità: "È troppo presto per arrendersi!"
                done = False   # FORZIAMO A CONTINUARE
                self.steps_in_episode += 1 # Il tempo passa
            else:
                # TEMPO MINIMO SUPERATO -> RESA ACCETTATA
                # L'agente ha provato per 400 passi, non ce l'ha fatta, vuole uscire.
                reward += -5.0 # penalità per non aver finito la missione
                done = True    
            
            return self._get_observation(), reward, done, truncated, {}
        # ---------------------------------------------------------
        """
        # 2. MOVIMENTO (Il resto rimane uguale)
        angle_deg = ANGULAR_ACTIONS_DEGREES[action]
        delta = math.radians(angle_deg)
        new_heading = self.current_heading + delta
        target_y = y + math.sin(new_heading) * SEGMENT_LENGTH
        target_x = x + math.cos(new_heading) * SEGMENT_LENGTH
        
        rr, cc = line(int(round(y)), int(round(x)), int(round(target_y)), int(round(target_x))) # we take the line between the target e the actual position
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid], cc[valid]

        new_pixels_count = 0
        for r, c in zip(rr, cc):
            coord = (int(r), int(c))
            if coord not in self.visited_pixels:
                self.visited_pixels.add(coord)
                new_pixels_count += 1
    


        # Muro
        if len(rr) == 0:

            if self.steps_in_episode < 5: 
                reward += -0.01
                done = False 
                self.steps_in_episode += 1
            else:
                reward += -0.05; 
                done = True

            return self._get_observation(), reward, done, truncated, {}

        new_y, new_x = rr[-1], cc[-1] #  position of last pixel of the line 
        self.current_pos = np.array([new_y, new_x])
        self.current_heading = new_heading
        
        # Aggiornamento Traccia
        for r, c in zip(rr, cc):
            coord = (int(r), int(c))
            self.current_segment.append(coord)
            self.drawn_nodes.add(coord)
            
        self.steps_in_episode += 1

        # REWARD MOVIMENTO
        avg_intensity = np.mean(self.heatmap[rr, cc]) # average intensity of the line 
        new_dist_wp = np.linalg.norm(self.current_pos - self.current_target)
        dist_impr = dist_to_wp - new_dist_wp # distance from wp (positive ok - negative penalty)

        # directional reward
        vec_to_wp = self.current_target - self.current_pos
        angle_to_wp = math.atan2(vec_to_wp[0], vec_to_wp[1])
        raw_angle_diff = angle_to_wp - self.current_heading
        angle_diff = (raw_angle_diff + math.pi) % (2*math.pi) - math.pi 
        directional_bonus = math.cos(angle_diff)

        if avg_intensity < SAFE_THRESHOLD:

            reward += -0.1 * new_pixels_count
         
            self.off_road_counter += 1

            if self.off_road_counter >= 3 and self.steps_in_episode > 20:
                reward += -0.5 
                done = True 
            else: 
                done = False
                    
        else:
            # Sulla strada
            r_int = (avg_intensity * new_pixels_count) * 0.5
            r_nav = (directional_bonus-0.5) * 2.0
            r_stagnation = 0.0 
            if new_pixels_count == 0: 
                # aggiungiamo una penalty   
                r_stagnation = -0.5

            reward += (r_int + r_nav +r_stagnation)/10
            self.off_road_counter = 0
            
            # Checkpoint Waypoint 
            if new_dist_wp < 15.0:
                
                if self.current_wp_index == len(self.mission_path) - 1:
                    reward += 2.0 
                    done = True 
                    self.last_run_success = True 
                    
                else:
                    # Waypoint intermedio 
                    reward += 1.5
                    self.current_wp_index += 1
                    self.current_target = self.mission_path[self.current_wp_index]
            


        if self.steps_in_episode >= self.max_steps: 
            done = True

        #if self.steps_in_episode % 50 == 0:
        #    obs = self._get_observation()
        #    # Canale 0: Strada
        #    plt.imsave(f"debug_road_{self.steps_in_episode}.png", obs[0], cmap='gray')
        #    # Canale 1: Target + Bussola
        #    plt.imsave(f"debug_compass_{self.steps_in_episode}.png", obs[1], cmap='gray')

            
        return self._get_observation(), reward, done, truncated, {}

    def _reconstruct_path(self, preds, start, end):
        path = []
        curr = end
        while curr != -9999 and curr != start:
            path.append(curr)
            curr = preds[curr]
            if curr < 0: break 
        path.append(start)
        return path[::-1]

    def _sample_path_by_distance(self, indices, verts, thresh):
        if not indices: return []
        sampled = [indices[0]]
        last = indices[0]
        for i in range(1, len(indices)-1):
            curr = indices[i]
            if np.linalg.norm(verts[curr] - verts[last]) >= thresh:
                sampled.append(curr)
                last = curr
        if indices[-1] != sampled[-1]: sampled.append(indices[-1])
        return sampled

    def _get_observation(self):
        """
        Returns 2-channel observation.
        Channel 0: Road Heatmap
        Channel 1: Target (1.0) + Compass/Heading (0.5)
        """
        # 1. Setup Coordinate
        cy, cx = int(round(self.current_pos[0])), int(round(self.current_pos[1]))
        H, W = self.heatmap.shape
        
        # Inizializza i layer vuoti
        road_layer = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        target_layer = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

        # --- CANALE 0: ESTRAZIONE STRADA ---
        v_top = cy - (CROP_SIZE // 2)
        v_left = cx - (CROP_SIZE // 2)
        v_bottom = v_top + CROP_SIZE
        v_right = v_left + CROP_SIZE

        src_top = max(0, v_top); src_bottom = min(H, v_bottom)
        src_left = max(0, v_left); src_right = min(W, v_right)

        if src_top < src_bottom and src_left < src_right:
            patch = self.heatmap[src_top:src_bottom, src_left:src_right]
            dst_top = src_top - v_top
            dst_left = src_left - v_left
            dst_bottom = dst_top + patch.shape[0]
            dst_right = dst_left + patch.shape[1]
            road_layer[dst_top:dst_bottom, dst_left:dst_right] = patch

        # --- CANALE 1: TARGET + BUSSOLA ---
        center = CROP_SIZE // 2

        # A. Disegno il Target (Pallino/Box) - Intensità 1.0
        if self.current_target is not None:
            dy = self.current_target[0] - self.current_pos[0]
            dx = self.current_target[1] - self.current_pos[1]
            dist = math.sqrt(dy**2 + dx**2)
            angle = math.atan2(dy, dx)
            
            show_dist = min(dist, center - 4) 
            ty = int(center + math.sin(angle) * show_dist)
            tx = int(center + math.cos(angle) * show_dist)
            
            t_min_y = max(0, ty - 2); t_max_y = min(CROP_SIZE, ty + 3)
            t_min_x = max(0, tx - 2); t_max_x = min(CROP_SIZE, tx + 3)
            target_layer[t_min_y:t_max_y, t_min_x:t_max_x] = 1.0

        # B. DISEGNO BUSSOLA (Nuova aggiunta) - Intensità 0.5
        # Disegniamo una linea che parte dal centro e punta nella direzione dell'agente
        heading_len = CROP_SIZE // 4  # Lunghezza lancetta (es. 32 pixel)
        
        h_end_y = int(center + math.sin(self.current_heading) * heading_len)
        h_end_x = int(center + math.cos(self.current_heading) * heading_len)
        
        # Disegna linea
        start_point = (int(center), int(center))
        end_point = (h_end_x, h_end_y)
        
        cv2.line(target_layer, start_point, end_point, color=0.5, thickness=3)

        # Stack finale
        final_obs = np.stack([road_layer, target_layer], axis=0)
        return final_obs

    def _select_start_point_fallback(self):
        if self.gt_data is None: return np.array([100, 100])
        v = self.gt_data['vertices']
        return np.array(v[np.random.choice(len(v))])

    def _load_new_map_data(self):
        if not self.file_list: 
            self.file_list = self.initial_file_list.copy()
        
        self.current_map_name = np.random.choice(self.file_list)
        try:
            hpath = os.path.join(self.heatmap_dir, f"{self.current_map_name}.npy")
            gpath = os.path.join(self.gt_graph_dir, f"{self.current_map_name}.pickle")
            self.heatmap = np.load(hpath)
            with open(gpath, 'rb') as f: self.gt_data = pickle.load(f)
            self.drawn_nodes = set()
            
            # --- NUOVO: Calcoliamo subito tutti i percorsi disponibili per questa mappa ---
            all_paths = self._find_all_valid_paths()

            self.original_total_paths = len(all_paths)

            for i, path_data in enumerate(all_paths):
                path_data['global_id'] = i

            if not all_paths:
                self.available_paths_cache = []
                print(f"[{self.split.upper()}] WARNING: no valid paths found in {self.current_map_name}!")  
            elif self.split == 'train': 
                self.available_paths_cache = all_paths[:]
            
            else:
                self.available_paths_cache = all_paths[:]
            

        except Exception as e:
            print(f"Errore caricamento {self.current_map_name}: {e}")
            self.file_list = [f for f in self.file_list if f != self.current_map_name]
            self._load_new_map_data()

    def render_frame(self, save_path=None, final_steps=None):
        """
        Renderizza lo stato. 
        Se l'episodio è appena stato resettato (Step 0), 
        mostra il 'Fantasma' dell'episodio precedente SULLO SFONDO CORRETTO.
        """
        if self.current_pos is None: return
        
        # --- SELEZIONE DATI DA VISUALIZZARE ---
        use_ghost = False
        # Se siamo all'inizio E abbiamo dati in memoria
        if len(self.current_segment) <= 1 and len(self.last_finished_segment) > 1 and self.last_heatmap is not None:
            use_ghost = True

        if use_ghost:
            # Usiamo i dati del passato
            heatmap_to_show = self.last_heatmap
            segment_to_plot = self.last_finished_segment
            path_to_plot = self.last_mission_path
            tgt_to_plot = self.last_target
            color_line = 'magenta'
            label_txt = f'Finished' 
        else:
            # Usiamo i dati vivi
            heatmap_to_show = self.heatmap
            segment_to_plot = self.current_segment
            path_to_plot = self.mission_path
            tgt_to_plot = self.current_target
            color_line = 'cyan'
            label_txt = 'Live'

        # --- INIZIO PLOT ---
        plt.figure(figsize=(10, 10))
        H, W = heatmap_to_show.shape
        
        plt.imshow(heatmap_to_show, cmap='plasma', vmin=0, vmax=1)

        if len(segment_to_plot) > 0:
            sy, sx = zip(*segment_to_plot)
            plt.plot(sx, sy, color=color_line, linewidth=2, alpha=0.9, label=label_txt)
            plt.scatter(sx[0], sy[0], c='yellow', s=100, zorder=10, label='Start')
            
            # La testa dell'agente la mostriamo solo se è vivo (non nel replay statico)
            if not use_ghost: 
                plt.scatter(sx[-1], sy[-1], c='lime', s=80, zorder=11, edgecolors='black')

        if len(path_to_plot) > 0:
            py, px = zip(*path_to_plot)
            plt.plot(px, py, c='white', linestyle='--', alpha=0.5)
            plt.scatter(px, py, c='white', s=25, alpha=0.6, zorder=8)

        if tgt_to_plot is not None:
            if not use_ghost:
                plt.plot([self.current_pos[1], tgt_to_plot[1]], 
                         [self.current_pos[0], tgt_to_plot[0]], c='red', ls=':', alpha=0.7)
            plt.scatter(tgt_to_plot[1], tgt_to_plot[0], c='deepskyblue', s=350, marker='*', zorder=12)

        plt.title(f"Status: {label_txt} | Steps: {final_steps}")  ##### check if works
        plt.legend(loc='upper right')
        plt.axis('off')
        
        if save_path:
            try: plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            except: pass
            finally: plt.close()