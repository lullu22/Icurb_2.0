import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import pickle
import json
import math
import networkx as nx
from graph_utils import extract_paths_data
from scipy.spatial import cKDTree 
from scipy.sparse.csgraph import dijkstra 
from skimage.draw import line
import matplotlib.pyplot as plt  
from matplotlib.collections import LineCollection
from tqdm import tqdm 

# --- CONFIGURATION ---
BASE_PROJECT_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/"

# flag to enable topological pathing
USE_TOPOLOGICAL_PATHING = True
MASK_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/seg/test_PMM-NY"
ENDPOINT_DIRECTORY = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/endpoint/test_PMM-NY"

GT_GRAPH_DIR = os.path.join(BASE_PROJECT_DIR, "records", "gt", "gt_graphs_2_RDP") 
GT_JSON_PATH = os.path.join(BASE_PROJECT_DIR, "dataset_manhattan", "data_split.json")     

IMAGE_SIZE = 1000
CROP_SIZE = 128           
SAFE_THRESHOLD = 0.1    
BORDER_WIDTH = 10        

ANGULAR_ACTIONS_DEGREES = np.array([-45, -30, -15,-10,-5, 0, 5, 10, 15, 30, 45], dtype=np.float32) 
stop_action_index= len(ANGULAR_ACTIONS_DEGREES)
SEGMENT_LENGTH = 4 
WAYPOINT_MIN_DIST = 10.0

class RoadDrawerEnv(gym.Env):
    def __init__(self, split='train', device='cpu', enable_reverse_learning = True):
        super().__init__()
        self.split = split
        self.device = device

        self.enable_reverse_learning = enable_reverse_learning

        if self.enable_reverse_learning: 
            print("MODE: REVERSED LEARNING")
        else: 
            print("MODE: STANDARD RECOVERY")

        if USE_TOPOLOGICAL_PATHING:
            self.heatmap_dir= MASK_DIR
            self.endpoint_dir = ENDPOINT_DIRECTORY
            print(f" --- MODE TOPOLOGICAL PATHING ENABLED: Using heatmaps from {self.heatmap_dir} ---")
        else:
            self.heatmap_dir = os.path.join(BASE_PROJECT_DIR, "RL", self.split, "heatmaps")
            self.gt_graph_dir = GT_GRAPH_DIR 
            print(f" --- MODE STANDARD PATHING: Using heatmaps from {self.heatmap_dir} ---")


        self.file_list = []
        # --- LOAD DATASET  ---
        if USE_TOPOLOGICAL_PATHING:

            if os.path.exists(self.endpoint_dir):
                all_files_raw = os.listdir(self.endpoint_dir)
                all_files = [os.path.splitext(f)[0] for f in all_files_raw if f.endswith('.png')]

                all_files.sort()

                split_ratio = 0.9
                split_index = int(len(all_files) * split_ratio)

                if split == 'train':
                    self.file_list = all_files[:split_index]
                    print(f"---MODE TOPOLOGICAL PATHING: {len(self.file_list)} training images---")
                else:
                    self.file_list = all_files[split_index:]
                    print(f"---MODE TOPOLOGICAL PATHING: {len(self.file_list)} validation images---")
            else: 
                print(f"Error: Endpoint directory {self.end_point_dir} does not exist.")
            

        else:
            try:
                with open(GT_JSON_PATH, 'r') as f:
                    dataset_info = json.load(f)
                    split_key = split if split in dataset_info else 'train'
                    self.file_list = dataset_info[split_key]
                    print(f"---MODE STANDARD PATHING: {len(self.file_list)} images---")
            except Exception as e:
                print(f"ERRORE JSON: {e}")
                self.file_list = []

        enable_overfit = False  #############################################################################################

        training_images = ["005250_04", "002247_02","000227_10","000250_13","002242_22"]
        validation_images = ["980200_40"]
        
        if enable_overfit: 
            if self.split == 'train':
                self.file_list = training_images
            else:
                self.file_list = validation_images
        
        if not enable_overfit: 
            self.file_list = self._filter_dataset()

            if len(self.file_list) == 0:
                print(f"Error: No valid files found for split '{self.split}'. Check dataset and paths.")
            
            self.initial_file_list = self.file_list.copy()
            
        
        self.initial_file_list = self.file_list.copy()

        #self.action_space = spaces.Discrete(len(ANGULAR_ACTIONS_DEGREES) + 1) # WITH STOP ACTION 

        self.angular_actions = ANGULAR_ACTIONS_DEGREES
        self.n_angles = len(self.angular_actions)
        
        if self.enable_reverse_learning:
            
            self.action_space = spaces.Discrete(self.n_angles * 2)
        else:
           
            self.action_space = spaces.Discrete(self.n_angles)


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

        self.predicted_graph = None 
        self.last_node_id = 0
        
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
            self.max_failures = 0   

        # counter of roads drawn
        self.off_road_counter = 0
        self.steps_penalty = -0.01
        self.global_episode_count = 0

        # RECOVERY CONFIGURATION 
        #---------------------------------#
        self.max_retries = 3           
        self.recovery_penalty = -1.0  
        self.retries_left = 0
        self.last_safe_pos = None     
        self.last_safe_heading = 0.0
        self.recovery_events = []     
        #---------------------------------#

        # Report variables 
        self.session_attempts = 0       
        self.session_history = []

        self.trajectory = []
        

    def _filter_dataset(self): 
        print(f"--- pre_check dataset ({self.split.upper()}) for valid path ---")
        print(f"Initial dataset size: {len(self.file_list)} images")

        valid_files = []
        discarded_count = 0 

        iteration = tqdm(self.file_list, desc="Filtering dataset", unit="images")

        for img_name in iteration:
            try:
                
                if USE_TOPOLOGICAL_PATHING:
                    # 1. Check Heatmap (PNG o JPG)
                    hpath = os.path.join(self.heatmap_dir, f"{img_name}.png")
                    if not os.path.exists(hpath): 
                        hpath = os.path.join(self.heatmap_dir, f"{img_name}.jpg")
                    
                    # 2. Check Endpoint (PNG o JPG)
                    epath = os.path.join(self.endpoint_dir, f"{img_name}.png")
                    if not os.path.exists(epath): 
                        epath = os.path.join(self.endpoint_dir, f"{img_name}.jpg")

                    
                    if not os.path.exists(hpath) or not os.path.exists(epath):
                        discarded_count += 1
                        continue
                    
                    valid_files.append(img_name)
                    continue 

                
                else: 
                    heatmap_path = os.path.join(self.heatmap_dir, f"{img_name}.npy")
                    graph_path = os.path.join(self.gt_graph_dir, f"{img_name}.pickle")

                    if not os.path.exists(heatmap_path) or not os.path.exists(graph_path):
                        discarded_count += 1
                        continue

                    
                    self.heatmap = np.load(heatmap_path)
                    with open(graph_path, 'rb') as f:
                        self.gt_data = pickle.load(f)

                    possible_missions = self._find_all_valid_paths()

                    if len(possible_missions) > 0:
                        valid_files.append(img_name)
                    else:
                        discarded_count += 1
            
            except Exception as e:
                discarded_count += 1
                continue

            # clean memory
            self.heatmap = None 
            self.gt_data = None

        print(f"Valid files found: {len(valid_files)} | Discarded: {discarded_count}", end='\r')
        return valid_files
    
            



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
            self.last_recovery_events = list(getattr(self, 'recovery_events', []))

            if hasattr(self, 'trajectory'):
                self.last_trajectory = list(self.trajectory)
            else:
                self.last_trajectory = []

            if hasattr(self,'predicted_graph') and self.predicted_graph is not None:
                self.last_predicted_graph = self.predicted_graph
            else:
                self.last_predicted_graph = None

            if self.heatmap is not None:
                self.last_heatmap = self.heatmap.copy()

        super().reset(seed=seed)

        self.recovery_events = []
        
        
        restor_old_mission = False

        if self.mission_cache is not None: 
            if (not self.last_run_success) and (self.consecutive_failures < self.max_failures): 
                restor_old_mission = True 
                self.consecutive_failures += 1
            else : 

                if len(self.session_history) > 0:
                    n = len(self.session_history)
                    
                   # compute total means
                    all_steps = [x['steps'] for x in self.session_history]
                    all_recs = [x['recoveries'] for x in self.session_history]
                    
                    avg_steps_tot = np.mean(all_steps)
                    avg_recs_tot = np.mean(all_recs)
                    
                    # trend analysis
                    window = max(1, int(n * 0.1)) 
                    
                    start_steps = np.mean(all_steps[:window])
                    end_steps = np.mean(all_steps[-window:])
                    
                    start_recs = np.mean(all_recs[:window])
                    end_recs = np.mean(all_recs[-window:])
                    
                    # Indicatori visivi
                    trend_steps = "increase" if end_steps > start_steps else "decrese"
                    trend_recs = "decrease" if end_recs < start_recs else "increase"
                    
                    result = "SUCCESS" if self.last_run_success else "FAIL"

                    print(f"\n[MAP REPORT] Map: {self.current_map_name} | Result: {result}")
                    print(f" -> Attempts: {n}")
                    print(f" -> GLOBAL AVG: Steps={avg_steps_tot:.1f}, Recs={avg_recs_tot:.1f}")
                    print(f" -> TREND (First {window} vs Last {window}):")
                    print(f"    - Steps: {start_steps:.1f} -> {end_steps:.1f} ({trend_steps})")
                    print(f"    - Recs:  {start_recs:.1f} -> {end_recs:.1f} ({trend_recs})")
                    print("--------------------------------------------------", flush=True)
                # =============================================================

                # Reset for next map
                self.consecutive_failures = 0
                self.last_run_success = False
                self.session_attempts = 0
                self.session_history = []

        if restor_old_mission: 
            data = self.mission_cache
            if self.current_map_name != data['map_name'] or self.heatmap is None: 
                self.file_list = [data['map_name']]
                self._load_new_map_data()
            
            self.mission_path = list(data['path'])
            self.current_pos = np.array(data['start_pos'])
            self.current_heading = data['start_heading']

            # Recovery 

            # ==================================================
            self.retries_left = self.max_retries
            self.last_safe_pos = self.current_pos.copy()
            self.last_safe_heading = self.current_heading

            self.recovery_events = []
            # ==================================================



            self.current_wp_index = 1 if len(self.mission_path) > 1 else 0 
            self.current_target = self.mission_path[self.current_wp_index]
            cy, cx = int(self.current_pos[0]), int(self.current_pos[1])
            self.current_segment = [(cy, cx)] 
            self.drawn_nodes = set([(cy, cx)])
            self.steps_in_episode = 0
            self.visited_pixels = set()          
            self.visited_pixels.add((cy, cx))    

            self.trajectory = [(cy, cx, False)] # (position (y,x), is_reversing ( True or False))

            ## initialize variables for dynamic graphs only in validation/test
            if self.split != 'train':
                self.predicted_graph = nx.Graph() # empty graph
                start_node_pos = (float(self.current_pos[0]), float(self.current_pos[1]))

                self.predicted_graph.add_node(0, pos=start_node_pos)
                self.last_node_id = 0 # ID of the last added node
            else : 
                self.predicted_graph = None
                
            
            return self._get_observation(), {}

        # -------------------------------------------------------
        # NEW MISSION
        # -------------------------------------------------------
        
        self._load_new_map_data()
        self.global_episode_count += 1
        
        path_selected_idx = -1 
        
        if self.available_paths_cache:
            # --- SELEZIONE NUOVO PATH ---
            path_selected_idx = np.random.randint(len(self.available_paths_cache))
            mission_data = self.available_paths_cache[path_selected_idx]

            if USE_TOPOLOGICAL_PATHING:
                raw_waypoints = mission_data['waypoints']
                self.mission_path = [np.array(wp) for wp in raw_waypoints]
            else:
            
                path_indices = mission_data['path_indices']
                gt_vertices = np.array(self.gt_data['vertices'])
            
                # Campionamento
                sparse_indices = self._sample_path_by_distance(path_indices, gt_vertices, WAYPOINT_MIN_DIST)
                self.mission_path = [gt_vertices[i] for i in sparse_indices]

            if len(self.mission_path) < 2:
                return self.reset()
            
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

        # Init State
        self.current_heading = base_angle + np.random.uniform(-0.3, 0.3)


        # Recovery 
        # ==================================================
        self.retries_left = self.max_retries
        self.last_safe_pos = self.current_pos.copy()
        self.last_safe_heading = self.current_heading

        self.recovery_events = []
        # ==================================================

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

        self.trajectory = [(cy, cx, False)] # (position (y,x), is_reversing ( True or False))


        if self.split != 'train':
            self.predicted_graph = nx.Graph() # Crea nuovo grafo vuoto
            start_node_pos = (float(self.current_pos[0]), float(self.current_pos[1]))
            self.predicted_graph.add_node(0, pos=start_node_pos)
            self.last_node_id = 0 
        else:
            self.predicted_graph = None

                
        return self._get_observation(), {}

        

    def step(self, action):

        pos_before_move = self.current_pos.copy()


        # save distance from waypoint 
        prev_dist_wp = np.linalg.norm(pos_before_move - self.current_target)

        # we compute the distance from the safe point before starting moving 
        if self.last_safe_pos is not None:
            prev_dist_from_safety = np.linalg.norm(pos_before_move - self.last_safe_pos) 
        else: 
            prev_dist_from_safety = 0.0

        

        done = False
        truncated = False
        reward = self.steps_penalty  # small penalty for each step taken 0,01

        
        y, x = self.current_pos
        H, W = self.heatmap.shape
       

        current_step_length = SEGMENT_LENGTH 
        current_angle_idx = 0
        is_reversing = False

        if self.enable_reverse_learning: 

            if action < self.n_angles: 
                #forward 
                current_angle_idx = action 
                current_step_length = SEGMENT_LENGTH

            else: 
                #backward
                current_angle_idx = action - self.n_angles
                current_step_length = -(SEGMENT_LENGTH * 0.5)
                is_reversing = True

        else: 
            current_angle_idx = action 
            current_step_length = SEGMENT_LENGTH
            is_reversing = False

        # 2. Step
        angle_deg = self.angular_actions[current_angle_idx]
        delta = math.radians(angle_deg)
        new_heading = self.current_heading + delta

        target_y = y + math.sin(new_heading) * current_step_length
        target_x = x + math.cos(new_heading) * current_step_length

        
        rr, cc = line(int(round(y)), int(round(x)), int(round(target_y)), int(round(target_x))) # we take the line between the target e the actual position
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid], cc[valid]



        new_pixels_count = 0
        for r, c in zip(rr, cc):
            coord = (int(r), int(c))
            if coord not in self.visited_pixels:
                self.visited_pixels.add(coord)
                new_pixels_count += 1



        # Limit border 
        #==================================================
        if len(rr) == 0:
            
            if self.steps_in_episode < 5: 
                reward += -0.01
                done = False 
                self.steps_in_episode += 1
            else:
                reward += -0.1; 
                done = True

            return self._get_observation(), reward, done, truncated, {}
        #==================================================

        new_y, new_x = rr[-1], cc[-1] #  position of last pixel of the line 
        self.current_pos = np.array([new_y, new_x])
        self.current_heading = new_heading

        self.trajectory.append((new_y, new_x, is_reversing))
        
        # update drawn segment and nodes 
        for r, c in zip(rr, cc):
            coord = (int(r), int(c))
            self.current_segment.append(coord)
            self.drawn_nodes.add(coord)
            
        self.steps_in_episode += 1

        # GRAPH CONSTRUCTION 
        # ==================================================
        if self.predicted_graph is not None:

            last_node_pos = self.predicted_graph.nodes[self.last_node_id]['pos']
            last_node_array = np.array(last_node_pos)

            # comupute distance from last node
            dist_from_last_node = np.linalg.norm(self.current_pos - last_node_array)

            #threshold to add new node 
            NODE_ADD_DISTANCE = 30.0

            if dist_from_last_node >= NODE_ADD_DISTANCE:
                #create new id 
                new_id = self.last_node_id + 1
                current_node_pos = (float(self.current_pos[0]), float(self.current_pos[1]))

                #we add the new node
                self.predicted_graph.add_node(new_id, pos=current_node_pos)
                #we add the edge between last node and current node
                self.predicted_graph.add_edge(self.last_node_id, new_id, weight=dist_from_last_node)
                #update last node id
                self.last_node_id = new_id
        #===================================================

        
        # REWARD AND RECOVERY 
        # ==================================================
        avg_intensity = np.mean(self.heatmap[rr, cc]) # average intensity of the line 
        new_dist_wp = np.linalg.norm(self.current_pos - self.current_target)

        progress_to_target = prev_dist_wp- new_dist_wp

        if self.last_safe_pos is not None: 
            new_dist_from_safety = np.linalg.norm(self.current_pos - self.last_safe_pos)
            recovery_progress = prev_dist_from_safety - new_dist_from_safety
        else: 
            recovery_progress = 0.0

        # directional reward
        vec_to_wp = self.current_target - self.current_pos
        angle_to_wp = math.atan2(vec_to_wp[0], vec_to_wp[1])
        raw_angle_diff = angle_to_wp - self.current_heading
        angle_diff = (raw_angle_diff + math.pi) % (2*math.pi) - math.pi 
        directional_bonus = math.cos(angle_diff)

        if avg_intensity < SAFE_THRESHOLD:

            self.off_road_counter += 1
            reward += -0.1 * new_pixels_count
         
            if self.enable_reverse_learning: 
               
                if recovery_progress > 0.05: 

                    if is_reversing: 
                        pain = 0.05 

                    else: 
                        pain = 0.1
                        
                else: 
                    pain = 0.5 + (0.1 * self.off_road_counter)
                    
                limit = 20
                reward -= pain 

                if self.off_road_counter >= limit:
                    reward -= 2.0 
                    done = True  

            else:

                if self.off_road_counter >= 3 :

                    # Recovery logic
                    if self.retries_left >0:

                        self.retries_left -= 1
                        reward += self.recovery_penalty

                        self.recovery_events.append((self.last_safe_pos[0], self.last_safe_pos[1], self.retries_left)) # (y, x, retries left)

                        self.current_pos = self.last_safe_pos.copy()
                        self.current_heading = self.last_safe_heading
                        self.off_road_counter = 0

                        #print(f"DEBUG: Recovery used! Left: {self.retries_left}", flush= True)

                    else:
                        reward += -0.5 
                        done = True 
           
        else:

            self.last_safe_pos = self.current_pos.copy()
            self.last_safe_heading = self.current_heading

        
            self.off_road_counter = 0

            if self.enable_reverse_learning and is_reversing: 
                reward += -1.0

            r_int = 0.0
            r_nav = 0.0
            r_progress = 0.0


            if  self.enable_reverse_learning: 
                r_progress = min(progress_to_target,1) 

                if r_progress > 0.0: 
                    r_int = (avg_intensity * new_pixels_count) * 0.1
                    r_nav = (directional_bonus-0.5) 
            else: 

                r_int = (avg_intensity * new_pixels_count) * 0.5
                r_nav = (directional_bonus-0.5) * 2.0

        
            r_stagnation = 0.0 

            if new_pixels_count == 0:   
                r_stagnation = -0.5

            reward += (r_int + r_nav + r_stagnation + r_progress)/10
            
            
            # Checkpoint Waypoint 
            if new_dist_wp < 15.0:
                
                if self.current_wp_index == len(self.mission_path) - 1:
                    reward += 2.0 
                    done = True 
                    self.last_run_success = True 
                    
                else:
                    # partial waypoint
                    reward += 1.5
                    self.current_wp_index += 1
                    self.current_target = self.mission_path[self.current_wp_index]
            
        ## anti-camping rule
        start_pos = self.mission_path[0] 
        dist_from_start = np.linalg.norm(self.current_pos - start_pos)

        if self.steps_in_episode > 20 and dist_from_start < 15.0:
            reward += -10.0  
            done = True    

        if self.steps_in_episode >= self.max_steps: 
            done = True

        if done: 
            used = self.max_retries - self.retries_left
            self.session_attempts += 1
            self.session_history.append({'steps': self.steps_in_episode, 'recoveries': used})

  

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
            all_paths = []


            if USE_TOPOLOGICAL_PATHING:
                hpath = os.path.join(self.heatmap_dir, f"{self.current_map_name}.png")
                self.heatmap = cv2.imread(hpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

                epath = os.path.join(self.endpoint_dir, f"{self.current_map_name}.png")
                endpoints_img = cv2.imread(epath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

                all_paths = extract_paths_data(self.heatmap, endpoints_img)
                self.gt_data = None

            else: 
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
                self.file_list = [f for f in self.file_list if f != self.current_map_name]
                self._load_new_map_data()
                return
            
            self.available_paths_cache = all_paths[:]
            
        except Exception as e:
            print(f"Errore caricamento {self.current_map_name}: {e}")
            self.file_list = [f for f in self.file_list if f != self.current_map_name]
            self._load_new_map_data()

    def render_frame(self, save_path=None, final_steps=None):
        
        if self.current_pos is None: return
       
        use_ghost = False
        # If we are at the start AND have data in memory (from previous episode)
        if len(self.current_segment) <= 1 and len(self.last_finished_segment) > 1 and self.last_heatmap is not None:
            use_ghost = True

        if use_ghost:
            # --- PAST DATA (GHOST) ---
            heatmap_to_show = self.last_heatmap
            traj_data = self.last_trajectory
            segment_to_plot = self.last_finished_segment
            path_to_plot = self.last_mission_path
            tgt_to_plot = self.last_target
            
            # Retrieve the ghost graph
            graph_to_plot = getattr(self, 'last_predicted_graph', None)
            rec_events = getattr(self, 'last_recovery_events', [])

            color_line = 'magenta'
            col_fwd = 'lime'     
            col_rev = 'red'
            label_txt = f'Finished' 
        else:
            # --- CURRENT DATA (LIVE) ---
            heatmap_to_show = self.heatmap
            traj_data = getattr(self, 'trajectory', [])
            segment_to_plot = self.current_segment
            path_to_plot = self.mission_path
            tgt_to_plot = self.current_target
            
            # Retrieve the live graph
            graph_to_plot = getattr(self, 'predicted_graph', None)
            rec_events = getattr(self, 'recovery_events', [])

            color_line = 'cyan'
            col_fwd = 'line'
            col_rev = 'red'
            label_txt = 'Live'

        # PLOT - DUAL PANEL SETUP
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        ax1, ax2 = axes[0], axes[1]

        steps_val = final_steps if final_steps else self.steps_in_episode

        ax1.set_title(f"GEOMETRY (Standard) | {label_txt} | Steps: {steps_val}")
        ax2.set_title(f"BEHAVIOR (Red=Reverse) | {label_txt}")

        # --- LOOP FOR COMMON ELEMENTS ON BOTH GRAPHS ---
        # (Here we draw everything that shouldn't change: background, graph, target, etc.)
        for ax in axes:
            ax.axis('off')
            ax.imshow(heatmap_to_show, cmap='plasma', vmin=0, vmax=1) # Using plasma for visibility

            
            # 2. MISSION PATH , connection of waypoints
            if len(path_to_plot) > 0:
                py, px = zip(*path_to_plot)
                ax.plot(px, py, c='white', linestyle='--', alpha=0.5)
                ax.scatter(px, py, c='white', s=25, alpha=0.6, zorder=8)

            # 3. TARGET
            if tgt_to_plot is not None:
                if not use_ghost:
                    ax.plot([self.current_pos[1], tgt_to_plot[1]], 
                             [self.current_pos[0], tgt_to_plot[0]], c='red', ls=':', alpha=0.7)
                
                hit_box = plt.Circle((tgt_to_plot[1], tgt_to_plot[0]), 15.0, color='lime', fill=False, linewidth= 0.5,linestyle = "--", zorder=12)
                ax.add_patch(hit_box)

            # 4. RECOVERY EVENTS (COLORED Xs)
            for (ry, rx, lives_left) in rec_events:
                 if lives_left == 2: c_rec = 'orange'
                 elif lives_left == 1: c_rec = 'yellow'
                 else: c_rec = 'magenta'
                 ax.scatter(rx, ry, c=c_rec, s=25, zorder=20, edgecolors='white', marker='X')

            # 5. START POINT
            if len(segment_to_plot) > 0:
                sy, sx = segment_to_plot[0]
                ax.scatter(sx, sy, c='yellow', s=100, zorder=20, label='Start')

        # -------------------
        # 1. GRAPH (NODES AND EDGES) only ax1
            if graph_to_plot is not None:
                for u, v in graph_to_plot.edges():
                    if 'pos' in graph_to_plot.nodes[u] and 'pos' in graph_to_plot.nodes[v]:
                        pos_u = graph_to_plot.nodes[u]['pos']
                        pos_v = graph_to_plot.nodes[v]['pos']
                        ax1.plot([pos_u[1], pos_v[1]], [pos_u[0], pos_v[0]], c='red', linewidth=1.5, alpha=0.7, zorder=15)

                for node_id in graph_to_plot.nodes():
                    if 'pos' in graph_to_plot.nodes[node_id]:
                        pos = graph_to_plot.nodes[node_id]['pos']
                        ax1.scatter(pos[1], pos[0], c='red', s=30, zorder=16, edgecolors='black')

            

        # -------------------
        # LEFT (ax1)
        # -------------------
        if len(segment_to_plot) > 0:
            sy, sx = zip(*segment_to_plot)
            ax1.plot(sx, sy, color=color_line, linewidth=2, alpha=0.9, label=label_txt)
            
            # Agent head
            if not use_ghost: 
                ax1.scatter(sx[-1], sy[-1], c='lime', s=80, zorder=11, edgecolors='black') 

        # -------------------
        # RIGHT (ax2)
        # -------------------
        if len(traj_data) > 1:
            # Swap coordinates y,x -> x,y for matplotlib
            points = np.array([(p[1], p[0]) for p in traj_data]) 
            
            # Create segments
            segments = np.concatenate([points[:-1], points[1:]], axis=1).reshape(-1, 2, 2)
            
            # Colors based on is_reversing (which is the 3rd element in traj_data tuple)
            is_rev_list = [p[2] for p in traj_data[1:]]
            colors = [col_rev if rev else col_fwd for rev in is_rev_list]

            # Fast drawing with LineCollection
            lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.9)
            ax2.add_collection(lc)
            
            # Agent head
            if not use_ghost:
                ex, ey = points[-1]
                ax2.scatter(ex, ey, c='lime', s=80, zorder=11, edgecolors='black')


        if save_path:
            try: plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            except: pass
            finally: plt.close()