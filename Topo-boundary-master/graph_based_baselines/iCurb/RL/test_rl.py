# Inference script for trained RL agent in RoadDrawerEnv.
# Work only in case where topological analysis is used for path extraction

#==================== IMPORTS ====================
import gymnasium as gym
import numpy as np
import cv2
import os
import time 
import torch
import math
import pickle
import matplotlib.pyplot as plt
import pandas as pd 
import networkx as nx
from graph_utils import extract_paths_data
from scipy.spatial import cKDTree
from stable_baselines3 import PPO
from tqdm import tqdm

# ==== CUSTOM IMPORTS ====
from road_drawer_env import RoadDrawerEnv
from graph_utils import extract_paths_data

# ==================== INFERENCE CONFIGURATION ====================

# Path to the trained model
MODEL_PATH = "./checkpoints_rl/rl_drawer_final.zip" 

# Output directories
OUTPUT_DIR = "./inference_results_Prova"        
OUTPUT_VIDEO_DIR = "./inference_videos_Prova" 
OUTPUT_CSV =  "./inference_CSV_Prova"
OUTPUT_GRAPH_DIR_PICKLE= "./inference_pickle_graphs_Prova"
OUTPUT_GRAPH_DIR = "./inference_graphs_Prova"


# data collection for metrics 
all_metrics_data = []

# TEST DATASET DIRECTORIES
TEST_MASK_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/seg/RL_Prova"

TEST_ENDPOINT_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/endpoint/RL_Prova"

TEST_RGB_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/dataset_prova/cropped_tiff"


# ANIMATION GENERATION FILTER
# ===========================
VIDEO_ENABLE = True 
TARGET_MAPS_LIST = ["002240_44","000235_21", "000227_01"]
FPS = 30
# ===========================


#RECOVERY ACTION 
# ===========================
RECOVERY = True
# ===========================


# SUBSET SELECTION 
# ===========================
TEST_START_IDX = 0   
TEST_END_IDX = None  # Set to None to process all maps
# ===========================


# compute metrics between agent trajectory and gt pixels 
def calculate_metrics(agent_traj, gt_pixels, tolerance = 5.0): 

    if len(agent_traj) < 2 or len(gt_pixels) < 2 : 
        return 0.0, 0.0, 0.0 
    
    agent_arr = np.array(agent_traj)
    gt_arr = np.array(gt_pixels)

    if gt_arr.shape[1] == 2:
        gt_arr = gt_arr[:, [1, 0]] # swap x,y -> y,x 

    # for pixel error
    #-----------------------
    tree = cKDTree(gt_arr)
    distances, _ = tree.query(agent_arr)
    #-----------------------

    # for covered path 
    #-----------------------
    tree_covered = cKDTree(agent_arr)
    distances_covered, _ = tree_covered.query(gt_arr)

    covered_pixels = np.sum(distances_covered <= tolerance)
    total_gt_pixels= len(gt_arr)

    if total_gt_pixels == 0: 
        return 0.0 

    accuracy_cov = (covered_pixels / total_gt_pixels) * 100
    #-----------------------

    return np.mean(distances), np.max(distances) , accuracy_cov

def run_inference():
    print("--- STARTING INFERENCE ---")

    # 1. INITIALIZE ENVIRONMENT
    # We use 'valid' split as a placeholder to avoid training logic
    env = RoadDrawerEnv(split='valid', device='cpu')
    
    env.heatmap_dir = TEST_MASK_DIR
    env.endpoint_dir = TEST_ENDPOINT_DIR

    # 2. RELOAD FILE LIST
    if os.path.exists(env.endpoint_dir):
        files = os.listdir(env.endpoint_dir)
        env.file_list = [os.path.splitext(f)[0] for f in files if f.endswith(('.png', '.jpg'))]
        env.file_list.sort() # Ensure deterministic order

        if TEST_END_IDX is not None: 
            end = TEST_END_IDX
        else: 
            end = len(env.file_list)

        env.file_list = env.file_list[TEST_START_IDX:end]

        print(f"--- FOUND {len(env.file_list)} MAPS IN TEST SET ---")
    else:
        print(f"CRITICAL ERROR: Test Endpoint directory not found: {env.endpoint_dir}")
        return

    # 3. LOAD MODEL
    print(f"Loading Model from: {MODEL_PATH}")
    try:
        # Load PPO model ensuring observation space compatibility
        model = PPO.load(MODEL_PATH, custom_objects={'observation_space': env.observation_space})
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True) # for images 
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True) # for videos 
    os.makedirs(OUTPUT_CSV, exist_ok=True) # for CSV
    os.makedirs(OUTPUT_GRAPH_DIR_PICKLE, exist_ok=True) # for pickle
    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True) # for graphs



    # 4. MAIN LOOP: PROCESS EACH MAP
    for map_name in tqdm(env.file_list, desc="Processing Maps"):
        
        # --- DECIDE WHETHER TO MAKE A VIDEO ---
        # Generate video only if the map name is in the target list and video generation is enabled
        MAKE_VIDEO = map_name in TARGET_MAPS_LIST and VIDEO_ENABLE

        # --- A. Load Map Data ---
        env.current_map_name = map_name
        
        # Load Heatmap (Mask)
        hpath = os.path.join(env.heatmap_dir, f"{map_name}.png")
        if not os.path.exists(hpath):
            hpath = hpath.replace('.png', '.jpg') # try jpg
        if not os.path.exists(hpath): 
            continue # Skip if missing
        
        heatmap_img = cv2.imread(hpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        env.heatmap = heatmap_img # Update environment state

        # Load Endpoint
        epath = os.path.join(env.endpoint_dir, f"{map_name}.png")
        if not os.path.exists(epath):
            epath = epath.replace('.png', '.jpg') # try jpg
        if not os.path.exists(epath): 
            continue # Skip if missing
        
        endpoint_img = cv2.imread(epath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


        # Load Real RGB/TIFF Image
        tpath = os.path.join(TEST_RGB_DIR, f"{map_name}.tiff")
        if not os.path.exists(tpath): 
            tpath = os.path.join(TEST_RGB_DIR, f"{map_name}.png")
        
        # (Heatmap version)
        vis_heatmap = (heatmap_img * 255).astype(np.uint8)
        vis_heatmap = cv2.cvtColor(vis_heatmap, cv2.COLOR_GRAY2BGR)
        
        # Real Image (Default to heatmap copy if real img is missing) 
        vis_real = vis_heatmap.copy()

        if os.path.exists(tpath):
            # Load UNCHANGED to handle 4 channels (TIFF)
            real_raw = cv2.imread(tpath, cv2.IMREAD_UNCHANGED)
            
            if real_raw is not None:
                # Handle 4 Channels: Keep only the first 3 (BGR), drop IR
                if len(real_raw.shape) == 3 and real_raw.shape[2] >= 3:
                    real_rgb = real_raw[:, :, :3] # we take only the first three channels (RGB)
                else:
                    real_rgb = real_raw
                
                
                #- control shape and type -
                # RESIZE to match Heatmap dimensions (Required for coordinate alignment), is not necessary, is already the same size
                h, w = heatmap_img.shape
                real_resized = cv2.resize(real_rgb, (w, h))
                
                # NORMALIZE if 16-bit, ensure uint8
                if real_resized.dtype != np.uint8:
                    if real_resized.max() > 255:
                        real_resized = (real_resized / 256).astype(np.uint8)
                    else:
                        real_resized = real_resized.astype(np.uint8)
                
                vis_real = real_resized 
                # ------------------------------------


        # --- B. Initialize Video Writer (ONLY IF NEEDED) ---
        out = None
        if MAKE_VIDEO:
            h, w = vis_heatmap.shape[:2]
            video_path = os.path.join(OUTPUT_VIDEO_DIR, f"video_{map_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            # Width is doubled because we stack images side-by-side
            out = cv2.VideoWriter(video_path, fourcc, FPS, (w * 2, h))

        # --- C. Extract Paths (Topological) ---
        all_paths_data = extract_paths_data(heatmap_img, endpoint_img)
        
        if not all_paths_data: 
            if out: 
                out.release()
            continue # Skip if no paths found 

        # --- D. Visualization Config ---
        # BGR Colors for OpenCV
        PATH_COLOR = (255, 255, 0)    # Cyan (Agent Trajectory)
        NODE_COLOR_PRIMARY = (0, 0, 255)    # Red (Primary Waypoints)
        NODE_COLOR_SECONDARY = (200, 255, 255) # Orange (Secondary Waypoints)
        EDGE_COLOR = (0,255,0)          # Lime (Edges between nodes)
        TRAJ_NODE_COLOR = (255, 0, 0) # Blue (Step Markers)
        STEP_MARKER_DIST = 30         # Draw a blue dot every 30 pixels
        SAMPLED_NODE_COLOR = (0, 255, 255) # Yellow (Sampled nodes every 25px)
        
           
        # --- GT SECTION (SAMPLED GRAPH: NODES EVERY 25px) ---
        STEP_SIZE = 25

        vis_gt_heatmap = vis_heatmap.copy()
        vis_gt_real = vis_real.copy()
        

        for p_data in all_paths_data: 
            original_waypoints = p_data.get('waypoints', [])
            gt_pixels = p_data.get('pixels', []) 

            if len(gt_pixels) < 2: 
                continue

            # 1. CREATE SAMPLED GRAPH FROM REAL PIXELS
            # Convert pixels to (x, y) format for calculations and drawing
            # gt_pixels is [row, col], so we flip -> (x, y)
            pixel_path_xy = [ (c, r) for r, c in gt_pixels ]
            
            sampled_nodes = []
            
            # Always add the first point (Start)
            if pixel_path_xy:
                sampled_nodes.append(pixel_path_xy[0])
            
            # Iterate through pixels and take one every 25 steps 
            # Note: since pixels are adjacent, the index corresponds roughly to distance, we assume 1 pixel = 1 unit distance
            
            for i in range(STEP_SIZE, len(pixel_path_xy), STEP_SIZE): # range(start, end, step)
                sampled_nodes.append(pixel_path_xy[i])
            
            # Always add the last point (End) if it's not too close to the previous one
            if pixel_path_xy:
                last_pt = pixel_path_xy[-1]
                if len(sampled_nodes) > 0:
                     dist_last = math.sqrt((last_pt[0]-sampled_nodes[-1][0])**2 + (last_pt[1]-sampled_nodes[-1][1])**2)
                     if dist_last > 5: # Avoid duplicates if the last step was close to the end
                         sampled_nodes.append(last_pt)
                else:
                     sampled_nodes.append(last_pt)

            # 2. DRAW EDGES (Lines between sampled nodes)
            for i in range(len(sampled_nodes) - 1):
                p1 = sampled_nodes[i]
                p2 = sampled_nodes[i+1]
                cv2.line(vis_gt_heatmap, p1, p2, EDGE_COLOR, 2)
                cv2.line(vis_gt_real, p1, p2, EDGE_COLOR, 2)

            # 3. DRAW SAMPLED NODES (Yellow - Small)
            for node in sampled_nodes:
                cv2.circle(vis_gt_heatmap, node, 6, SAMPLED_NODE_COLOR, -1)
                cv2.circle(vis_gt_real, node, 6, SAMPLED_NODE_COLOR, -1)

            # 4. OVERLAY ORIGINAL WAYPOINTS (Red - Large)
            # To show the "true" topological nodes
            # 4. OVERLAY ORIGINAL WAYPOINTS (COLORED BY TYPE)
            wp_types = p_data.get('waypoint_types', [])
            
            # Default to all primary if types missing
            if not wp_types:
                wp_types = [0] * len(original_waypoints)

            for i, wp in enumerate(original_waypoints): # (index , waypoint) i -> 0,1,2... and wp -> [y,x]
                center = (int(wp[1]), int(wp[0]))

                if wp_types[i] == 1:
                    color = NODE_COLOR_SECONDARY
                else:
                    color = NODE_COLOR_PRIMARY   

                cv2.circle(vis_gt_heatmap, center, 4, color, -1)
                cv2.circle(vis_gt_real, center, 4, color, -1)
        # -----------------------------------------------------------

        # metrics initialization 
        img_errors = []
        img_time_error = []
        img_successes = 0
        total_paths_in_img = 0
    
        #copy of rgb image for visualization of graphs 
        vis_graph_accumulated = vis_real.copy()

        # --- E. Simulation Loop ---
        for  path_idx ,path_data in enumerate(all_paths_data):
            raw_waypoints = path_data['waypoints']
            
            # 1. Setup Episode
            env.mission_path = [np.array(wp) for wp in raw_waypoints]
            if len(env.mission_path) < 2: 
                continue

            env.last_run_success = False

            # Set Initial State
            env.current_pos = env.mission_path[0]
            env.current_wp_index = 1
            env.current_target = env.mission_path[env.current_wp_index]
            
            # Calculate Heading
            dy = env.current_target[0] - env.current_pos[0]
            dx = env.current_target[1] - env.current_pos[1]
            env.current_heading = math.atan2(dy, dx)
            
            # Reset Counters
            env.current_segment = [(int(env.current_pos[0]), int(env.current_pos[1]))]
            env.visited_pixels = set()
            env.steps_in_episode = 0
            env.off_road_counter = 0
            
            # --- Draw Static Elements (Waypoints) ---
            # Draw immediately on both images so they appear fixed
            current_wp_types = path_data.get('waypoint_types', [])

            # Default to all primary if types missing
            if not current_wp_types:
                current_wp_types = [0] * len(raw_waypoints)

            for i, wp in enumerate(raw_waypoints):
                center = (int(wp[1]), int(wp[0]))
                
                if i < len(current_wp_types) and current_wp_types[i] == 1:
                    color = NODE_COLOR_SECONDARY
                else:
                    color = NODE_COLOR_PRIMARY
                
                cv2.circle(vis_heatmap, center, 4, color, -1)
                cv2.circle(vis_real, center, 4, color, -1)
            
            # Draw Start Point
            start_pt = (int(env.current_pos[1]), int(env.current_pos[0]))
            cv2.circle(vis_heatmap, start_pt, 6, (0, 255, 0), 2)
            cv2.circle(vis_real, start_pt, 6, (0, 255, 0), 2)

            # RECOVERY VARIABLES
            #=======================
            last_safe_pos = env.current_pos.copy() 
            last_safe_heading = env.current_heading
            last_safe_step_count = 0
            MAX_RETRIES = 3        
            GLOBAL_MAX_RECOVERIES = 20     
            retries_left = MAX_RETRIES 
            total_recoveries = 0 
            consecutive_off_road = 0   
            recovery_steps_left = 0   

            actions_tried_in_recovery = []
            #=======================
        

            trajectory = [] # here we save the trajectory of the agent
            node_positions = [] # here we save the positions of the nodes added to the graph by the agent


            # BACKUP VISUALIZATION (for recovery drawing) 
            #=======================
            vis_heatmap_backup = vis_heatmap.copy()
            vis_real_backup = vis_real.copy()
            #=======================

            # RECOVERY VARIABLES for trajectory drawing
            #=======================
            last_safe_traj_idx = 0
            last_safe_node_count = 0 
            last_safe_graph_id = 0
            #=======================

            # INITIALIZATION GRAPH
            #=======================
            G = nx.Graph()
            node_id = 0 
            #=======================

            # START TIME MEASUREMENT
            start_time = time.perf_counter()

            # 2. Run Agent Loop
            obs = env._get_observation()
            done = False
            
            prev_pos = (int(env.current_pos[1]), int(env.current_pos[0]))

            
            G.add_node(node_id,pos = prev_pos, type = "start")
            last_node_id = node_id
            

            trajectory.append(prev_pos)

            dist_acc = 0.0

            while not done:
                
                if recovery_steps_left > 0 and RECOVERY: 


                    if recovery_steps_left == 5:
                        dominant_action, _ = model.predict(obs, deterministic=True)
                        
                        obs_tensor, _ = model.policy.obs_to_tensor(obs) 

                        with torch.no_grad():
                            # we take the distribution 
                            dist = model.policy.get_distribution(obs_tensor)

                            # probability values 
                            probs = dist.distribution.probs.cpu().numpy()[0]
                            
                            probs[dominant_action] = 0

                            if actions_tried_in_recovery: 
                                probs[actions_tried_in_recovery] = 0.0
                            
                            total_prob = probs.sum()
                            if total_prob > 0:
                                probs = probs / total_prob 
                            else:
                                #limit case, all zero 
                                probs = np.ones_like(probs) / len(probs)

                        action = np.random.choice(len(probs), p=probs)
                        

                        current_banned = actions_tried_in_recovery + [int(dominant_action)]
                    
                        print(f"[RECOVERY MEMORY] retries_left: {retries_left} |" f"Banned: {list(set(current_banned))} -> Action: {action}")
                        
                        actions_tried_in_recovery.append(int(action))
                    
                    else: 
                        action, _ = model.predict(obs, deterministic=True) 

                    
                    recovery_steps_left -= 1
                    
                else: 
               
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, truncated, info = env.step(action)

                if RECOVERY:
                    
                    #=========== Recovery Action + check position ===========#

                    # A. Check position 
                    cur_y, cur_x = int(env.current_pos[0]), int(env.current_pos[1])
                    h_img, w_img = env.heatmap.shape
                    
                    is_on_road = False
                    if 0 <= cur_y < h_img and 0 <= cur_x < w_img:
                        if env.heatmap[cur_y, cur_x] > 0.25: # threshold
                            is_on_road = True

                    if is_on_road:
                        last_safe_pos = env.current_pos.copy() # Update checkpoint
                        last_safe_heading = env.current_heading
                        last_safe_step_count = env.steps_in_episode
                        consecutive_off_road = 0

                        actions_tried_in_recovery = []
                        retries_left = MAX_RETRIES

                        recovery_steps_left = 0

                        #update trajectory index for drawing
                        last_safe_traj_idx = len(trajectory)
                        last_safe_node_count = len(node_positions)
                        last_safe_graph_id = last_node_id
                        
                    else:
                        consecutive_off_road += 1 

                    # B. Recovery
                    needs_rescue = (consecutive_off_road > 5) or (done and not env.last_run_success)

                    if needs_rescue and retries_left > 0 and total_recoveries < GLOBAL_MAX_RECOVERIES:

                        # 1. go back in safe position
                        env.current_pos = last_safe_pos.copy()
                        env.current_heading = last_safe_heading
                        
                        # 2. Reset variables of Environment 
                        env.off_road_counter = 0 
                        env.steps_in_episode = last_safe_step_count
                        
                        # 3. Reset local variables
                        consecutive_off_road = 0
                        done = False 
                        
                        # 4.Random mode 
                        recovery_steps_left = 5 
                        retries_left -= 1
                        total_recoveries += 1 

                        # cancel last part of trajectory for visualization
                        trajectory = trajectory[:last_safe_traj_idx]

                        # cancel last part of node positions for visualization
                        node_positions = node_positions[:last_safe_node_count]

                        # remove last nodes from the graph
                        nodes_to_remove = [n for n in G.nodes() if isinstance(n, int) and n > last_safe_graph_id]

                        node_id = last_safe_graph_id
                        last_node_id = last_safe_graph_id

                        # Remove nodes from the graph
                        if nodes_to_remove:
                            G.remove_nodes_from(nodes_to_remove)

                        #restore backup visualization
                        vis_heatmap = vis_heatmap_backup.copy()
                        vis_real = vis_real_backup.copy()

                        # redraw the trajectory and node positions up to the last safe point
                        # we use polylines for better performance
                        if len(trajectory) > 1:
                            pts = np.array(trajectory, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(vis_heatmap, [pts], isClosed=False, color=PATH_COLOR, thickness=2)
                            cv2.polylines(vis_real, [pts], isClosed=False, color=PATH_COLOR, thickness=2)

                        for node_pos in node_positions:
                            cv2.circle(vis_heatmap, node_pos, 7, TRAJ_NODE_COLOR, -1) # Blue Dot
                            cv2.circle(vis_real, node_pos, 7, TRAJ_NODE_COLOR, -1)
                        
                        if len(trajectory) > 0:
                            prev_pos = trajectory[-1]
                        else:
                            prev_pos = (int(last_safe_pos[1]), int(last_safe_pos[0]))

                        dist_acc = 0.0 # reset distance accumulator because we are back in a safe position

                        # 5. update observation
                        obs = env._get_observation()

                        #visualization of recovery point
                        rec_pt = (int(last_safe_pos[1]), int(last_safe_pos[0]))

                        if retries_left == 2:   
                            rec_color = (0, 165, 255)  # orange 
                        elif retries_left == 1:  
                            rec_color = (0, 255, 255)  # yellow
                        else:                    
                            rec_color = (255, 0, 255)  # magenta

                        cv2.circle(vis_heatmap, rec_pt, 5, rec_color, -1)
                        cv2.circle(vis_real, rec_pt, 5, rec_color, -1)

                        # Backup visualization of recovery point
                        cv2.circle(vis_heatmap_backup, rec_pt, 5, rec_color, -1)
                        cv2.circle(vis_real_backup, rec_pt, 5, rec_color, -1)


                        prev_pos = (int(last_safe_pos[1]), int(last_safe_pos[0]))

                    #=======================================#
                    

                
                curr_pos = (int(env.current_pos[1]), int(env.current_pos[0]))

                trajectory.append(curr_pos)
                
                # DRAW TRAIL (Always draw on the image, regardless of video)
                cv2.line(vis_heatmap, prev_pos, curr_pos, PATH_COLOR, thickness=2)
                cv2.line(vis_real, prev_pos, curr_pos, PATH_COLOR, thickness=2)

                # Draw Markers every 30px
                step_dist = math.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
                dist_acc += step_dist
                
                if dist_acc >= STEP_MARKER_DIST:

                    node_id += 1 
                    G.add_node( node_id, pos = curr_pos, type = "path_node")
                    G.add_edge(last_node_id, node_id, weight = dist_acc)
                    last_node_id = node_id

                    # Save the position of the node
                    node_positions.append(curr_pos)
                    

                    cv2.circle(vis_heatmap, curr_pos, 7, TRAJ_NODE_COLOR, -1) # Blue Dot
                    cv2.circle(vis_real, curr_pos, 7, TRAJ_NODE_COLOR, -1)
                    dist_acc = 0

                prev_pos = curr_pos

                # WRITE FRAME (Only if making video)
                if MAKE_VIDEO and out is not None:
                    combined_frame = cv2.hconcat([vis_heatmap, vis_real])
                    out.write(combined_frame)

                if env.steps_in_episode > 1500: break

            #### CLOSE GRAPH ######
            if dist_acc > 0:
                node_id += 1 
                end_pos = curr_pos
                G.add_node( node_id, pos = end_pos, type = "end")
                G.add_edge(last_node_id, node_id, weight = dist_acc)
            else: 
                nx.set_node_attributes(G, {last_node_id: "end"}, name="type")

            #######################

            #### END CREATION TIME ###
            end_time = time.perf_counter()

            execution_time = end_time - start_time  #total time for path 

            steps_taken = len(trajectory)
            ms_per_steps = (execution_time/steps_taken)*1000 if steps_taken > 0 else 0 # time for each step in the path 
            

        

            # SAVE Graph ##########
            graph_filename = f"graph_{map_name}_path_{path_idx}.pickle"
            with open(os.path.join(OUTPUT_GRAPH_DIR_PICKLE, graph_filename), 'wb') as f: # 'wb' means write binary 
                pickle.dump(G, f)
            ######################

            #VISUALIZATION OF GRAPH ON THE RGB IMAGE
            EDGE_COLOR_GRAPH = (0, 165, 255)  # orange
            NODE_COLOR_GRAPH = (255, 0, 0)    # blu
            START_COLOR = (0, 255, 0)         # green
            END_COLOR = (0, 0, 255)           # Red

            # Edge
            for u, v, data in G.edges(data=True):
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                pt1 = (int(pos_u[0]), int(pos_u[1]))
                pt2 = (int(pos_v[0]), int(pos_v[1]))
                cv2.line(vis_graph_accumulated, pt1, pt2, EDGE_COLOR_GRAPH, thickness=2)

            # Nodes
            for node_id, data in G.nodes(data=True):
                pos = data['pos']
                pt = (int(pos[0]), int(pos[1]))
                    
                color = NODE_COLOR_GRAPH
                radius = 5
                if data.get('type') == 'start':
                    color = START_COLOR
                    radius = 6
                elif data.get('type') == 'end':
                    color = END_COLOR
                    radius = 6
                    
                cv2.circle(vis_graph_accumulated, pt, radius, color, -1)



            # Draw End Point (Success/Fail)
            col_end = (0, 255, 0) if env.last_run_success else (0, 0, 255)
            cv2.circle(vis_heatmap, curr_pos, 8, col_end, -1 if env.last_run_success else 2)
            cv2.circle(vis_real, curr_pos, 8, col_end, -1 if env.last_run_success else 2)

            gt_pixels = path_data.get('pixels',[])

            # dense sequence for the computation of the metrics 
            dense_trajectory = []

            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1): 
                    p1 = trajectory[i] # current 
                    p2 = trajectory[i+1] # next step 

                    # number of point between p1 and p2 

                    distance = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
                    

                    if distance == 0: # if the agent remain stationary 
                        continue  
                    ys = np.linspace(p1[0], p2[0], distance + 1)
                    xs = np.linspace(p1[1], p2[1], distance + 1) 

                    for k in range(len(ys)-1): 
                        dense_trajectory.append((int(ys[k]), int(xs[k])))

                dense_trajectory.append(trajectory[-1])
            
            else: 
                dense_trajectory = trajectory


            # Calculate Metrics
            avg_err, max_err, _ = calculate_metrics(trajectory, gt_pixels, tolerance=5.0)
            accuracy_cov = calculate_metrics(dense_trajectory, gt_pixels, tolerance=5.0)[2]


            all_metrics_data.append({
                "Image": map_name,
                "Path_ID": path_idx,
                "Success": 1 if env.last_run_success else 0,
                "Steps": len(trajectory),
                "Completion_Rate": round(accuracy_cov, 2),
                "Avg_Error_Px": round(avg_err, 2),
                "Max_Error_Px": round(max_err, 2),
                "Recoveries_Used": total_recoveries,
                "Time_sec": round(execution_time, 4),
                "Time_Per_step_ms": round(ms_per_steps, 2)
            })

            img_errors.append(avg_err)
            img_time_error.append(execution_time)
            total_paths_in_img +=1 
            if env.last_run_success: 
                img_successes +=1

            print(f"-> Path {path_idx}: Success={env.last_run_success}, time = {execution_time:.2f}s, compl={accuracy_cov:.2f}%, Err={avg_err:.2f}px, Recovery_used = {total_recoveries}")
            
            # Final Pause in Video
            if MAKE_VIDEO and out is not None:
                final_frame = cv2.hconcat([vis_heatmap, vis_real])
                for _ in range(FPS): out.write(final_frame)

        # Close video file if it was opened
        if out is not None: out.release()

        # saving graph 
        graph_img_filename = f"vis_graph_{map_name}_ALL_PATHS.png"
        cv2.imwrite(os.path.join(OUTPUT_GRAPH_DIR, graph_img_filename), vis_graph_accumulated)

        if total_paths_in_img > 0:
            avg_img_error = np.mean(img_errors)
            success_rate = (img_successes / total_paths_in_img) * 100
            avg_img_time_error = np.mean(img_time_error)
            tot_time = np.sum(img_time_error)


            
            print(f"\n[REPORT] Image: {map_name}")
            print(f"  - Paths Processed: {total_paths_in_img}")
            print(f"  - Success Rate:    {success_rate:.1f}%")
            print(f"  - Avg Error:       {avg_img_error:.2f} px")
            print(f"  - Avg Time:        {avg_img_time_error:.2f} s")
            print(f"  - Total Time:      {tot_time:.2f} s")
            print("-" * 30)

        # --- F. SAVE FINAL STATIC IMAGE (ALWAYS) ---
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        vis_heatmap_rgb = cv2.cvtColor(vis_heatmap, cv2.COLOR_BGR2RGB)
        vis_real_rgb = cv2.cvtColor(vis_real, cv2.COLOR_BGR2RGB)
        vis_gt_heatmap_rgb = cv2.cvtColor(vis_gt_heatmap,cv2.COLOR_BGR2RGB)
        vis_gt_real_rgb = cv2.cvtColor(vis_gt_real,cv2.COLOR_BGR2RGB)

        # Create Matplotlib Figure
        fig, axes = plt.subplots(2, 2, figsize=(35, 35))
        
        # Panel 1: Heatmap
        axes[0,0].imshow(vis_heatmap_rgb)
        axes[0,0].set_title("Agent on Heatmap", fontsize=15)
        axes[0,0].axis('off')

        # Panel 2: Real RGB
        axes[0,1].imshow(vis_real_rgb)
        axes[0,1].set_title("Agent on Real RGB", fontsize=15)
        axes[0,1].axis('off')

        # Panel 3: GT heatmap
        axes[1,0].imshow(vis_gt_heatmap_rgb)
        axes[1,0].set_title("Ground Truth on Heatmap", fontsize=15)
        axes[1,0].axis('off')

        # Panel 4: GT heatmap
        axes[1,1].imshow(vis_gt_real_rgb)
        axes[1,1].set_title("Ground Truth on Real RGB", fontsize=15)
        axes[1,1].axis('off')
 
        plt.tight_layout()
        
        # Save PNG
        save_path = os.path.join(OUTPUT_DIR, f"compare_{map_name}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig) # Free memory

    if all_metrics_data: 
        df_final = pd.DataFrame(all_metrics_data)


        total_paths = len(df_final)
        final_success_rate = (df_final["Success"].mean())*100
        final_avg_error = df_final["Avg_Error_Px"].mean()
        final_completion = (df_final["Completion_Rate"].mean())
        avg_time = df_final["Time_sec"].mean()
        avg_time_step = df_final["Time_Per_step_ms"].mean()


        print("\n" + "="*40)
        print("          FINAL REPORT")
        print("="*40)
        print(f"Total path processed: {total_paths}")
        print(f"Total Success Rate:  {final_success_rate:.2f}%")
        print(f"Total Completion Rate: {final_completion:.2f}%")
        print(f"Total Avg Error:     {final_avg_error:.2f} px")
        print(f"Total Avg Time:      {avg_time:.2f} s")
        print(f"Total Avg Time/step: {avg_time_step:.2f} ms")
        print("="*40 + "\n")

        df_final.to_csv(os.path.join(OUTPUT_CSV, "final_metrics.csv"), index=False)
    else:
        print("\n No collected Data.")



    print(f"\n--- INFERENCE COMPLETE ---")
   

if __name__ == "__main__":
    run_inference() 