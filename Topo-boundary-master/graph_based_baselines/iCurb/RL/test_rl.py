import gymnasium as gym
import numpy as np
import cv2
import os
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.spatial import cKDTree
from stable_baselines3 import PPO
from tqdm import tqdm

# --- CUSTOM IMPORTS ---
from road_drawer_env import RoadDrawerEnv
from graph_utils import extract_paths_data

# ==================== INFERENCE CONFIGURATION ====================

# Path to the trained model
MODEL_PATH = "./checkpoints_rl/rl_drawer_final.zip" 

# Output directories
OUTPUT_DIR = "./inference_results"        
OUTPUT_VIDEO_DIR = "./inference_videos" 
OUTPUT_CSV =  "./inference_CSV"

# data collection for metrics 
all_metrics_data = []

# TEST DATASET DIRECTORIES
TEST_MASK_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/seg/RL"

TEST_ENDPOINT_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/init_vertex/records/endpoint/RL"

TEST_RGB_DIR = "/localhome/c-lcuffaro/Topo-boundary-master_def./Topo-boundary-master/graph_based_baselines/iCurb/dataset_manhattan/cropped_tiff"


# ANIMATION GENERATION FILTER
# ===========================
TARGET_MAPS_LIST = ["002240_44","000235_21", "000227_01"]
FPS = 30
# ===========================


#RECOVERY ACTION 
RECOVERY = True

def calculate_metrics(agent_traj, gt_pixels): 

    if len(agent_traj) < 2 or len(gt_pixels) < 2 : 
        return 0.0, 0.0 
    
    agent_arr = np.array(agent_traj)
    gt_arr = np.array(gt_pixels)

    if gt_arr.shape[1] == 2:
        gt_arr = gt_arr[:, [1, 0]]

    tree = cKDTree(gt_arr)
    distances, _ = tree.query(agent_arr)

    return np.mean(distances), np.max(distances)
    

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

    # 4. MAIN LOOP: PROCESS EACH MAP
    for map_name in tqdm(env.file_list, desc="Processing Maps"):
        
        # --- DECIDE WHETHER TO MAKE A VIDEO ---
        # Generate video only if the map name is in the target list
        MAKE_VIDEO = map_name in TARGET_MAPS_LIST
        
        # --- A. Load Map Data Manually ---
        env.current_map_name = map_name
        
        # Load Heatmap (Mask)
        hpath = os.path.join(env.heatmap_dir, f"{map_name}.png")
        if not os.path.exists(hpath):
            hpath = hpath.replace('.png', '.jpg')
        if not os.path.exists(hpath): 
            continue # Skip if missing
        
        heatmap_img = cv2.imread(hpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        env.heatmap = heatmap_img # Update environment state

        # Load Endpoint
        epath = os.path.join(env.endpoint_dir, f"{map_name}.png")
        if not os.path.exists(epath):
            epath = epath.replace('.png', '.jpg')
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
        
        # Real Image Canvas (Default to heatmap copy if real img is missing) 
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
                
                # RESIZE to match Heatmap dimensions (Required for coordinate alignment), is not necessary 
                h, w = heatmap_img.shape
                real_resized = cv2.resize(real_rgb, (w, h))
                
                # Normalize if 16-bit, ensure uint8
                if real_resized.dtype != np.uint8:
                    if real_resized.max() > 255:
                        real_resized = (real_resized / 256).astype(np.uint8)
                    else:
                        real_resized = real_resized.astype(np.uint8)
                
                vis_real = real_resized

     



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
        NODE_COLOR = (0, 0, 255)      # Red (Ideal Waypoints)
        EDGE_COLOR = (0,255,0)          # Lime
        TRAJ_NODE_COLOR = (255, 0, 0) # Blue (Step Markers)
        STEP_MARKER_DIST = 30         # Draw a blue dot every 30 pixels
        SAMPLED_NODE_COLOR = (0, 255, 255) # Yellow (Sampled nodes every 30px)
        ORIGINAL_WP_COLOR = (0, 0, 255)    # Red (Original Waypoints/Intersections)

           
        # --- GT SECTION (SAMPLED GRAPH: NODES EVERY 30px) ---
        vis_gt_heatmap = vis_heatmap.copy()
        vis_gt_real = vis_real.copy()
        

        for p_data in all_paths_data: 
            original_waypoints = p_data.get('waypoints', [])
            gt_pixels = p_data.get('pixels', []) 

            if len(gt_pixels) < 2: continue

            # 1. CREATE SAMPLED GRAPH FROM REAL PIXELS
            # Convert pixels to (x, y) format for calculations and drawing
            # gt_pixels is [row, col], so we flip -> (x, y)
            pixel_path_xy = [ (c, r) for r, c in gt_pixels ]
            
            sampled_nodes = []
            
            # Always add the first point (Start)
            if pixel_path_xy:
                sampled_nodes.append(pixel_path_xy[0])
            
            # Iterate through pixels and take one every 30 steps (approx 30px distance)
            # Note: since pixels are adjacent, the index corresponds roughly to distance
            STEP_SIZE = 25
            for i in range(STEP_SIZE, len(pixel_path_xy), STEP_SIZE):
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
            for wp in original_waypoints:
                center = (int(wp[1]), int(wp[0]))
                cv2.circle(vis_gt_heatmap, center, 4, ORIGINAL_WP_COLOR, -1)
                cv2.circle(vis_gt_real, center, 4, ORIGINAL_WP_COLOR, -1)
        # -----------------------------------------------------------

        # metrics initialization 
        img_errors = []
        img_successes = 0
        total_paths_in_img = 0
    
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
            # Draw immediately on both canvases so they appear fixed
            for wp in raw_waypoints:
                center = (int(wp[1]), int(wp[0]))
                cv2.circle(vis_heatmap, center, 4, NODE_COLOR, -1)
                cv2.circle(vis_real, center, 4, NODE_COLOR, -1)
            
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
            retries_left = MAX_RETRIES 
            total_recoveries = 0 
            consecutive_off_road = 0   
            recovery_steps_left = 0   

            actions_tried_in_recovery = []
            #=======================


            trajectory = []

            # 2. Run Agent Loop
            obs = env._get_observation()
            done = False
            
            prev_pos = (int(env.current_pos[1]), int(env.current_pos[0]))

            trajectory.append(prev_pos)

            dist_acc = 0.0

            while not done:
                
                if recovery_steps_left > 0 and RECOVERY: 


                    if recovery_steps_left == 10:
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
                    
                        print(f"  [RECOVERY MEMORY] retries_left: {retries_left} | "
                          f"Banned: {list(set(current_banned))} -> Action: {action}")
                        
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
                        if env.heatmap[cur_y, cur_x] > 0.15: # threshold
                            is_on_road = True

                    if is_on_road:
                        last_safe_pos = env.current_pos.copy() # Update checkpoint
                        last_safe_heading = env.current_heading
                        last_safe_step_count = env.steps_in_episode
                        consecutive_off_road = 0

                        actions_tried_in_recovery = []
                        retries_left = MAX_RETRIES
                        
                    else:
                        consecutive_off_road += 1 

                    # B. Recovery
                    needs_rescue = (consecutive_off_road > 5) or (done and not env.last_run_success)

                    if needs_rescue and retries_left > 0:

                        # 1. go back in safe position
                        env.current_pos = last_safe_pos.copy()
                        env.current_heading = last_safe_heading
                        
                        # 2. Reset variables ofEnvironment 
                        env.off_road_counter = 0 
                        env.steps_in_episode = last_safe_step_count
                        
                        # 3. Reset local variables
                        consecutive_off_road = 0
                        done = False 
                        
                        # 4.Random mode 
                        recovery_steps_left = 10 
                        retries_left -= 1
                        total_recoveries += 1 
                        
                        # 5. update observation 
                        obs = env._get_observation()
                        
                        #visualization 
                        rec_pt = (int(last_safe_pos[1]), int(last_safe_pos[0]))

                        if retries_left == 2:   
                            rec_color = (0, 165, 255)  #orange 
                        elif retries_left == 1:  
                            rec_color = (0, 255, 255)  #yellow
                        else:                    
                            rec_color = (255, 0, 255)  #purple

                        cv2.circle(vis_heatmap, rec_pt, 5, rec_color, -1)
                        cv2.circle(vis_real, rec_pt, 5, rec_color, -1)


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
                    cv2.circle(vis_heatmap, curr_pos, 7, TRAJ_NODE_COLOR, -1) # Blue Dot
                    cv2.circle(vis_real, curr_pos, 7, TRAJ_NODE_COLOR, -1)
                    dist_acc = 0

                prev_pos = curr_pos

                # WRITE FRAME (Only if making video)
                if MAKE_VIDEO and out is not None:
                    combined_frame = cv2.hconcat([vis_heatmap, vis_real])
                    out.write(combined_frame)

                if env.steps_in_episode > 1500: break

            # Draw End Point (Success/Fail)
            col_end = (0, 255, 0) if env.last_run_success else (0, 0, 255)
            cv2.circle(vis_heatmap, curr_pos, 8, col_end, -1 if env.last_run_success else 2)
            cv2.circle(vis_real, curr_pos, 8, col_end, -1 if env.last_run_success else 2)

            gt_pixels = path_data.get('pixels',[])

            avg_err, max_err = calculate_metrics(trajectory, gt_pixels)

            all_metrics_data.append({
                "Image": map_name,
                "Path_ID": path_idx,
                "Success": 1 if env.last_run_success else 0,
                "Steps": len(trajectory),
                "Avg_Error_Px": round(avg_err, 2),
                "Max_Error_Px": round(max_err, 2),
                "Recoveries_Used": total_recoveries
            })

            img_errors.append(avg_err)
            total_paths_in_img +=1 
            if env.last_run_success: 
                img_successes +=1

            print(f"-> Path {path_idx}: Success={env.last_run_success}, Err={avg_err:.2f}px, Recovery_used = {total_recoveries}")
            
            # Final Pause in Video
            if MAKE_VIDEO and out is not None:
                final_frame = cv2.hconcat([vis_heatmap, vis_real])
                for _ in range(FPS): out.write(final_frame)

        # Close video file if it was opened
        if out is not None: out.release()

        if total_paths_in_img > 0:
            avg_img_error = np.mean(img_errors)
            success_rate = (img_successes / total_paths_in_img) * 100
            
            print(f"\n[REPORT] Image: {map_name}")
            print(f"  - Paths Processed: {total_paths_in_img}")
            print(f"  - Success Rate:    {success_rate:.1f}%")
            print(f"  - Avg Error:       {avg_img_error:.2f} px")
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

        print("\n" + "="*40)
        print("          FINAL REPORT")
        print("="*40)
        print(f"Total path processed: {total_paths}")
        print(f"Total Success Rate:  {final_success_rate:.2f}%")
        print(f"Total Avg Error:     {final_avg_error:.2f} px")
        print("="*40 + "\n")

        df_final.to_csv(os.path.join(OUTPUT_CSV, "final_metrics.csv"), index=False)
    else:
        print("\n No collected Data.")



    print(f"\n--- INFERENCE COMPLETE ---")
    print(f"Images saved in: {OUTPUT_DIR}")
    print(f"Videos saved in: {OUTPUT_VIDEO_DIR}")
    print(f"CSV saved in: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_inference() 