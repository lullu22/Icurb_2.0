import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree

# ================= CONFIGURATION =================
# Flag to close loops if topology suggests it
CLOSE_LOOP = True 
# Margin in pixels to consider a node as "on the border"
BORDER_MARGIN = 15 
# Flag to enable Meta-Graph logic for ordering waypoints
META_GRAPH_LOGIC = False
# Distance threshold to consider primary and secondary nodes as distinct
DISTANCE_THRESHOLD_PS = 25.0
# =================================================

def is_on_border(node, shape, margin):
    """ Returns True if the node is close to the image borders. """
    h, w = shape
    r, c = node
    if r <= margin or r >= h - margin: 
        return True
    if c <= margin or c >= w - margin: 
        return True
    return False

def build_pixel_graph(skeleton):
    """ Builds a NetworkX graph from skeleton pixels. """
    rows, cols = np.where(skeleton > 0)
    G = nx.Graph()
    node_set = set(zip(rows, cols))
    
    for r, c in node_set:
        G.add_node((r, c))
        # 8-neighbor connectivity
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc 
                if (nr, nc) in node_set:
                    weight = np.sqrt(dr**2 + dc**2)
                    G.add_edge((r, c), (nr, nc), weight=weight)
    return G

if META_GRAPH_LOGIC:
    # Meta-Graph Logic
    def get_sequential_order(G, component_nodes):
        """
        Reorders nodes using Meta-Graph logic.
        Identifies if it's a line or a loop and orders waypoints accordingly.
        """
        # --- SORTING LOGIC ---
        MetaG = nx.Graph()
        MetaG.add_nodes_from(component_nodes)
        node_set = set(component_nodes) 
        n = len(component_nodes)
        
        # Build Meta-Graph based on reachability via skeleton
        for i in range(n):
            u = component_nodes[i] 
            try: 
                paths = nx.single_source_dijkstra_path(G, u, weight='weight')
            except: 
                continue

            for j in range(i + 1, n):
                v = component_nodes[j]
                if v not in paths: continue
                
                path_set = set(paths[v])
                intermediaries = path_set.intersection(node_set)
                
                # Direct connection check (only u and v as key nodes in the path)
                if len(intermediaries) == 2:
                    MetaG.add_edge(u, v, weight=len(paths[v]))

        # Determine start node based on graph degree
        degrees = dict(MetaG.degree())
        endpoints = [k for k, d in degrees.items() if d == 1]
        
        start_node = None
        if endpoints:
            # LINE Topology: Start from one endpoint
            endpoints = sorted(endpoints, key=lambda x: (x[0], x[1]))
            start_node = endpoints[0] 
        else:
            # LOOP Topology: Start from the top-leftmost node
            if len(component_nodes) > 0:
                sorted_nodes = sorted(component_nodes, key=lambda x: (x[0], x[1]))
                start_node = sorted_nodes[0]
            else: 
                return []

        return list(nx.dfs_preorder_nodes(MetaG, source=start_node))
else:
    # greedy Nearest Neighbor Logic
    def get_sequential_order(G, component_nodes):

        n = len(component_nodes)
        # Matrix NxN of distances initialized to Infinity
        dist_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(dist_matrix, 0)


        # Fill the matrix with real shortest path distances
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    # Calculate shortest path distance in the pixel graph
                    u = component_nodes[i]
                    v = component_nodes[j]
                    d = nx.shortest_path_length(G, u, v, weight='weight')
                    
                    # The matrix is symmetric (distance A->B = B->A)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                except nx.NetworkXNoPath:
                    pass  # Leave as infinity if no path exists

        # Find the most distant node pair
        flat_idx = np.argmax(dist_matrix)
        i_start, j_end = np.unravel_index(flat_idx, dist_matrix.shape)

        # IF THE GRAPH IS DISCONNECTED: Exit 
        if dist_matrix[i_start, j_end] == np.inf:
            return []

        # Choose deterministic start (e.g., smallest pixel coordinates)
        if component_nodes[i_start] < component_nodes[j_end]:
            start_idx = i_start
        else:
            start_idx = j_end
        

        ordered_indices = [start_idx]
        visited = {start_idx}
        curr_idx = start_idx

        while len(ordered_indices) < n:
            # 1. Get distances from current node
            dists = dist_matrix[curr_idx].copy()
            
            # 2. Mask visited nodes
            dists[list(visited)] = np.inf
            
            # 3. Find nearest unvisited node
            next_idx = np.argmin(dists)
            
            # stop if no reachable nodes remain
            if dists[next_idx] == np.inf:
                break
                
            # 4. move to next node
            ordered_indices.append(next_idx)
            visited.add(next_idx)
            curr_idx = next_idx
        
        return [component_nodes[i] for i in ordered_indices]

def process_single_component(G, nodes_to_visit, image_shape):
    """ Processes a single connected component and returns full pixel path + ordered waypoints. """
    
    # 1. Get Ordered Waypoints 
    try:
        ordered_wps = get_sequential_order(G, nodes_to_visit)
    except Exception:
        # Simple fallback if meta-graph fails
        ordered_wps = sorted(nodes_to_visit, key=lambda x: (x[0], x[1]))

    full_path_indices = []
    if not ordered_wps: return [], []

    # 2. Reconstruct pixel path (for internal calcs or visualization)
    for i in range(len(ordered_wps) - 1):
        u = ordered_wps[i]
        v = ordered_wps[i+1]
        try:
            path = nx.shortest_path(G, u, v, weight='weight') # In this case we use Dijkstra
            # Avoid duplicating junction points
            full_path_indices.extend(path if len(full_path_indices) == 0 else path[1:]) # we take the full path for the first segment, then skip first node 
        except nx.NetworkXNoPath: 
            continue

    # 3. Loop Closure Logic
    if CLOSE_LOOP and len(ordered_wps) > 2:
        start_node = ordered_wps[0]
        end_node = ordered_wps[-1]
        
        # If start and end are NOT on borders, it's likely a closed loop (e.g., internal roundabout)
        # check if start and end node are not on borders 
        if not (is_on_border(start_node, image_shape, BORDER_MARGIN) or 
                is_on_border(end_node, image_shape, BORDER_MARGIN)):
            try:
                path_back = nx.shortest_path(G, end_node, start_node, weight='weight')
                path_set = set(path_back)
                node_set = set(ordered_wps)
                
                # Only close if the return path doesn't cross other visited waypoints but only start/end nodes
                if len(path_set.intersection(node_set)) <= 2:
                    full_path_indices.extend(path_back[1:])
                    ordered_wps.append(start_node) # Adds start node to the end to close the loop
            except: 
                pass

    # RETURN: (All pixels, Ordered Waypoints)
    return full_path_indices, ordered_wps

def extract_paths_data(heatmap, endpoints_map):
    
    """
    MAIN FUNCTION FOR RL.
    Input:
        heatmap: np.array (H, W) float 0-1 or uint8 (road)
        endpoints_map: np.array (H, W) float 0-1 or uint8 (target points)
    Output:
        results: List of dictionaries. Each dict contains:
                 - 'pixels': list of (row, col) tuples for the full path
                 - 'waypoints': list of (row, col) tuples for KEY TARGETS
                 - 'waypoint_types': list of integers (0=primary, 1=secondary)
    """

    H, W = heatmap.shape
    
    # 1. Binarization and Skeletonization
    # Use low threshold (0.2) to capture weak lines
    binary_road = (heatmap > 0.2).astype(np.uint8)
    skel = skeletonize(binary_road).astype(np.uint8)
    
    # Build graph on the full image skeleton
    G = build_pixel_graph(skel)

    skeleton_nodes = np.array(list(G.nodes()))
    if len(skeleton_nodes) == 0:
        return []
    
    # Build KD-Tree 
    tree = cKDTree(skeleton_nodes)

    # 2. Find target nodes (Endpoint map)
    primary_nodes = peak_local_max(endpoints_map, min_distance=3, threshold_abs=0.1, exclude_border=False)

    snapped_primary_nodes = []
    if len(primary_nodes) > 0:
        # Snap to nearest skeleton pixel within 25 pixels
        _, indices = tree.query(primary_nodes, distance_upper_bound=25.0) # if we don't find anything within 25 pixels, we set index = len(skeleton_nodes)
        
        for idx in indices:
            if idx < len(skeleton_nodes): # If valid neighbor found (only value < len(skeleton_nodes) is valid)
                snapped_primary_nodes.append(tuple(skeleton_nodes[idx]))


    # 3. Find secondary target nodes (from heatmap peaks)
    secondary_nodes = peak_local_max(heatmap, min_distance=60, threshold_abs=0.6, exclude_border=False)

    snapped_secondary_nodes = []
    if len(secondary_nodes) > 0:
        # Snap to nearest skeleton pixel within 10 pixels
        _, indices = tree.query(secondary_nodes, distance_upper_bound=10.0)
        
        for idx in indices:
            if idx < len(skeleton_nodes): # If valid neighbor found
                snapped_secondary_nodes.append(tuple(skeleton_nodes[idx]))
        
    unique_primary_nodes = set(snapped_primary_nodes)
    unique_secondary_nodes = set(snapped_secondary_nodes)

    final_nodes_map = {}
    for first_node in unique_primary_nodes:
        final_nodes_map[first_node] = 'primary'


    if len(unique_primary_nodes) > 0:

        primary_tree = cKDTree(list(unique_primary_nodes))

        for second_node in unique_secondary_nodes:
            dist, _ = primary_tree.query(second_node) # check distance of secondary to nearest primary
            if dist >= DISTANCE_THRESHOLD_PS:
                final_nodes_map[second_node] = 'secondary'
    
    else: 
        for second_node in unique_secondary_nodes:
            final_nodes_map[second_node] = 'secondary'

    all_target_nodes = list(final_nodes_map.keys())
    
   

    # 4. Process Connected Components
    # (Separates different roads that do not touch)
    components = list(nx.connected_components(G))
    results = []

    for comp in components:
        # Take only mapped nodes belonging to this component/road
        nodes_in_comp = [n for n in all_target_nodes if n in comp]
        
        # We need at least 2 points (start/end) to make a path
        if len(nodes_in_comp) < 2: 
            continue
            
        path_pixels, waypoints = process_single_component(G, nodes_in_comp, (H, W))
        
        # Filter out tiny paths (noise)
        if len(path_pixels) > 10:

            # Get waypoint types for potential further use
            wp_types = [] 
            for wp in waypoints:
                # recover type from final map
                # default to 'secondary' if not found (should not happen)
                raw_type = final_nodes_map.get(wp, 'secondary') 
                
                # 0 = primary, 1 = secondary
                if raw_type == 'primary':
                    wp_types.append(0)
                else:
                    wp_types.append(1)

            results.append({
                'pixels': path_pixels,     # Full road (pixel by pixel)
                'waypoints': waypoints,
                'waypoint_types': wp_types   
            })

    return results

