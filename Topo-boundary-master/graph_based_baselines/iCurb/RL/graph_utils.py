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

def process_single_component(G, nodes_to_visit, image_shape):
    """ Processes a single connected component and returns full pixel path + ordered waypoints. """
    
    # 1. Get Ordered Waypoints (Your targets!)
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
            path = nx.shortest_path(G, u, v, weight='weight')
            # Avoid duplicating junction points
            full_path_indices.extend(path if len(full_path_indices) == 0 else path[1:])
        except nx.NetworkXNoPath: 
            continue

    # 3. Loop Closure Logic
    if CLOSE_LOOP and len(ordered_wps) > 2:
        start_node = ordered_wps[0]
        end_node = ordered_wps[-1]
        
        # If start and end are NOT on borders, it's likely a closed loop (e.g., internal roundabout)
        if not (is_on_border(start_node, image_shape, BORDER_MARGIN) or 
                is_on_border(end_node, image_shape, BORDER_MARGIN)):
            try:
                path_back = nx.shortest_path(G, end_node, start_node, weight='weight')
                path_set = set(path_back)
                node_set = set(ordered_wps)
                
                # Only close if the return path doesn't cross other visited waypoints
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
    """
    H, W = heatmap.shape
    
    # 1. Binarization and Skeletonization
    # Use low threshold (0.2) to capture weak lines
    binary_road = (heatmap > 0.2).astype(np.uint8)
    skel = skeletonize(binary_road).astype(np.uint8)
    
    # Build graph on the full image skeleton
    G = build_pixel_graph(skel)

    # 2. Find target nodes (Endpoints)
    detected_nodes = peak_local_max(endpoints_map, min_distance=5, threshold_abs=0.2, exclude_border=False)
    
    skeleton_nodes = np.array(list(G.nodes()))
    if len(skeleton_nodes) == 0: return []
    
    # 3. Snap Endpoints to Skeleton
    tree = cKDTree(skeleton_nodes)
    # Search for nearest skeleton pixel within 15 pixels
    _, indices = tree.query(detected_nodes, distance_upper_bound=10.0)
    
    mapped_nodes = []
    num_total_nodes = len(skeleton_nodes)
    for idx in indices:
        if idx < num_total_nodes: # If valid neighbor found
            mapped_nodes.append(tuple(skeleton_nodes[idx]))
    mapped_nodes = list(set(mapped_nodes)) # Remove duplicates

    # 4. Process Connected Components
    # (Separates different roads that do not touch)
    components = list(nx.connected_components(G))
    results = []

    for comp in components:
        # Take only mapped nodes belonging to this component/road
        nodes_in_comp = [n for n in mapped_nodes if n in comp]
        
        # We need at least 2 points (start/end) to make a path
        if len(nodes_in_comp) < 2: 
            continue
            
        path_pixels, waypoints = process_single_component(G, nodes_in_comp, (H, W))
        
        # Filter out tiny paths (noise)
        if len(path_pixels) > 10:
            results.append({
                'pixels': path_pixels,     # Full road (pixel by pixel)
                'waypoints': waypoints     # YOUR KEY TARGETS (ordered)
            })

    return results

