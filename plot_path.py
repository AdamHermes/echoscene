import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not found")
    exit(1)

def is_clear(x, z, furniture_boxes, l, w):
    clearance = 0.3
    for box in furniture_boxes:
        min_x, max_x, min_z, max_z = box
        if (min_x - clearance <= x <= max_x + clearance) and (min_z - clearance <= z <= max_z + clearance):
            return False
    if x < clearance or x > l - clearance or z < clearance or z > w - clearance:
        return False
    return True

scene_name = "SecondBedroom-6482"
folders = [
    "baseline",
    "physcene_guidance",
    "released_full_model"
]

controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)

output_dir = "/Users/lehoangan/.gemini/antigravity-cli/brain/c21352c1-abf1-4f85-aaae-e44c32e9145d/scratch"
os.makedirs(output_dir, exist_ok=True)

for folder in folders:
    json_path = f"../output/{folder}/vis/2050/procthor_scenes/{scene_name}.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        house_data = json.load(f)
        
    rd = house_data.get('room_dims', {})
    l = rd.get('l', 0)
    w = rd.get('w', 0)
    furniture_boxes = house_data.get('furniture_boxes', [])
    
    controller.reset(scene=house_data)
    
    spawn_x, spawn_z = l / 2.0, w / 2.0
    found = False
    for x in np.arange(0.25, l, 0.25):
        for z in np.arange(0.25, w, 0.25):
            if is_clear(x, z, furniture_boxes, l, w):
                spawn_x, spawn_z = x, z
                found = True
                break
        if found: break
        
    controller.step(action="Teleport", position={"x": spawn_x, "y": 0.9, "z": spawn_z}, forceAction=True)
    
    # ---------------------------------------------------------
    # 4. PATHFINDING (Python-native to avoid Unity NavMesh bugs)
    # ---------------------------------------------------------
    import networkx as nx
    from shapely.geometry import Polygon, LineString
    
    objects_raw = house_data.get('objects_raw', [])
    min_x = min(o['position']['x'] - o['size']['x']/2 for o in objects_raw)
    max_x = max(o['position']['x'] + o['size']['x']/2 for o in objects_raw)
    min_z = min(o['position']['z'] - o['size']['z']/2 for o in objects_raw)
    max_z = max(o['position']['z'] + o['size']['z']/2 for o in objects_raw)
    fx = (max_x + min_x) / 2.0
    fz = (max_z + min_z) / 2.0
        
    offset_x = -(fx - l/2.0)
    offset_z = -(fz - w/2.0)
    
    # Rebuild polygons mathematically
    polygons = []
    for obj in objects_raw:
        min_y = max(0.0, obj['position']['y'] - obj['size']['y'] / 2.0)
        if min_y >= 1.0: continue
        cx = obj['position']['x'] + offset_x
        cz = obj['position']['z'] + offset_z
        obj_w, obj_d = obj['size']['x'], obj['size']['z']
        rad = obj['rotation']['y']
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        hw, hd = obj_w / 2.0, obj_d / 2.0
        def rot(lx, lz): return cx + lx*cos_t - lz*sin_t, cz + lx*sin_t + lz*cos_t
        corners = [rot(-hw, -hd), rot(hw, -hd), rot(hw, hd), rot(-hw, hd)]
        polygons.append(Polygon(corners).buffer(0.22)) # Tighter buffer to prevent closing off narrow pathways
        
    G = nx.Graph()
    nodes = [(x, z) for x in np.arange(0.2, l-0.2, 0.1) for z in np.arange(0.2, w-0.2, 0.1)]
    valid_nodes = []
    for x, z in nodes:
        pt = Polygon([(x, z), (x+0.01, z), (x+0.01, z+0.01), (x, z+0.01)])
        if not any(poly.intersects(pt) for poly in polygons):
            valid_nodes.append((round(x, 2), round(z, 2)))
            
    valid_nodes_set = set(valid_nodes)
    for (x, z) in valid_nodes:
        G.add_node((x, z))
        for dx, dz in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1), (-0.1, -0.1), (0.1, 0.1), (-0.1, 0.1), (0.1, -0.1)]:
            nx_z = round(x + dx, 2)
            nz_z = round(z + dz, 2)
            if (nx_z, nz_z) in valid_nodes_set:
                G.add_edge((x, z), (nx_z, nz_z), weight=math.hypot(dx, dz))
                
    start_pos = {"x": spawn_x, "z": spawn_z}
    
    # Snap start to grid
    start_n = min(valid_nodes, key=lambda n: math.dist([start_pos['x'], start_pos['z']], n))
    
    # Ensure goal is in the same connected component so a path always exists
    reachable_nodes = list(nx.node_connected_component(G, start_n))
    if len(reachable_nodes) < 2:
        print(f"{folder}: Agent is completely stuck. Skipping.")
        continue
        
    goal_n = max(reachable_nodes, key=lambda n: math.dist(start_n, n))
    goal_pos = {"x": goal_n[0], "z": goal_n[1]}
    
    try:
        raw_path = nx.shortest_path(G, source=start_n, target=goal_n, weight='weight')
        # Line-of-Sight Smoothing
        smoothed_path = [raw_path[0]]
        curr = 0
        while curr < len(raw_path) - 1:
            furthest = curr + 1
            for j in range(len(raw_path) - 1, curr, -1):
                line = LineString([raw_path[curr], raw_path[j]])
                if not any(poly.intersects(line) for poly in polygons):
                    furthest = j
                    break
            smoothed_path.append(raw_path[furthest])
            curr = furthest
        path_points = [{'x': p[0], 'z': p[1]} for p in smoothed_path]
    except (nx.NetworkXNoPath, ValueError):
        path_points = [start_pos, goal_pos]
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, l)
    ax.set_ylim(0, w)
    ax.set_aspect('equal')
    
    # Plot room bounds
    ax.add_patch(patches.Rectangle((0, 0), l, w, fill=False, edgecolor='black', linewidth=3))
    
    # Re-calculate offset logic from convert script
    objects_raw = house_data.get('objects_raw', [])
    min_x = min(o['position']['x'] - o['size']['x']/2 for o in objects_raw)
    max_x = max(o['position']['x'] + o['size']['x']/2 for o in objects_raw)
    min_z = min(o['position']['z'] - o['size']['z']/2 for o in objects_raw)
    max_z = max(o['position']['z'] + o['size']['z']/2 for o in objects_raw)
    fx = (max_x + min_x) / 2.0
    fz = (max_z + min_z) / 2.0
        
    offset_x = -(fx - l/2.0)
    offset_z = -(fz - w/2.0)
    
    for obj in objects_raw:
        min_y = max(0.0, obj['position']['y'] - obj['size']['y'] / 2.0)
        if min_y >= 1.0: 
            continue # ignore high ceiling objects for the floor plot
            
        cx = obj['position']['x'] + offset_x
        cz = obj['position']['z'] + offset_z
        obj_w, obj_d = obj['size']['x'], obj['size']['z']
        theta = obj['rotation']['y']
        
        # AI2-THOR / Unity uses a left-handed coordinate system.
        hw, hd = obj_w / 2.0, obj_d / 2.0
        # The furniture walls in AI2-THOR were generated explicitly using CCW math in convert_echoscene_to_procthor.py
        # So we MUST use exact CCW math here so the visual polygons align with the physics mesh
        rad = theta
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        
        def rot(lx, lz):
            return cx + lx*cos_t - lz*sin_t, cz + lx*sin_t + lz*cos_t
            
        corners = [rot(-hw, -hd), rot(hw, -hd), rot(hw, hd), rot(-hw, hd)]
        ax.add_patch(patches.Polygon(corners, closed=True, fill=True, color='brown', alpha=0.5))
        
    # Also plot the valid positions if desired
    if valid_nodes:
        x_vals = [n[0] for n in valid_nodes]
        z_vals = [n[1] for n in valid_nodes]
        ax.scatter(x_vals, z_vals, color='gray', s=5, alpha=0.5, zorder=1, label='Reachable Area (NavMesh)')
        
    # Plot path
    if path_points:
        px = [p['x'] for p in path_points]
        pz = [p['z'] for p in path_points]
        ax.plot(px, pz, c='blue', linewidth=2, label='Path')
        
    # Plot Start and Goal
    ax.scatter([start_pos['x']], [start_pos['z']], c='green', marker='*', s=200, label='Start')
    ax.scatter([goal_pos['x']], [goal_pos['z']], c='red', marker='X', s=150, label='Goal')
    
    plt.title(f"{scene_name} - {folder}")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend(loc='upper right')
    
    out_img = os.path.join(output_dir, f"{scene_name}_{folder}.png")
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    
    # Also copy to the original requested folder as per user request
    user_dir = f"../output/{folder}/vis/2050"
    os.system(f"cp {out_img} {user_dir}/{scene_name}_path.png")
    print(f"Saved plot to {out_img} and {user_dir}/{scene_name}_path.png")

controller.stop()
