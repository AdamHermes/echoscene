import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import distance_transform_edt

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

import argparse
parser = argparse.ArgumentParser(description="Visualize path planning on EchoScene layouts.")
parser.add_argument("--scene_name", type=str, required=True, help="Name of the scene (e.g. SecondBedroom-6482)")
parser.add_argument("--folders", type=str, nargs='+', required=True, help="List of folder names to process (e.g. baseline physcene_guidance)")
args = parser.parse_args()

scene_name = args.scene_name
folders = args.folders

controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)

output_dir = "/Users/lehoangan/.gemini/antigravity-cli/brain/c21352c1-abf1-4f85-aaae-e44c32e9145d/scratch"
os.makedirs(output_dir, exist_ok=True)

for folder in folders:
    json_path = f"./output/{folder}/vis/2050/procthor_scenes/{scene_name}.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        house_data = json.load(f)
        
    rd = house_data.get('room_dims', {})
    l = rd.get('l', 0)
    w = rd.get('w', 0)
    controller.reset(scene=house_data)
    
    res = 0.05
    grid_w = int(math.ceil(l / res))
    grid_h = int(math.ceil(w / res))
    occupancy_grid = np.ones((grid_w, grid_h))
    
    occupancy_grid[0, :] = 0
    occupancy_grid[-1, :] = 0
    occupancy_grid[:, 0] = 0
    occupancy_grid[:, -1] = 0
    
    import matplotlib.path as mpltPath
    X_centers, Z_centers = np.meshgrid(np.arange(grid_w) * res + res/2, np.arange(grid_h) * res + res/2, indexing='ij')
    grid_points = np.column_stack((X_centers.flatten(), Z_centers.flatten()))
    
    objects_raw = house_data.get('objects_raw', [])
    if objects_raw:
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
                continue
                
            cx = obj['position']['x'] + offset_x
            cz = obj['position']['z'] + offset_z
            hw = obj['size']['x'] / 2.0
            hd = obj['size']['z'] / 2.0
            
            theta = obj['rotation']['y']
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            def rot(lx, lz):
                return cx + lx*cos_t - lz*sin_t, cz + lx*sin_t + lz*cos_t
                
            corners = [rot(-hw, -hd), rot(hw, -hd), rot(hw, hd), rot(-hw, hd)]
            path = mpltPath.Path(corners)
            # Use radius=res to expand the collision check, guaranteeing any partially covered cell is marked blue
            mask = path.contains_points(grid_points, radius=res).reshape(grid_w, grid_h)
            occupancy_grid[mask] = 0
        
    distances = distance_transform_edt(occupancy_grid)
    
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)
    spawn_x = max_idx[0] * res
    spawn_z = max_idx[1] * res
    
    # Snap to the 0.25m grid to keep AI2-THOR pathfinding perfectly aligned
    spawn_x = round(spawn_x / 0.25) * 0.25
    spawn_z = round(spawn_z / 0.25) * 0.25
    
    print(f"[{folder}] Spawn chosen by EDT at ({spawn_x:.2f}, {spawn_z:.2f}) with clearance {distances[max_idx]*res:.2f}m")
        
    controller.step(action="Teleport", position={"x": spawn_x, "y": 0.9, "z": spawn_z}, forceAction=True)
    
    # Get all reachable points
    rp_event = controller.step(action="GetReachablePositions")
    valid_positions = []
    if rp_event.metadata["lastActionSuccess"]:
        reachable_positions = rp_event.metadata["actionReturn"]
        valid_positions = [p for p in reachable_positions if 0 <= p['x'] <= l and 0 <= p['z'] <= w]
        
    if not valid_positions:
        print(f"{folder}: No reachable positions found.")
        continue
        
    # Find the furthest point from start as goal
    start_pos = {"x": spawn_x, "y": 0.9, "z": spawn_z}
    goal_pos = start_pos
    max_dist = -1
    for p in valid_positions:
        dist = math.dist([start_pos['x'], start_pos['z']], [p['x'], p['z']])
        if dist > max_dist:
            max_dist = dist
            goal_pos = p
            
    # ---------------------------------------------------------------------------------
    # NATIVE AI2-THOR PATHFINDING
    # Since convert_echoscene_to_procthor.py completely seals the furniture to the ceiling
    # with horizontal lids, the AI2-THOR NavMesh baker cannot leak voxels inside the furniture.
    # Therefore, GetShortestPathToPoint now returns a mathematically perfect path that
    # correctly respects the agent's 0.25m collision radius without cutting corners.
    # ---------------------------------------------------------------------------------
    path_event = controller.step(action="GetShortestPathToPoint", target=goal_pos)
    path_points = []
    if path_event.metadata["lastActionSuccess"]:
        path_points = path_event.metadata["actionReturn"]["corners"]
    else:
        print(f"{folder}: Failed to get shortest path. Using straight line.")
        path_points = [start_pos, goal_pos]
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, l)
    ax.set_ylim(0, w)
    ax.set_aspect('equal')
    
    # Plot room bounds
    ax.add_patch(patches.Rectangle((0, 0), l, w, fill=False, edgecolor='black', linewidth=3))
    
    from matplotlib.colors import ListedColormap
    # Plot Occupancy Grid cells (0: occupied -> blue (opaque), 1: free -> green (transparent))
    # Colors defined as (R, G, B, Alpha)
    cmap = ListedColormap([(0.0, 0.0, 1.0, 0.6), (0.0, 1.0, 0.0, 0.15)])
    X_grid, Z_grid = np.meshgrid(np.arange(grid_w + 1) * res, np.arange(grid_h + 1) * res)
    ax.pcolormesh(X_grid, Z_grid, occupancy_grid.T, cmap=cmap, vmin=0, vmax=1, edgecolors='none')
    
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
    if valid_positions:
        # Draw the 0.25m x 0.25m Walkable Area cells
        for p in valid_positions:
            rect = patches.Rectangle((p['x'] - 0.125, p['z'] - 0.125), 0.25, 0.25, linewidth=0, facecolor='yellow', alpha=0.5, zorder=2)
            ax.add_patch(rect)
            
        rx = [p['x'] for p in valid_positions]
        rz = [p['z'] for p in valid_positions]
        ax.scatter(rx, rz, c='black', s=10, label='NavMesh Nodes', zorder=3)
        
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
    
    user_dir = f"./output/{folder}/vis/2050"
    plot_path = os.path.join(user_dir, f"{scene_name}_path.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")

controller.stop()
