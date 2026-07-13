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
            
    # Get shortest path
    path_event = controller.step(action="GetShortestPath", position=goal_pos)
    path_points = []
    if path_event.metadata["lastActionSuccess"]:
        path_points = path_event.metadata["actionReturn"]["corners"]
    else:
        print(f"{folder}: Failed to get shortest path. Using straight line.")
        path_points = [start_pos, goal_pos]
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, l)
    ax.set_ylim(0, w)
    ax.set_aspect('equal')
    
    # Plot room bounds
    ax.add_patch(patches.Rectangle((0, 0), l, w, fill=False, edgecolor='black', linewidth=3))
    
    # Plot furniture
    for box in furniture_boxes:
        min_x, max_x, min_z, max_z = box
        ax.add_patch(patches.Rectangle((min_x, min_z), max_x - min_x, max_z - min_z, fill=True, color='brown', alpha=0.5))
        
    # Plot reachable points
    if valid_positions:
        rx = [p['x'] for p in valid_positions]
        rz = [p['z'] for p in valid_positions]
        ax.scatter(rx, rz, c='lightgray', s=10, label='Reachable Area (NavMesh)')
        
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
