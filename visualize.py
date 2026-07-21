import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_obb_corners(x, z, l, w, angle_rad):
    """Calculates the 4 corners of the Oriented Bounding Box (OBB)."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append([x + rx, z + rz])
    return np.array(rotated)

def main():
    parser = argparse.ArgumentParser(description="Visualize a scene from a physcene input JSON.")
    parser.add_argument("--json", type=str, default="physcene_collision_input_all.json",
                        help="Path to the JSON file.")
    # Support both `--scene` and `--Library-2159` style passing
    parser.add_argument("scene", type=str, nargs='?', default=None, help="Scene ID to visualize, e.g., Library-2159")
    
    # Alternatively parse unknown args as scene
    args, unknown = parser.parse_known_args()
    
    scene_id = args.scene
    if scene_id is None:
        for unk in unknown:
            if unk.startswith("--"):
                scene_id = unk[2:]
                break
            else:
                scene_id = unk
                break
                
    if scene_id is None:
        print("Please provide a scene ID, e.g. python visualize.py --Library-2159")
        return
        
    if not os.path.exists(args.json):
        # Fallback to check other potential files
        fallback = "physcene_collision_input_all.json"
        if os.path.exists(fallback):
            print(f"File {args.json} not found. Using {fallback} instead.")
            args.json = fallback
        else:
            print(f"File {args.json} not found.")
            return
            
    print(f"Loading data from {args.json}...")
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    scene_ids = data.get("scene_ids", [])
    if scene_id not in scene_ids:
        print(f"Scene ID {scene_id} not found in {args.json}.")
        return
        
    idx = scene_ids.index(scene_id)
    print(f"Found scene {scene_id} at index {idx}.")
    
    class_labels = data.get("class_labels", [])[idx]
    translations = data.get("translations", [])[idx]
    sizes = data.get("sizes", [])[idx]
    angles = data.get("angles", [])[idx]
    
    # Optional: filter out by objectness if it exists
    objectness = data.get("objectness", [])
    if objectness:
        obj_mask = objectness[idx]
    else:
        obj_mask = [[1.0]] * len(class_labels)
        
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Generate distinct colors for classes
    cmap = plt.colormaps.get_cmap('tab20')
    
    all_xs = []
    all_zs = []
    
    for i in range(len(class_labels)):
        # Parse data
        cls_idx = int(np.argmax(class_labels[i]))
        is_layout = (obj_mask[i][0] < 0.5) or (cls_idx == 14) # Usually floor/wall has objectness 0 or is the last class
        
        # translations: [x, y, z] -> we need x and z
        x = translations[i][0]
        z = translations[i][2]
        
        # sizes: [l, h, w] -> we need l and w
        l = sizes[i][0]
        w = sizes[i][2]
        
        # Ignore outlier/scene anchor (extremely small size + layout class)
        if is_layout and l < 0.15 and w < 0.15:
            continue
            
        name = "Floor/Layout" if is_layout else f"Class_{cls_idx}"
        color = "#bdc3c7" if is_layout else cmap(cls_idx % 20)
        alpha = 0.25 if is_layout else 0.7
        zorder = 0 if is_layout else 10
        
        # angle: radians
        angle_rad = angles[i][0]
        
        obb_corners = get_obb_corners(x, z, l, w, angle_rad)
        min_x, min_z = np.min(obb_corners, axis=0)
        max_x, max_z = np.max(obb_corners, axis=0)
        
        all_xs.extend(obb_corners[:, 0])
        all_zs.extend(obb_corners[:, 1])
        
        # Draw OBB (Oriented Bounding Box)
        obb_polygon = patches.Polygon(
            obb_corners, closed=True, facecolor=color, 
            edgecolor='black', alpha=alpha, linewidth=1.5, zorder=zorder
        )
        ax.add_patch(obb_polygon)
        
        # Label
        if not is_layout:
            ax.text(x, z, name, ha='center', va='center', 
                    fontsize=9, weight='bold', zorder=zorder+1,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        else:
            # For layout/floor, draw label at the corner to not obstruct
            ax.text(min_x, min_z, name, ha='left', va='bottom', 
                    fontsize=8, color='#555555', zorder=zorder+1)
                
    # View settings
    ax.set_aspect('equal')
    # Auto-scale limits based on object corners
    if len(all_xs) > 0:
        margin = 0.5
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_zs) - margin, max(all_zs) + margin)
    else:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        
    ax.grid(True, linestyle=':', alpha=0.5)
    
    title = f"Scene: {scene_id} from {args.json}"
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    
    plt.show()

if __name__ == "__main__":
    main()
