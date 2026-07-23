import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def get_obb_corners(x, z, l, w, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append([x + rx, z + rz])
    return np.array(rotated)

def get_classes_from_mesh_dir(mesh_dir, scene_id, data, scene_idx):
    class_labels = np.array(data["class_labels"][scene_idx])
    num_classes = class_labels.shape[1]
    classes = [f"Obj_{i}" for i in range(num_classes)]
    
    scene_mesh_dir = os.path.join(mesh_dir, scene_id)
    if not os.path.exists(scene_mesh_dir):
        return classes
        
    cats = np.argmax(class_labels, axis=1)
    instance_id = 1
    for j in range(len(cats)):
        cat_id = cats[j]
        is_object = data["objectness"][scene_idx][j][0] == 1.0
        
        if not is_object:
            continue
            
        pattern = os.path.join(scene_mesh_dir, f"*_{cat_id}_{instance_id}.obj")
        import glob
        matched_files = glob.glob(pattern)
        
        if len(matched_files) > 0:
            query_label = os.path.basename(matched_files[0]).split('_')[0]
            classes[cat_id] = query_label
            
        instance_id += 1
        
    return classes

def load_scene_from_json(file_path, scene_id, mesh_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    if scene_id not in data["scene_ids"]:
        return None
        
    scene_idx = data["scene_ids"].index(scene_id)
    class_labels = np.array(data["class_labels"][scene_idx])
    cats = np.argmax(class_labels, axis=1)
    translations = np.array(data["translations"][scene_idx])
    sizes = np.array(data["sizes"][scene_idx])
    angles_rad = np.array(data["angles"][scene_idx])
    angles_deg = angles_rad * 180.0 / np.pi
    
    classes = get_classes_from_mesh_dir(mesh_dir, scene_id, data, scene_idx)
    num_classes = class_labels.shape[1]
    color_palette = np.array(sns.color_palette('hls', num_classes))
    
    objects = []
    for j in range(len(cats)):
        cat_id = cats[j]
        class_name = classes[cat_id] if cat_id < len(classes) else f"Obj_{cat_id}"
        
        is_object = data["objectness"][scene_idx][j][0] == 1.0
        l, h, w = sizes[j]
        
        if not is_object:
            # If it's a dummy object with large dimensions, it's the floor
            if l > 1.0 and w > 1.0:
                class_name = "floor"
            else:
                continue
                
        # Skip lamps since they aren't involved in collisions
        if "lamp" in class_name.lower():
            continue
            
        px, py, pz = translations[j]
        angle = angles_deg[j].item()
        
        color = color_palette[cat_id]
        hex_color = '#%02x%02x%02x' % tuple((color * 255).astype(int))
        
        objects.append({
            "name": class_name,
            "cat_id": cat_id,
            "l": l, "w": w, "x": px, "z": pz, "angle": angle,
            "color": hex_color
        })
    return objects

def plot_scene(ax, objects, title, limits):
    ax.clear()
    for obj in objects:
        obb_corners = get_obb_corners(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
        
        min_x, min_z = np.min(obb_corners, axis=0)
        max_x, max_z = np.max(obb_corners, axis=0)
        
        alpha = 0.25 if obj["name"] == "floor" else 0.7
        obb_polygon = patches.Polygon(
            obb_corners, closed=True, facecolor=obj["color"] if obj["name"] != "floor" else "#cccccc", 
            edgecolor='black' if obj["name"] != "floor" else "none", alpha=alpha, linewidth=1.5
        )
        ax.add_patch(obb_polygon)
        
        if obj["name"] != "floor":
            # Draw AABB (Dashed Red Line)
            aabb_rect = patches.Rectangle(
                (min_x, min_z), max_x - min_x, max_z - min_z, 
                linewidth=1, edgecolor='#e74c3c', facecolor='none', 
                linestyle='--', alpha=0.6
            )
            ax.add_patch(aabb_rect)
            
            ax.text(obj["x"], obj["z"], obj["name"], ha='center', va='center', 
                    fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_aspect('equal')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def get_scene_bounds(objects):
    all_x = []
    all_z = []
    for obj in objects:
        obb_corners = get_obb_corners(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
        all_x.extend(obb_corners[:, 0])
        all_z.extend(obb_corners[:, 1])
    if not all_x:
        return -3.5, 3.5, -3.5, 3.5
    return min(all_x), max(all_x), min(all_z), max(all_z)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True, help="Scene ID or 'all' to process all scenes")
    parser.add_argument("--json1", required=True)
    parser.add_argument("--json2", required=True)
    parser.add_argument("--old_mesh_dir", required=True, help="Directory containing original .obj files to extract class names")
    parser.add_argument("--out", default="compare_vis.png", help="Output file if processing a single scene")
    parser.add_argument("--out_dir", default="compare", help="Output directory if processing all scenes")
    args = parser.parse_args()

    with open(args.json1, 'r') as f:
        data1 = json.load(f)
    
    scene_ids = data1["scene_ids"]
    if args.scene_id != "all":
        if args.scene_id not in scene_ids:
            print(f"Scene {args.scene_id} not found in {args.json1}")
            return
        scene_ids = [args.scene_id]
    else:
        os.makedirs(args.out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, s_id in enumerate(scene_ids):
        print(f"Processing {s_id} ({i+1}/{len(scene_ids)})...")
        objects1 = load_scene_from_json(args.json1, s_id, args.old_mesh_dir)
        objects2 = load_scene_from_json(args.json2, s_id, args.old_mesh_dir)

        if objects1 is None or objects2 is None:
            print(f"Skipping {s_id} because it's missing from one of the JSONs.")
            continue
            
        bounds1 = get_scene_bounds(objects1)
        bounds2 = get_scene_bounds(objects2)
        
        min_x = min(bounds1[0], bounds2[0]) - 0.5
        max_x = max(bounds1[1], bounds2[1]) + 0.5
        min_z = min(bounds1[2], bounds2[2]) - 0.5
        max_z = max(bounds1[3], bounds2[3]) + 0.5
        limits = (min_x, max_x, min_z, max_z)
        
        plot_scene(ax1, objects1, f"Input:\n{args.json1.split('/')[-3]}/{os.path.basename(args.json1)}", limits)
        plot_scene(ax2, objects2, f"Resolved:\n{args.json2.split('/')[-3]}/{os.path.basename(args.json2)}", limits)
        
        plt.tight_layout()
        
        if args.scene_id == "all":
            out_path = os.path.join(args.out_dir, f"{s_id}_compare.png")
        else:
            out_path = args.out
            
        plt.savefig(out_path, dpi=150)
        
    print("Done!")

if __name__ == "__main__":
    main()
