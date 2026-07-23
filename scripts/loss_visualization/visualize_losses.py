import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import seaborn as sns
import glob

def get_obb_corners(x, z, l, w, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    dx = l / 2
    dz = w / 2
    
    corners = [
        (-dx, -dz),
        (dx, -dz),
        (dx, dz),
        (-dx, dz)
    ]
    
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
        matched_files = glob.glob(pattern)
        
        if len(matched_files) > 0:
            query_label = os.path.basename(matched_files[0]).split('_')[0]
            classes[cat_id] = query_label
            
        instance_id += 1
        
    return classes

def load_scene(file_path, scene_id, mesh_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    scene_ids = data["scene_ids"]
    if scene_id not in scene_ids:
        return None
        
    scene_idx = scene_ids.index(scene_id)
    class_labels = np.array(data["class_labels"][scene_idx])
    translations = np.array(data["translations"][scene_idx])
    sizes = np.array(data["sizes"][scene_idx])
    angles_rad = np.array(data["angles"][scene_idx])
    angles_deg = angles_rad * 180.0 / np.pi
    
    classes = get_classes_from_mesh_dir(mesh_dir, scene_id, data, scene_idx)
    num_classes = class_labels.shape[1]
    color_palette = np.array(sns.color_palette('hls', num_classes))
    
    cats = np.argmax(class_labels, axis=1)
    objects = []
    
    for j in range(len(cats)):
        cat_id = cats[j]
        class_name = classes[cat_id] if cat_id < len(classes) else f"Obj_{cat_id}"
        
        is_object = data["objectness"][scene_idx][j][0] == 1.0
        l, h, w = sizes[j]
        
        if not is_object:
            if l > 1.0 and w > 1.0:
                class_name = "floor"
            else:
                continue
                
        if "lamp" in class_name.lower():
            continue
            
        px, py, pz = translations[j]
        angle = angles_deg[j].item()
        
        objects.append({
            "name": class_name,
            "x": px,
            "y": py,
            "z": pz,
            "l": l,
            "h": h,
            "w": w,
            "angle": angle,
            "color": color_palette[cat_id],
            "corners": get_obb_corners(px, pz, l, w, angle)
        })
    return objects

def get_scene_bounds(objects):
    all_x = []
    all_z = []
    for obj in objects:
        all_x.extend(obj["corners"][:, 0])
        all_z.extend(obj["corners"][:, 1])
    if not all_x:
        return -3.5, 3.5, -3.5, 3.5
    return min(all_x), max(all_x), min(all_z), max(all_z)

def setup_plot(ax, title, bounds):
    ax.clear()
    pad = 0.5
    ax.set_xlim(bounds[0] - pad, bounds[1] + pad)
    ax.set_ylim(bounds[2] - pad, bounds[3] + pad)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis('off')

def plot_base_objects(ax, objects, fade_floor=True):
    for obj in objects:
        if obj["name"] == "floor":
            # Just draw the floor bounds in grey
            alpha = 0.15 if fade_floor else 0.4
            poly = patches.Polygon(obj["corners"], closed=True, facecolor="#cccccc", edgecolor='none', alpha=alpha)
            ax.add_patch(poly)
            
            # Floor boundary
            boundary = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--')
            ax.add_patch(boundary)
        else:
            poly = patches.Polygon(obj["corners"], closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.5, linewidth=1.0)
            ax.add_patch(poly)

def visualize_outer_loss(objects, bounds, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Room Outer Loss\n(Penalizes objects extending past room boundaries)", bounds)
    
    plot_base_objects(ax, objects, fade_floor=False)
    
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        # In the original loss, they use AABB of the floor or AABB of objects against floor.
        # Here we use exact shapely intersections for accurate visualization.
        floor_poly = Polygon(floor_obj["corners"])
        
        for obj in objects:
            if obj["name"] == "floor": continue
            obj_poly = Polygon(obj["corners"])
            
            if not floor_poly.contains(obj_poly):
                try:
                    diff = obj_poly.difference(floor_poly)
                    if not diff.is_empty:
                        # Calculate distance penalty (L1 distance of center to boundary)
                        obj_center = np.array([obj["x"], obj["z"]])
                        bounds = floor_poly.bounds  # minx, miny, maxx, maxy
                        
                        dx = max(bounds[0] - obj_center[0], 0, obj_center[0] - bounds[2])
                        dy = max(bounds[1] - obj_center[1], 0, obj_center[1] - bounds[3])
                        dist = dx + dy
                        
                        # Scale alpha based on distance (cap at 0.9)
                        alpha_val = min(max(dist * 0.5, 0.3), 0.95)
                        
                        if diff.geom_type == 'Polygon':
                            geoms = [diff]
                        else:
                            geoms = diff.geoms
                        for g in geoms:
                            x, y = g.exterior.xy
                            ax.fill(x, y, color='red', alpha=alpha_val, zorder=10)
                            ax.plot(x, y, color='darkred', linewidth=1.5, zorder=10)
                except:
                    pass
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def visualize_collision_loss(objects, bounds, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Collision Loss\n(Penalizes overlaps between objects)", bounds)
    
    plot_base_objects(ax, objects, fade_floor=True)
    
    # Calculate intersections between all pairs of objects (excluding floor)
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    for i in range(len(furnitures)):
        poly1 = Polygon(furnitures[i]["corners"])
        for j in range(i + 1, len(furnitures)):
            poly2 = Polygon(furnitures[j]["corners"])
            if poly1.intersects(poly2):
                try:
                    intersection = poly1.intersection(poly2)
                    if not intersection.is_empty:
                        # Calculate IoU for weight
                        iou = intersection.area / (poly1.area + poly2.area - intersection.area)
                        
                        # Scale alpha based on IoU (cap at 0.95)
                        alpha_val = min(max(iou * 3.0, 0.3), 0.95)
                        
                        if intersection.geom_type == 'Polygon':
                            geoms = [intersection]
                        else:
                            geoms = intersection.geoms
                        for g in geoms:
                            x, y = g.exterior.xy
                            ax.fill(x, y, color='red', alpha=alpha_val, zorder=10)
                            ax.plot(x, y, color='darkred', linewidth=1.5, zorder=10)
                except:
                    pass
                    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def visualize_walkable_loss(objects, bounds, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Walkable (Reachability) Loss\n(Penalizes objects placed near the center (0,0))", bounds)
    
    # Draw heatmap background
    pad = 0.5
    x = np.linspace(bounds[0] - pad, bounds[1] + pad, 500)
    z = np.linspace(bounds[2] - pad, bounds[3] + pad, 500)
    xx, zz = np.meshgrid(x, z)
    dist_sq = xx**2 + zz**2
    sigma = 0.5
    walk_penalty = np.exp(-dist_sq / sigma)
    
    extent = [bounds[0] - pad, bounds[1] + pad, bounds[2] - pad, bounds[3] + pad]
    im = ax.imshow(walk_penalty, extent=extent, origin='lower', cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    
    # Plot origin marker
    ax.plot(0, 0, marker='x', color='darkred', markersize=10, markeredgewidth=2, zorder=5)
    
    # Plot objects on top
    for obj in objects:
        if obj["name"] == "floor": continue
        
        # Calculate penalty for this specific object
        dist_sq_obj = obj["x"]**2 + obj["z"]**2
        penalty = np.exp(-dist_sq_obj / sigma)
        
        # Color the object face itself red if it has high penalty
        # Mix the original color with red based on penalty weight
        orig_color = np.array(obj["color"])
        red_color = np.array([1.0, 0.0, 0.0])
        blended_color = orig_color * (1 - penalty) + red_color * penalty
        
        poly = patches.Polygon(obj["corners"], closed=True, facecolor=blended_color, edgecolor='black', alpha=0.9, linewidth=1.0)
        ax.add_patch(poly)
        
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--old_mesh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    objects = load_scene(args.json, args.scene_id, args.old_mesh_dir)
    if objects is None:
        print(f"Scene {args.scene_id} not found in {args.json}")
        return
        
    bounds = get_scene_bounds(objects)
    
    print(f"Generating Outer Loss Visualization...")
    visualize_outer_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_outer_loss.png"))
    
    print(f"Generating Collision Loss Visualization...")
    visualize_collision_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_collision_loss.png"))
    
    print(f"Generating Walkable Loss Visualization...")
    visualize_walkable_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss.png"))
    
    print(f"Done! Check the {args.out_dir} directory.")

if __name__ == "__main__":
    main()
