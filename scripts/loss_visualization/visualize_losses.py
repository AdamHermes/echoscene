import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import seaborn as sns
import glob
import cv2

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
            "id": j + 1,
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

def load_triples(scene_id, custom_json=None):
    if custom_json and os.path.exists(custom_json):
        try:
            with open(custom_json, 'r') as f:
                d = json.load(f)
                if 'scans' in d:
                    for scan in d['scans']:
                        if scan['scan'] == scene_id:
                            return scan['relationships'], scan.get('objects', {})
        except Exception as e:
            print(f"Error loading custom relational json: {e}")
            
    front_files = glob.glob('FRONT/relationships_*.json')
    for fp in front_files:
        try:
            with open(fp, 'r') as f:
                d = json.load(f)
                for scan in d.get('scans', []):
                    if scan['scan'] == scene_id:
                        return scan['relationships'], scan.get('objects', {})
        except Exception:
            pass
    return [], {}

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
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

def plot_base_objects(ax, objects, fade_floor=True, show_labels=False):
    for obj in objects:
        if obj["name"] == "floor":
            alpha = 0.15 if fade_floor else 0.4
            poly = patches.Polygon(obj["corners"], closed=True, facecolor="#e8e8e8", edgecolor='none', alpha=alpha, zorder=2)
            ax.add_patch(poly)
            
            boundary = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', zorder=9)
            ax.add_patch(boundary)
        else:
            poly = patches.Polygon(obj["corners"], closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.55, linewidth=1.0, zorder=3)
            ax.add_patch(poly)
            if show_labels:
                ax.text(obj["x"], obj["z"], f"#{obj['id']} {obj['name']}", fontsize=8, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.85), zorder=15)

def visualize_outer_loss(objects, bounds, out_path, show_text=False):
    """Visualize Room Outer Loss: penalizes objects extending past room floor boundaries."""
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Room Outer Loss\n(Penalizes objects extending past room boundaries)" if show_text else None, bounds)
    
    plot_base_objects(ax, objects, fade_floor=False, show_labels=False)
    
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        floor_poly = Polygon(floor_obj["corners"])
        
        for obj in objects:
            if obj["name"] == "floor": continue
            obj_poly = Polygon(obj["corners"])
            
            if not floor_poly.contains(obj_poly):
                try:
                    diff = obj_poly.difference(floor_poly)
                    if not diff.is_empty:
                        obj_center = np.array([obj["x"], obj["z"]])
                        fbounds = floor_poly.bounds
                        
                        dx = max(fbounds[0] - obj_center[0], 0, obj_center[0] - fbounds[2])
                        dy = max(fbounds[1] - obj_center[1], 0, obj_center[1] - fbounds[3])
                        dist = dx + dy
                        
                        alpha_val = min(max(dist * 0.5, 0.3), 0.95)
                        
                        if diff.geom_type == 'Polygon':
                            geoms = [diff]
                        else:
                            geoms = diff.geoms
                        for g in geoms:
                            x, y = g.exterior.xy
                            ax.fill(x, y, color='red', alpha=alpha_val, zorder=10)
                            ax.plot(x, y, color='darkred', linewidth=1.5, zorder=10)
                except Exception:
                    pass
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_collision_loss(objects, bounds, out_path, show_text=False):
    """Visualize Collision Loss: penalizes overlaps between objects."""
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Collision Loss\n(Penalizes overlaps between objects)" if show_text else None, bounds)
    
    plot_base_objects(ax, objects, fade_floor=True, show_labels=False)
    
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    for i in range(len(furnitures)):
        poly1 = Polygon(furnitures[i]["corners"])
        for j in range(i + 1, len(furnitures)):
            poly2 = Polygon(furnitures[j]["corners"])
            if poly1.intersects(poly2):
                try:
                    intersection = poly1.intersection(poly2)
                    if not intersection.is_empty:
                        iou = intersection.area / (poly1.area + poly2.area - intersection.area)
                        alpha_val = min(max(iou * 3.0, 0.3), 0.95)
                        
                        if intersection.geom_type == 'Polygon':
                            geoms = [intersection]
                        else:
                            geoms = intersection.geoms
                        for g in geoms:
                            x, y = g.exterior.xy
                            ax.fill(x, y, color='red', alpha=alpha_val, zorder=10)
                            ax.plot(x, y, color='darkred', linewidth=1.5, zorder=10)
                except Exception:
                    pass
                    
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_center_penalty_loss(objects, bounds, out_path, sigma=0.5, show_text=False):
    """
    Visualize Center Penalty Walkability Loss as a pure, clean Radial Gaussian density field.
    No connecting lines between dots/objects, no concentric circles, no text, no scale bar.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Center Penalty Walkable Loss\n(Radial Gaussian penalty centered at room origin)" if show_text else None, bounds)
    plot_base_objects(ax, objects, fade_floor=False, show_labels=False)
    
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    res = 400
    pad = 0.5
    grid_x = np.linspace(bounds[0] - pad, bounds[1] + pad, res)
    grid_z = np.linspace(bounds[2] - pad, bounds[3] + pad, res)
    GXX, GZZ = np.meshgrid(grid_x, grid_z)
    dist_sq = GXX**2 + GZZ**2
    center_heat = np.exp(-dist_sq / sigma)
    
    extent = [bounds[0] - pad, bounds[1] + pad, bounds[2] - pad, bounds[3] + pad]
    ax.imshow(center_heat, extent=extent, origin='lower', aspect='equal', cmap='plasma', alpha=0.65, zorder=5)
    
    # Overlay object outlines on top for crisp visual boundaries
    for obj in furnitures:
        poly_outline = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='#222222', linewidth=1.2, zorder=8)
        ax.add_patch(poly_outline)
        
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        boundary = patches.Polygon(floor_obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.8, linestyle='--', zorder=9)
        ax.add_patch(boundary)
        
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_edge_gaussian_loss(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5, sigma_scale=0.5, show_text=False):
    """
    Visualize Exact OBB Edge-Gaussian Walkability Loss as a pure, clean 2D Gaussian density field.
    Continuous Edge-Gaussian field radiating from rotated object boundaries across the floor.
    No connecting lines between dots/objects, no circles/rings, no text, no scale bar.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Edge-Gaussian Walkable Loss\n(Continuous Gaussian field radiating from object boundaries)" if show_text else None, bounds)
    plot_base_objects(ax, objects, fade_floor=False, show_labels=False)
    
    furnitures = [o for o in objects if o["name"] != "floor" and o["y"] < robot_hight_real]
    
    res = 400
    pad = 0.5
    gx = np.linspace(bounds[0] - pad, bounds[1] + pad, res)
    gz = np.linspace(bounds[2] - pad, bounds[3] + pad, res)
    GXX, GZZ = np.meshgrid(gx, gz)
    
    combined_heatmap = np.zeros_like(GXX)
    
    for obj in furnitures:
        cx, cz = obj["x"], obj["z"]
        l, w = obj["l"], obj["w"]
        angle_rad = np.deg2rad(obj["angle"])
        
        rel_x = GXX - cx
        rel_z = GZZ - cz
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        local_x =  rel_x * cos_a + rel_z * sin_a
        local_z = -rel_x * sin_a + rel_z * cos_a
        
        dx = np.maximum(np.abs(local_x) - l / 2.0, 0.0)
        dz = np.maximum(np.abs(local_z) - w / 2.0, 0.0)
        d_edge = np.sqrt(dx**2 + dz**2)
        
        sigma = (l + w) / 2.0 * sigma_scale
        g_field = np.exp(-d_edge**2 / (2.0 * sigma**2))
        combined_heatmap += g_field

    extent = [bounds[0] - pad, bounds[1] + pad, bounds[2] - pad, bounds[3] + pad]
    ax.imshow(combined_heatmap, extent=extent, origin='lower', aspect='equal', cmap='plasma', alpha=0.65, zorder=5)
    
    # Overlay object outlines on top for crisp visual boundaries
    for obj in furnitures:
        poly_outline = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='#222222', linewidth=1.2, zorder=8)
        ax.add_patch(poly_outline)
        
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        boundary = patches.Polygon(floor_obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.8, linestyle='--', zorder=9)
        ax.add_patch(boundary)

    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_object_gaussian_loss(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5, sigma_scale=1.0, show_text=False):
    """
    Visualize Object-Centered 2D Gaussian Walkability Loss (gausv1).
    Each ground-level object emits an oriented 2D Gaussian centered at its bounding box center.
    No connecting lines between dots/objects, no circles/rings, no text, no scale bar.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Object Gaussian Walkable Loss\n(Oriented 2D Gaussian fields centered at each object)" if show_text else None, bounds)
    plot_base_objects(ax, objects, fade_floor=False, show_labels=False)
    
    furnitures = [o for o in objects if o["name"] != "floor" and o["y"] < robot_hight_real]
    
    res = 400
    pad = 0.5
    gx = np.linspace(bounds[0] - pad, bounds[1] + pad, res)
    gz = np.linspace(bounds[2] - pad, bounds[3] + pad, res)
    GXX, GZZ = np.meshgrid(gx, gz)
    
    combined_heatmap = np.zeros_like(GXX)
    
    for obj in furnitures:
        cx, cz = obj["x"], obj["z"]
        l, w = obj["l"], obj["w"]
        angle_rad = np.deg2rad(obj["angle"])
        
        rel_x = GXX - cx
        rel_z = GZZ - cz
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        local_x =  rel_x * cos_a + rel_z * sin_a
        local_z = -rel_x * sin_a + rel_z * cos_a
        
        # Oriented 2D Gaussian semi-axes matched to bounding box dimensions
        sigma_x = max(l / 2.0 * sigma_scale, 0.05)
        sigma_z = max(w / 2.0 * sigma_scale, 0.05)
        
        z_val = (local_x / sigma_x)**2 + (local_z / sigma_z)**2
        g_field = np.exp(-0.5 * z_val)
        combined_heatmap += g_field

    extent = [bounds[0] - pad, bounds[1] + pad, bounds[2] - pad, bounds[3] + pad]
    ax.imshow(combined_heatmap, extent=extent, origin='lower', aspect='equal', cmap='plasma', alpha=0.65, zorder=5)
    
    # Overlay object outlines on top for crisp visual boundaries
    for obj in furnitures:
        poly_outline = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='#222222', linewidth=1.2, zorder=8)
        ax.add_patch(poly_outline)
        
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        boundary = patches.Polygon(floor_obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.8, linestyle='--', zorder=9)
        ax.add_patch(boundary)

    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_pathfinding_loss(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5, show_text=False):
    """
    Visualize Pathfinding / Reachability Walkable Loss as a clean standalone 2D free-space map.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    if not floor_obj:
        print("Warning: No floor object found for pathfinding loss visualization.")
        return
        
    floor_corners = floor_obj["corners"]
    floor_cx, floor_cz = floor_obj["x"], floor_obj["z"]
    floor_corners_centered = floor_corners - np.array([floor_cx, floor_cz])
    
    all_pts_centered = [floor_corners_centered]
    for o in furnitures:
        all_pts_centered.append(o["corners"] - np.array([floor_cx, floor_cz]))
    all_pts = np.vstack(all_pts_centered)
    scale = np.abs(all_pts).max() + 0.2
    
    image_size = 512
    def map_to_image(point):
        x, y = point
        return int(x / scale * image_size / 2) + image_size // 2, int(y / scale * image_size / 2) + image_size // 2

    robot_width_px = max(1, int(robot_width_real / scale * image_size / 2))
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    floor_img_pts = np.array([map_to_image(v) for v in floor_corners_centered], np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [floor_img_pts], (255, 0, 0))
    
    kernel = np.ones((robot_width_px, robot_width_px), dtype=np.uint8)
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    
    for obj in furnitures:
        if obj["y"] > robot_hight_real: continue
        rel_x = obj["x"] - floor_cx
        rel_z = obj["z"] - floor_cz
        center = map_to_image((rel_x, rel_z))
        size_px = (max(1, int(obj["l"] / scale * image_size / 2)),
                   max(1, int(obj["w"] / scale * image_size / 2)))
        angle_deg = -obj["angle"]
        box_points = cv2.boxPoints(((center[0], center[1]), size_px, angle_deg))
        box_points = np.intp(box_points)
        cv2.drawContours(image, [box_points], 0, (0, 255, 0), robot_width_px)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    floor_mask = image[:, :, 0] == 255
    obj_mask = image[:, :, 1] == 255
    walkable = floor_mask & ~obj_mask
    blocked = floor_mask & obj_mask

    walkable_u8 = (walkable * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_u8, connectivity=8)

    vis = np.ones((image_size, image_size, 3), dtype=np.float32) * 0.95
    vis[floor_mask] = [0.88, 0.88, 0.88]
    component_colors = [[0.3, 0.8, 0.3], [0.3, 0.65, 0.85], [0.85, 0.85, 0.3], [0.7, 0.35, 0.85]]
    for label_id in range(1, num_labels):
        mask = labels == label_id
        color = component_colors[(label_id - 1) % len(component_colors)]
        vis[mask] = color
    vis[blocked] = [0.95, 0.45, 0.4]

    extent_c = [-scale, scale, -scale, scale]
    ax.imshow(vis, extent=extent_c, origin='lower', aspect='equal')
    
    # Overlay object outlines & floor
    for obj in furnitures:
        if obj["y"] > robot_hight_real: continue
        obj_corners_centered = obj["corners"] - np.array([floor_cx, floor_cz])
        poly = patches.Polygon(obj_corners_centered, closed=True, facecolor='none', edgecolor='black', linewidth=1.2, zorder=10)
        ax.add_patch(poly)
    floor_poly = patches.Polygon(floor_corners_centered, closed=True, facecolor='none', edgecolor='black', linewidth=1.8, linestyle='--', zorder=10)
    ax.add_patch(floor_poly)

    if show_text:
        total_floor_px = floor_mask.sum()
        walkable_px = walkable.sum()
        walkable_pct = 100.0 * walkable_px / max(total_floor_px, 1)
        ax.set_title(f"Pathfinding Walkable Loss\n(Walkable: {walkable_pct:.1f}% | Components: {num_labels - 1})", fontsize=12, fontweight='bold', pad=12)
        
    ax.set_xlim(-scale - 0.2, scale + 0.2)
    ax.set_ylim(-scale - 0.2, scale + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

def visualize_relational_loss(objects, bounds, out_path, triples=None, predicate_names=None, margin=0.05, close_threshold=0.45, stand_threshold=0.04, show_text=False):
    """
    Visualize Relational Guidance Loss as a clean standalone 2D diagram.
    Highlights violated spatial relationships with bold red/crimson constraint arrows and markers.
    No title text, no labels, no scale bar.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Relational Guidance Loss" if show_text else None, bounds)
    plot_base_objects(ax, objects, fade_floor=False, show_labels=show_text)
    
    obj_by_id = {o["id"]: o for o in objects}
    spatial_preds = {'left': 1, 'right': 2, 'front': 3, 'behind': 4, 'close by': 5, 'above': 6, 'standing on': 7}
    
    evaluated_list = []
    
    if triples:
        for edge in triples:
            if len(edge) < 4:
                s_idx, p_idx, o_idx = edge[0], edge[1], edge[2]
                p_name = str(p_idx)
            else:
                s_idx, o_idx, p_idx, p_name = edge[0], edge[1], edge[2], edge[3]
                
            if p_name not in spatial_preds:
                continue
                
            sub_obj = obj_by_id.get(s_idx)
            obj_obj = obj_by_id.get(o_idx)
            
            if not sub_obj or not obj_obj or sub_obj["id"] == obj_obj["id"]:
                continue
                
            xs, ys, zs = sub_obj["x"], sub_obj["y"], sub_obj["z"]
            xo, yo, zo = obj_obj["x"], obj_obj["y"], obj_obj["z"]
            
            loss = 0.0
            
            if p_name in ('left', '1'):
                val = zs - zo
                loss = max(0.0, val + margin)
            elif p_name in ('right', '2'):
                val = zs - zo
                loss = max(0.0, -val + margin)
            elif p_name in ('front', '3'):
                val = xs - xo
                loss = max(0.0, -val + margin)
            elif p_name in ('behind', '4'):
                val = xs - xo
                loss = max(0.0, val + margin)
            elif p_name in ('close by', '5'):
                dist = np.sqrt((xs - xo)**2 + (zs - zo)**2)
                loss = max(0.0, dist - close_threshold)
            elif p_name in ('standing on', 'above', '6', '7'):
                diff_y = abs(ys - yo)
                loss = max(0.0, diff_y - stand_threshold)

            if loss > 1e-4:
                evaluated_list.append({
                    's_id': sub_obj['id'], 's_name': sub_obj['name'],
                    'o_id': obj_obj['id'], 'o_name': obj_obj['name'],
                    'p_name': p_name, 'loss': loss,
                    'sub_pos': (xs, zs), 'obj_pos': (xo, zo)
                })

    # Highlight objects involved in violations with subtle red border/tint
    violated_obj_ids = set()
    for rel in evaluated_list:
        violated_obj_ids.add(rel['s_id'])
        violated_obj_ids.add(rel['o_id'])

    for obj in objects:
        if obj["id"] in violated_obj_ids and obj["name"] != "floor":
            poly_viol = patches.Polygon(obj["corners"], closed=True, facecolor='red', edgecolor='darkred', alpha=0.15, linewidth=1.5, zorder=6)
            ax.add_patch(poly_viol)

    drawn_pairs = set()
    for rel in evaluated_list:
        pair_key = (rel['s_id'], rel['o_id'], rel['p_name'])
        if pair_key in drawn_pairs: continue
        drawn_pairs.add(pair_key)
        
        xs, zs = rel['sub_pos']
        xo, zo = rel['obj_pos']
        
        color = '#d90429'
        lw = 2.2
        alpha = 0.9
        
        if rel['p_name'] in ('left', 'right', 'front', 'behind'):
            ax.annotate('', xy=(xo, zo), xytext=(xs, zs),
                        arrowprops=dict(arrowstyle="-|>", color=color, linestyle='--', lw=lw, alpha=alpha, mutation_scale=16), zorder=12)
            ax.scatter([xs, xo], [zs, zo], color=color, s=28, zorder=14)
        elif rel['p_name'] == 'close by':
            ax.plot([xs, xo], [zs, zo], color=color, linestyle='--', linewidth=lw, alpha=alpha, zorder=11)
            ax.scatter([xs, xo], [zs, zo], color=color, s=28, zorder=14)
        else:
            ax.annotate('', xy=(xo, zo), xytext=(xs, zs),
                        arrowprops=dict(arrowstyle="-|>", color=color, linestyle='--', lw=lw, alpha=alpha, mutation_scale=14), zorder=12)

    if show_text:
        viol_patch = patches.Patch(color='#d90429', label='Violated Spatial Constraint (Loss > 0)')
        leg = ax.legend(handles=[viol_patch], loc='lower left', fontsize=9, framealpha=0.9)
        if leg:
            leg.set_zorder(20)

    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0, format=ext)
    plt.close(fig)

# Backward compatibility alias
visualize_walkable_loss = visualize_edge_gaussian_loss
visualize_walkable_loss_components = visualize_edge_gaussian_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--old_mesh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--relational_json", default=None, help="Path to relational triples json")
    parser.add_argument("--ext", default=".png", help="Output file extension (e.g. .png, .svg, .pdf)")
    parser.add_argument("--robot_width_real", type=float, default=0.35, help="Agent width in meters for walkability computation")
    parser.add_argument("--show_text", action="store_true", default=False, help="Whether to show titles, labels, and text annotations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    objects = load_scene(args.json, args.scene_id, args.old_mesh_dir)
    if objects is None:
        print(f"Scene {args.scene_id} not found in {args.json}")
        return
        
    bounds = get_scene_bounds(objects)
    
    print(f"Generating Outer Loss Visualization...")
    visualize_outer_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_outer_loss{args.ext}"), show_text=args.show_text)
    
    print(f"Generating Collision Loss Visualization...")
    visualize_collision_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_collision_loss{args.ext}"), show_text=args.show_text)
    
    print(f"Generating Walkable Loss Visualization (Edge-Gaussian)...")
    visualize_edge_gaussian_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss{args.ext}"), robot_width_real=args.robot_width_real, show_text=args.show_text)
    
    print(f"Generating Walkable Loss Component Visualizations...")
    visualize_center_penalty_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_center_penalty{args.ext}"), show_text=args.show_text)
    visualize_object_gaussian_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_object_gaussian{args.ext}"), robot_width_real=args.robot_width_real, show_text=args.show_text)
    visualize_edge_gaussian_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_edge_gaussian{args.ext}"), robot_width_real=args.robot_width_real, show_text=args.show_text)
    visualize_pathfinding_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_pathfinding{args.ext}"), robot_width_real=args.robot_width_real, show_text=args.show_text)
    
    print(f"Generating Relational Loss Visualization...")
    triples, _ = load_triples(args.scene_id, args.relational_json)
    visualize_relational_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_relational_loss{args.ext}"), triples=triples, show_text=args.show_text)
    
    print(f"Done! All loss visualizations generated in {args.out_dir}")

if __name__ == "__main__":
    main()
