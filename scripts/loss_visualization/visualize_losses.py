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
import heapq

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
            "id": j + 1,  # 1-indexed original index for scene graph relationship matching
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
    """Load scene graph relationships/triples for a given scene_id."""
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
            
    # Search FRONT directory as fallback
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
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

def plot_base_objects(ax, objects, fade_floor=True, show_labels=False):
    for obj in objects:
        if obj["name"] == "floor":
            alpha = 0.15 if fade_floor else 0.4
            poly = patches.Polygon(obj["corners"], closed=True, facecolor="#cccccc", edgecolor='none', alpha=alpha)
            ax.add_patch(poly)
            
            boundary = patches.Polygon(obj["corners"], closed=True, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--')
            ax.add_patch(boundary)
        else:
            poly = patches.Polygon(obj["corners"], closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.5, linewidth=1.0)
            ax.add_patch(poly)
            if show_labels:
                ax.text(obj["x"], obj["z"], f"#{obj['id']} {obj['name']}", fontsize=8, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.8), zorder=15)

def visualize_outer_loss(objects, bounds, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Room Outer Loss\n(Penalizes objects extending past room boundaries)", bounds)
    
    plot_base_objects(ax, objects, fade_floor=False)
    
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
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)

def visualize_collision_loss(objects, bounds, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Collision Loss\n(Penalizes overlaps between objects)", bounds)
    
    plot_base_objects(ax, objects, fade_floor=True)
    
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
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)

def heuristic_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_shortest_path(matrix, start, end):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_set = [(0, start)]
    parent_map = {}
    g_cost = {node: float('inf') for node in np.ndindex(matrix.shape)}
    g_cost[start] = 0
    count = 0
    while open_set and count < 5000:
        count += 1
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in parent_map:
                path.append(current)
                current = parent_map[current]
            path.append(start)
            return path[::-1]

        for direction in directions:
            new_node = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= new_node[0] < matrix.shape[0] and 0 <= new_node[1] < matrix.shape[1]:
                tentative_g_cost = g_cost[current] + matrix[new_node]
                if tentative_g_cost < g_cost[new_node]:
                    parent_map[new_node] = current
                    g_cost[new_node] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic_distance(new_node, end) * 0.01
                    heapq.heappush(open_set, (f_cost, new_node))

    return None

def visualize_center_penalty_loss(objects, bounds, out_path, sigma=0.5):
    """Visualize Center Penalty Walkability Loss."""
    fig, ax = plt.subplots(figsize=(8, 8))
    setup_plot(ax, "Center Penalty Walkable Loss\n(Radial Gaussian penalty centered at room origin)", bounds)
    plot_base_objects(ax, objects, fade_floor=False)
    
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    grid_x = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, 200)
    grid_z = np.linspace(bounds[2] - 0.5, bounds[3] + 0.5, 200)
    GXX, GZZ = np.meshgrid(grid_x, grid_z)
    dist_sq = GXX**2 + GZZ**2
    center_heat = np.exp(-dist_sq / sigma)
    
    extent = [bounds[0] - 0.5, bounds[1] + 0.5, bounds[2] - 0.5, bounds[3] + 0.5]
    heatmap_img = ax.imshow(center_heat, extent=extent, origin='lower', aspect='equal', cmap='magma', alpha=0.6, zorder=5)
    
    ax.plot(0, 0, 'w*', markersize=14, markeredgecolor='black', zorder=15, label="Room Center (0,0)")
    
    # Draw concentric distance rings
    for r in [0.5, 1.0, 1.5, 2.0]:
        ring = patches.Circle((0, 0), r, fill=False, edgecolor='white', linestyle=':', alpha=0.7, zorder=6)
        ax.add_patch(ring)
        ax.text(r * 0.707, r * 0.707, f"{r}m", color='white', fontsize=7, alpha=0.8, zorder=7)

    for obj in furnitures:
        dist_c = np.sqrt(obj["x"]**2 + obj["z"]**2)
        pen_c = np.exp(-(obj["x"]**2 + obj["z"]**2) / sigma)
        ax.plot(obj["x"], obj["z"], 'ro', markersize=6, zorder=12)
        ax.plot([0, obj["x"]], [0, obj["z"]], 'r--', alpha=0.6, linewidth=1.2, zorder=11)
        if pen_c > 0.05:
            ax.text(obj["x"], obj["z"] + 0.15, f"{pen_c:.2f}", color='darkred', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.1", fc="yellow", ec="red", lw=0.8, alpha=0.9), zorder=15)
        
    fig.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label="Center Penalty Density")
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)

def visualize_edge_gaussian_loss(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5, sigma_scale=0.5):
    """Visualize Exact OBB Edge-Gaussian Walkability Loss (Component 1 & Component 2)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    furnitures = [o for o in objects if o["name"] != "floor" and o["y"] < robot_hight_real]
    
    res = 300
    gx = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, res)
    gz = np.linspace(bounds[2] - 0.5, bounds[3] + 0.5, res)
    GXX, GZZ = np.meshgrid(gx, gz)
    
    ax1 = axes[0]
    setup_plot(ax1, "Edge-Gaussian Component 1: Floor OBB Heatmap\n(Gaussians radiate directly from rotated rectangular edges)", bounds)
    plot_base_objects(ax1, objects, fade_floor=False)
    
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

    extent = [bounds[0] - 0.5, bounds[1] + 0.5, bounds[2] - 0.5, bounds[3] + 0.5]
    hm1 = ax1.imshow(combined_heatmap, extent=extent, origin='lower', aspect='equal', cmap='plasma', alpha=0.6, zorder=5)
    fig.colorbar(hm1, ax=ax1, fraction=0.046, pad=0.04, label="Edge Gaussian Density")

    ax2 = axes[1]
    setup_plot(ax2, "Edge-Gaussian Component 2: Individual 2D Object Heatmaps & Repulsion\n(Every object emits a filled 2D Edge-Gaussian heatmap radiating from its rotated edges)", bounds)
    plot_base_objects(ax2, objects, fade_floor=False)
    
    for i, obj in enumerate(furnitures):
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
        g_single = np.exp(-d_edge**2 / (2.0 * sigma**2))
        
        levels = np.linspace(0.15, 1.0, 15)
        ax2.contourf(GXX, GZZ, g_single, levels=levels, cmap='magma', alpha=0.25, zorder=6)

    for i in range(len(furnitures)):
        obj_i = furnitures[i]
        sigma_i = (obj_i["l"] + obj_i["w"]) / 2.0 * sigma_scale
        
        for j in range(i + 1, len(furnitures)):
            obj_j = furnitures[j]
            
            dx_ij = obj_j["x"] - obj_i["x"]
            dz_ij = obj_j["z"] - obj_i["z"]
            angle_i = np.deg2rad(obj_i["angle"])
            
            loc_x =  dx_ij * np.cos(angle_i) + dz_ij * np.sin(angle_i)
            loc_z = -dx_ij * np.sin(angle_i) + dz_ij * np.cos(angle_i)
            
            dist_x = max(abs(loc_x) - obj_i["l"]/2.0, 0.0)
            dist_z = max(abs(loc_z) - obj_i["w"]/2.0, 0.0)
            d_edge_i = np.sqrt(dist_x**2 + dist_z**2)
            
            avg_size_j = (obj_j["l"] + obj_j["w"]) / 2.0
            edge_to_edge = max(0.0, d_edge_i - avg_size_j / 2.0)
            
            penalty = np.exp(-edge_to_edge**2 / (2.0 * sigma_i**2))
            
            if penalty > 0.02:
                ax2.plot([obj_i["x"], obj_j["x"]], [obj_i["z"], obj_j["z"]], color='cyan', linewidth=1.5 + penalty*3.5, alpha=min(0.95, penalty + 0.2), linestyle='-', zorder=12)
                ax2.text((obj_i["x"] + obj_j["x"])/2, (obj_i["z"] + obj_j["z"])/2, f"{penalty:.2f}", color='navy', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="cyan", ec="blue", lw=1, alpha=0.9), zorder=15)

    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)

def visualize_walkable_loss_components(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5, sigma_scale=0.5, sigma_center=0.5):
    """Visualize multi-component walkable loss in a comprehensive 2x2 grid layout."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    if not floor_obj:
        print("Warning: No floor object found for walkable loss components visualization.")
        return
        
    floor_corners = floor_obj["corners"]
    floor_cx, floor_cz = floor_obj["x"], floor_obj["z"]
    floor_corners_centered = floor_corners - np.array([floor_cx, floor_cz])
    
    all_pts_centered = [floor_corners_centered]
    for o in furnitures:
        all_pts_centered.append(o["corners"] - np.array([floor_cx, floor_cz]))
    all_pts = np.vstack(all_pts_centered)
    scale = np.abs(all_pts).max() + 0.2
    
    image_size = 256
    def map_to_image(point):
        x, y = point
        return int(x / scale * image_size / 2) + image_size // 2, int(y / scale * image_size / 2) + image_size // 2

    # -------------------------------------------------------------
    # Panel A (Top-Left): Pathfinding & Reachability
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
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

    vis = np.ones((image_size, image_size, 3), dtype=np.float32) * 0.92
    vis[floor_mask] = [0.85, 0.85, 0.85]
    component_colors = [[0.3, 0.8, 0.3], [0.3, 0.65, 0.85], [0.85, 0.85, 0.3], [0.7, 0.35, 0.85]]
    for label_id in range(1, num_labels):
        mask = labels == label_id
        color = component_colors[(label_id - 1) % len(component_colors)]
        vis[mask] = color
    vis[blocked] = [0.95, 0.45, 0.4]

    extent_c = [-scale, scale, -scale, scale]
    ax_a.imshow(vis, extent=extent_c, origin='lower', aspect='equal')
    
    # Overlay object outlines & floor
    for obj in furnitures:
        if obj["y"] > robot_hight_real: continue
        obj_corners_centered = obj["corners"] - np.array([floor_cx, floor_cz])
        poly = patches.Polygon(obj_corners_centered, closed=True, facecolor='none', edgecolor='black', linewidth=1.2)
        ax_a.add_patch(poly)
    floor_poly = patches.Polygon(floor_corners_centered, closed=True, facecolor='none', edgecolor='black', linewidth=2.0, linestyle='--')
    ax_a.add_patch(floor_poly)

    # Calculate Dijkstra shortest path if multiple islands exist
    path_info = ""
    if num_labels > 2:
        area_1, area_2 = np.zeros_like(walkable_u8), np.zeros_like(walkable_u8)
        for label in range(1, num_labels):
            m = (labels == label).astype(np.uint8)
            if m.sum() > area_2.sum(): area_2 = m.copy()
            if area_2.sum() > area_1.sum(): area_2, area_1 = area_1.copy(), area_2.copy()
        if area_2.sum() > 20:
            d1 = cv2.distanceTransform(area_1, cv2.DIST_L2, 5)
            p1 = np.unravel_index(np.argmax(d1), area_1.shape)
            d2 = cv2.distanceTransform(area_2, cv2.DIST_L2, 5)
            p2 = np.unravel_index(np.argmax(d2), area_2.shape)
            
            box_wall_map = (obj_mask.astype(np.float32) + (~floor_mask).astype(np.float32) * 5.0)
            shortest_path = find_shortest_path(box_wall_map, p1, p2)
            if shortest_path:
                px_coords = np.array(shortest_path)
                # Convert grid image coordinates to centered room coordinates
                map_x = (px_coords[:, 1] - image_size/2) * 2 / image_size * scale
                map_z = (px_coords[:, 0] - image_size/2) * 2 / image_size * scale
                ax_a.plot(map_x, map_z, color='cyan', linewidth=3.0, linestyle='-', zorder=12, label="Dijkstra Shortest Path")
                ax_a.scatter([map_x[0], map_x[-1]], [map_z[0], map_z[-1]], color='yellow', s=70, zorder=15)
                path_info = f" | Disconnected Islands Path Active ({len(shortest_path)} steps)"

    total_floor_px = floor_mask.sum()
    walkable_px = walkable.sum()
    walkable_pct = 100.0 * walkable_px / max(total_floor_px, 1)
    ax_a.set_title(f"(A) Component 1: Reachability & Pathfinding\nWalkable: {walkable_pct:.1f}% | Components: {num_labels - 1}{path_info}", fontsize=11, fontweight='bold')
    ax_a.set_xlim(-scale - 0.2, scale + 0.2)
    ax_a.set_ylim(-scale - 0.2, scale + 0.2)
    ax_a.set_aspect('equal')
    ax_a.axis('off')

    # -------------------------------------------------------------
    # Panel B (Top-Right): Center Penalty
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    setup_plot(ax_b, "(B) Component 2: Room Center Penalty\n(Radial Gaussian centered at room origin)", bounds)
    plot_base_objects(ax_b, objects, fade_floor=False)
    
    grid_x = np.linspace(bounds[0] - 0.5, bounds[1] + 0.5, 200)
    grid_z = np.linspace(bounds[2] - 0.5, bounds[3] + 0.5, 200)
    GXX, GZZ = np.meshgrid(grid_x, grid_z)
    center_heat = np.exp(-(GXX**2 + GZZ**2) / sigma_center)
    
    extent_b = [bounds[0] - 0.5, bounds[1] + 0.5, bounds[2] - 0.5, bounds[3] + 0.5]
    hm_b = ax_b.imshow(center_heat, extent=extent_b, origin='lower', aspect='equal', cmap='magma', alpha=0.55, zorder=5)
    ax_b.plot(0, 0, 'w*', markersize=12, markeredgecolor='black', zorder=15)
    
    for r in [0.5, 1.0, 1.5]:
        ax_b.add_patch(patches.Circle((0, 0), r, fill=False, edgecolor='white', linestyle=':', alpha=0.6, zorder=6))

    cp_total = 0.0
    for obj in furnitures:
        pen_c = np.exp(-(obj["x"]**2 + obj["z"]**2) / sigma_center)
        cp_total += pen_c
        ax_b.plot(obj["x"], obj["z"], 'ro', markersize=5, zorder=12)
        ax_b.plot([0, obj["x"]], [0, obj["z"]], 'r--', alpha=0.5, linewidth=1.0, zorder=11)
        if pen_c > 0.05:
            ax_b.text(obj["x"], obj["z"] + 0.12, f"{pen_c:.2f}", color='darkred', fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.1", fc="yellow", ec="red", lw=0.6, alpha=0.85), zorder=15)
    fig.colorbar(hm_b, ax=ax_b, fraction=0.046, pad=0.04, label="Center Penalty Density")

    # -------------------------------------------------------------
    # Panel C (Bottom-Left): Edge-Gaussian Component 1 (Floor Clearance Heatmap)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    setup_plot(ax_c, "(C) Component 3a: Floor Edge-Gaussian Heatmap\n(Gaussians radiate directly from rotated OBB edges)", bounds)
    plot_base_objects(ax_c, objects, fade_floor=False)
    
    c1_heatmap = np.zeros_like(GXX)
    furnitures_ground = [o for o in objects if o["name"] != "floor" and o["y"] < robot_hight_real]
    for obj in furnitures_ground:
        cx, cz = obj["x"], obj["z"]
        l, w = obj["l"], obj["w"]
        angle_rad = np.deg2rad(obj["angle"])
        rel_x, rel_z = GXX - cx, GZZ - cz
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        local_x =  rel_x * cos_a + rel_z * sin_a
        local_z = -rel_x * sin_a + rel_z * cos_a
        dx = np.maximum(np.abs(local_x) - l / 2.0, 0.0)
        dz = np.maximum(np.abs(local_z) - w / 2.0, 0.0)
        d_edge = np.sqrt(dx**2 + dz**2)
        sigma = (l + w) / 2.0 * sigma_scale
        g_field = np.exp(-d_edge**2 / (2.0 * sigma**2))
        c1_heatmap += g_field

    hm_c = ax_c.imshow(c1_heatmap, extent=extent_b, origin='lower', aspect='equal', cmap='plasma', alpha=0.6, zorder=5)
    fig.colorbar(hm_c, ax=ax_c, fraction=0.046, pad=0.04, label="Edge Density")

    # -------------------------------------------------------------
    # Panel D (Bottom-Right): Edge-Gaussian Component 2 (Pairwise Repulsion)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    setup_plot(ax_d, "(D) Component 3b: Pairwise OBB Edge Repulsion\n(Penalizes overlapping edge influence zones between object pairs)", bounds)
    plot_base_objects(ax_d, objects, fade_floor=False)
    
    for obj in furnitures_ground:
        cx, cz = obj["x"], obj["z"]
        l, w = obj["l"], obj["w"]
        angle_rad = np.deg2rad(obj["angle"])
        rel_x, rel_z = GXX - cx, GZZ - cz
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        local_x =  rel_x * cos_a + rel_z * sin_a
        local_z = -rel_x * sin_a + rel_z * cos_a
        dx = np.maximum(np.abs(local_x) - l / 2.0, 0.0)
        dz = np.maximum(np.abs(local_z) - w / 2.0, 0.0)
        d_edge = np.sqrt(dx**2 + dz**2)
        sigma = (l + w) / 2.0 * sigma_scale
        g_single = np.exp(-d_edge**2 / (2.0 * sigma**2))
        ax_d.contourf(GXX, GZZ, g_single, levels=np.linspace(0.15, 1.0, 12), cmap='magma', alpha=0.22, zorder=6)

    c2_rep_total = 0.0
    for i in range(len(furnitures_ground)):
        obj_i = furnitures_ground[i]
        sigma_i = (obj_i["l"] + obj_i["w"]) / 2.0 * sigma_scale
        for j in range(i + 1, len(furnitures_ground)):
            obj_j = furnitures_ground[j]
            dx_ij = obj_j["x"] - obj_i["x"]
            dz_ij = obj_j["z"] - obj_i["z"]
            angle_i = np.deg2rad(obj_i["angle"])
            loc_x =  dx_ij * np.cos(angle_i) + dz_ij * np.sin(angle_i)
            loc_z = -dx_ij * np.sin(angle_i) + dz_ij * np.cos(angle_i)
            dist_x = max(abs(loc_x) - obj_i["l"]/2.0, 0.0)
            dist_z = max(abs(loc_z) - obj_i["w"]/2.0, 0.0)
            d_edge_i = np.sqrt(dist_x**2 + dist_z**2)
            avg_size_j = (obj_j["l"] + obj_j["w"]) / 2.0
            edge_to_edge = max(0.0, d_edge_i - avg_size_j / 2.0)
            penalty = np.exp(-edge_to_edge**2 / (2.0 * sigma_i**2))
            if penalty > 0.02:
                c2_rep_total += penalty
                ax_d.plot([obj_i["x"], obj_j["x"]], [obj_i["z"], obj_j["z"]], color='cyan', linewidth=1.2 + penalty*3.0, alpha=min(0.95, penalty + 0.25), zorder=12)
                ax_d.text((obj_i["x"] + obj_j["x"])/2, (obj_i["z"] + obj_j["z"])/2, f"{penalty:.2f}", color='navy', fontsize=8, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.15", fc="cyan", ec="blue", lw=0.8, alpha=0.85), zorder=15)

    plt.suptitle("Multi-Component Walkable Guidance Loss Breakdown", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1, format=ext)
    plt.close(fig)

def visualize_relational_loss(objects, bounds, out_path, triples=None, predicate_names=None, margin=0.05, close_threshold=0.45, stand_threshold=0.04):
    """Visualize Relational Guidance Loss with a 2D Spatial Scene Graph Diagram & Evaluation Table Card."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1.25, 1.0]})
    
    # -------------------------------------------------------------
    # Panel 1 (Left): Spatial Scene Graph Overlay on Floorplan
    # -------------------------------------------------------------
    ax1 = axes[0]
    setup_plot(ax1, "(A) Spatial Scene Graph Relational Guidance Overlay\n(Enforces Directional, Proximity, & Support Constraints)", bounds)
    plot_base_objects(ax1, objects, fade_floor=False, show_labels=True)
    
    obj_by_id = {o["id"]: o for o in objects}
    spatial_preds = {'left': 1, 'right': 2, 'front': 3, 'behind': 4, 'close by': 5, 'above': 6, 'standing on': 7}
    
    evaluated_list = []
    total_rel_loss = 0.0
    
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
            metric_desc = ""
            
            if p_name in ('left', '1'):
                val = zs - zo
                loss = max(0.0, val + margin)
                metric_desc = f"z_s - z_o = {val:+.2f}m (<= -{margin:.2f}m)"
            elif p_name in ('right', '2'):
                val = zs - zo
                loss = max(0.0, -val + margin)
                metric_desc = f"z_s - z_o = {val:+.2f}m (>= +{margin:.2f}m)"
            elif p_name in ('front', '3'):
                val = xs - xo
                loss = max(0.0, -val + margin)
                metric_desc = f"x_s - x_o = {val:+.2f}m (>= +{margin:.2f}m)"
            elif p_name in ('behind', '4'):
                val = xs - xo
                loss = max(0.0, val + margin)
                metric_desc = f"x_s - x_o = {val:+.2f}m (<= -{margin:.2f}m)"
            elif p_name in ('close by', '5'):
                dist = np.sqrt((xs - xo)**2 + (zs - zo)**2)
                loss = max(0.0, dist - close_threshold)
                metric_desc = f"dist_2d = {dist:.2f}m (<= {close_threshold:.2f}m)"
            elif p_name in ('standing on', 'above', '6', '7'):
                diff_y = abs(ys - yo)
                loss = max(0.0, diff_y - stand_threshold)
                metric_desc = f"|y_s - y_o| = {diff_y:.2f}m (<= {stand_threshold:.2f}m)"

            status = "VIOLATED" if loss > 1e-4 else "SATISFIED"
            evaluated_list.append({
                's_id': sub_obj['id'], 's_name': sub_obj['name'],
                'o_id': obj_obj['id'], 'o_name': obj_obj['name'],
                'p_name': p_name, 'loss': loss, 'status': status,
                'metric_desc': metric_desc,
                'sub_pos': (xs, zs), 'obj_pos': (xo, zo)
            })
            total_rel_loss += loss

    # Draw relation vectors & badges on Panel A
    # Select key/unique relations for clarity
    drawn_pairs = set()
    for rel in evaluated_list:
        pair_key = (rel['s_id'], rel['o_id'], rel['p_name'])
        if pair_key in drawn_pairs: continue
        drawn_pairs.add(pair_key)
        
        xs, zs = rel['sub_pos']
        xo, zo = rel['obj_pos']
        
        is_violated = rel['status'] == "VIOLATED"
        color = '#d9534f' if is_violated else '#5cb85c'
        ls = '--' if is_violated else '-'
        lw = 2.0 if is_violated else 1.2
        alpha = 0.85 if is_violated else 0.45
        
        if rel['p_name'] in ('left', 'right', 'front', 'behind'):
            ax1.annotate('', xy=(xo, zo), xytext=(xs, zs),
                         arrowprops=dict(arrowstyle="->", color=color, linestyle=ls, lw=lw, alpha=alpha, mutation_scale=14), zorder=12)
            if is_violated and rel['loss'] > 0.05:
                mid_x, mid_z = (xs + xo)/2, (zs + zo)/2
                ax1.text(mid_x, mid_z, f"{rel['p_name']}\n+{rel['loss']:.2f}", color='white', fontsize=7, fontweight='bold',
                         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="darkred", lw=0.8, alpha=0.9), zorder=16)
        elif rel['p_name'] == 'close by':
            ax1.add_patch(patches.Circle((xo, zo), close_threshold, fill=False, edgecolor=color, linestyle=':', alpha=0.7, zorder=6))
            ax1.plot([xs, xo], [zs, zo], color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=11)

    # Panel A Legend
    sat_patch = patches.Patch(color='#5cb85c', label='Satisfied Constraint (Loss = 0)')
    viol_patch = patches.Patch(color='#d9534f', label='Violated Constraint (Loss > 0)')
    ax1.legend(handles=[sat_patch, viol_patch], loc='lower left', fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2 (Right): Relational Loss Scorecard & Breakdown Table
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title("(B) Relational Guidance Loss Scorecard & Detailed Breakdown", fontsize=11, fontweight='bold', pad=10)
    
    num_eval = len(evaluated_list)
    num_viol = sum(1 for r in evaluated_list if r['status'] == "VIOLATED")
    num_sat = num_eval - num_viol
    avg_loss = total_rel_loss / max(1, num_eval)
    
    # Render summary header box
    header_text = (f"Total Spatial Relations Evaluated: {num_eval}\n"
                   f"Satisfied: {num_sat} ({100.0*num_sat/max(1,num_eval):.1f}%) | Violated: {num_viol} ({100.0*num_viol/max(1,num_eval):.1f}%)\n"
                   f"Total Relational Guidance Loss: {total_rel_loss:.4f}  (Avg Loss: {avg_loss:.4f})")
    
    ax2.text(0.5, 0.92, header_text, fontsize=10, fontweight='bold', ha='center', va='top', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#ced4da", lw=1.5), zorder=10)
    
    # Build detailed table of top violations & sample relations
    table_data = []
    # Show violations first, sorted by loss descending
    sorted_rels = sorted(evaluated_list, key=lambda x: (x['status'] != "VIOLATED", -x['loss']))
    
    for r in sorted_rels[:16]:  # Show top 16 relations for clean formatting
        status_symbol = "VIOLATED" if r['status'] == "VIOLATED" else "OK"
        table_data.append([
            f"#{r['s_id']} {r['s_name'][:12]}",
            f"--[{r['p_name']}]-->",
            f"#{r['o_id']} {r['o_name'][:12]}",
            r['metric_desc'],
            f"{r['loss']:.4f}",
            status_symbol
        ])
        
    if table_data:
        col_labels = ["Subject", "Predicate", "Object", "Measured Metric vs Target", "Loss", "Status"]
        col_widths = [0.18, 0.15, 0.18, 0.32, 0.10, 0.12]
        
        tab = ax2.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths,
                        loc='center', cellLoc='center', bbox=[0.0, 0.05, 1.0, 0.78])
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        
        # Style table headers and cells
        for (row, col), cell in tab.get_celld().items():
            cell.set_edgecolor('#dee2e6')
            cell.set_linewidth(0.8)
            if row == 0:
                cell.set_facecolor('#e9ecef')
                cell.set_text_props(fontweight='bold', color='#212529')
            else:
                row_status = table_data[row-1][5]
                if row_status == "VIOLATED":
                    cell.set_facecolor('#fdf2f2')
                    if col == 5:
                        cell.set_text_props(fontweight='bold', color='#d9534f')
                else:
                    cell.set_facecolor('#f4fdf4')
                    if col == 5:
                        cell.set_text_props(fontweight='bold', color='#5cb85c')

    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1, format=ext)
    plt.close(fig)

# Backward compatibility aliases
visualize_walkable_loss = visualize_walkable_loss_components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--old_mesh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--relational_json", default=None, help="Path to relational triples json")
    parser.add_argument("--ext", default=".png", help="Output file extension (e.g. .png, .svg, .pdf)")
    parser.add_argument("--robot_width_real", type=float, default=0.35, help="Agent width in meters for walkability computation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    objects = load_scene(args.json, args.scene_id, args.old_mesh_dir)
    if objects is None:
        print(f"Scene {args.scene_id} not found in {args.json}")
        return
        
    bounds = get_scene_bounds(objects)
    
    print(f"Generating Outer Loss Visualization...")
    visualize_outer_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_outer_loss{args.ext}"))
    
    print(f"Generating Collision Loss Visualization...")
    visualize_collision_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_collision_loss{args.ext}"))
    
    print(f"Generating Walkable Loss Visualization (Multi-Component)...")
    visualize_walkable_loss_components(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss{args.ext}"), robot_width_real=args.robot_width_real)
    
    print(f"Generating Walkable Loss Component Visualizations (Pathfinding, Center Penalty, Edge-Gaussian)...")
    visualize_center_penalty_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_center_penalty{args.ext}"))
    visualize_edge_gaussian_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss_edge_gaussian{args.ext}"), robot_width_real=args.robot_width_real)
    
    print(f"Generating Relational Loss Visualization...")
    triples, _ = load_triples(args.scene_id, args.relational_json)
    visualize_relational_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_relational_loss{args.ext}"), triples=triples)
    
    print(f"Done! All loss visualizations generated in {args.out_dir}")

if __name__ == "__main__":
    main()
