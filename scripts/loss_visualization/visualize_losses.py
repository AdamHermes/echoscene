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
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
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
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)

def visualize_walkable_loss(objects, bounds, out_path, robot_width_real=0.35, robot_hight_real=1.5):
    """Visualize walkability loss using the same grid-based approach as physical_guidance.py.
    
    Renders floor plan on a 256x256 image, erodes by agent width, draws object OBBs
    with agent-width expansion, and shows walkable vs blocked regions with connected
    component analysis.
    """
    import cv2
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Find the floor object
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    furnitures = [o for o in objects if o["name"] != "floor"]
    
    if not floor_obj:
        # Fallback: just draw objects with a note
        for ax_idx, (agent_size, title) in enumerate([(robot_width_real, f"Agent={robot_width_real}m (config)"), (0.5, "Agent=0.5m (old default)")]):
            ax = axes[ax_idx]
            ax.set_xlim(bounds[0] - 0.5, bounds[1] + 0.5)
            ax.set_ylim(bounds[2] - 0.5, bounds[3] + 0.5)
            ax.set_aspect('equal')
            ax.set_title(f"Walkability: {title}\n(no floor object found)", fontsize=11)
            for obj in furnitures:
                poly = patches.Polygon(obj["corners"], closed=True, facecolor=obj["color"], edgecolor='black', alpha=0.6)
                ax.add_patch(poly)
        plt.tight_layout()
        ext = os.path.splitext(out_path)[1].strip('.')
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
        plt.close(fig)
        return
    
    # Get floor vertices and compute floor centroid
    floor_corners = floor_obj["corners"]
    floor_cx, floor_cz = floor_obj["x"], floor_obj["z"]
    
    # Center floor corners and furniture relative to floor centroid
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
    
    for ax_idx, (agent_size, title) in enumerate([
        (robot_width_real, f"Agent={robot_width_real}m (config)"),
        (0.5, "Agent=0.5m (old default)")
    ]):
        ax = axes[ax_idx]
        robot_width_px = max(1, int(agent_size / scale * image_size / 2))
        
        # Build the walkability image (same as physical_guidance.py)
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Draw floor polygon (centered)
        floor_img_pts = np.array([map_to_image(v) for v in floor_corners_centered], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(image, [floor_img_pts], (255, 0, 0))
        
        # Erode floor by agent width
        kernel = np.ones((robot_width_px, robot_width_px), dtype=np.uint8)
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        
        # Draw object OBBs with agent-width expansion (relative to floor centroid)
        for obj in furnitures:
            # Filter by height (only objects touching the ground)
            if obj["y"] > robot_hight_real:
                continue
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
        
        # Compute walkable area
        floor_mask = image[:, :, 0] == 255
        obj_mask = image[:, :, 1] == 255
        walkable = floor_mask & ~obj_mask
        blocked = floor_mask & obj_mask
        outside = ~floor_mask
        
        # Connected components on walkable area
        walkable_u8 = (walkable * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(walkable_u8, connectivity=8)
        
        # Compute stats
        total_floor_px = floor_mask.sum()
        walkable_px = walkable.sum()
        walkable_pct = 100.0 * walkable_px / max(total_floor_px, 1)
        
        # Build colored visualization
        vis = np.ones((image_size, image_size, 3), dtype=np.float32) * 0.92  # light gray bg
        
        # Floor area (light gray)
        vis[floor_mask] = [0.85, 0.85, 0.85]
        
        # Walkable regions (green, different shades for components)
        component_colors = [
            [0.4, 0.85, 0.4],  # green
            [0.4, 0.7, 0.85],  # blue-ish
            [0.85, 0.85, 0.4], # yellow-ish
            [0.7, 0.4, 0.85],  # purple
        ]
        for label_id in range(1, num_labels):
            mask = labels == label_id
            color = component_colors[(label_id - 1) % len(component_colors)]
            vis[mask] = color
        
        # Blocked areas (salmon/red)
        vis[blocked] = [1.0, 0.5, 0.4]
        
        # Agent-width erosion zone (show as darker pink border)
        original_floor = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.fillPoly(original_floor, [floor_img_pts], 255)
        erosion_zone = (original_floor == 255) & ~floor_mask
        vis[erosion_zone] = [0.9, 0.3, 0.3]  # dark red for erosion border
        
        # Plot using matplotlib (in centered coordinate frame)
        extent = [-scale, scale, -scale, scale]
        ax.imshow(vis, extent=extent, origin='lower', aspect='equal')
        
        # Overlay object outlines (centered)
        for obj in furnitures:
            if obj["y"] > robot_hight_real:
                continue
            obj_corners_centered = obj["corners"] - np.array([floor_cx, floor_cz])
            poly = patches.Polygon(obj_corners_centered, closed=True, facecolor='none',
                                   edgecolor='black', linewidth=1.2, linestyle='-')
            ax.add_patch(poly)
        
        # Floor boundary (centered)
        floor_poly = patches.Polygon(floor_corners_centered, closed=True, facecolor='none',
                                      edgecolor='black', linewidth=2.0, linestyle='--')
        ax.add_patch(floor_poly)
        
        # Draw agent size indicator at center
        agent_circle = patches.Circle((0, 0), agent_size / 2, fill=False,
                                       edgecolor='navy', linewidth=2, linestyle=':', zorder=10)
        ax.add_patch(agent_circle)
        ax.plot(0, 0, 'x', color='navy', markersize=8, markeredgewidth=2, zorder=10)
        
        ax.set_title(f"Walkability: {title}\n"
                     f"Walkable: {walkable_pct:.1f}% | Components: {num_labels - 1}",
                     fontsize=11)
        ax.set_xlim(-scale - 0.2, scale + 0.2)
        ax.set_ylim(-scale - 0.2, scale + 0.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    ext = os.path.splitext(out_path)[1].strip('.')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format=ext)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--old_mesh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
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
    
    print(f"Generating Walkable Loss Visualization...")
    visualize_walkable_loss(objects, bounds, os.path.join(args.out_dir, f"{args.scene_id}_walkable_loss{args.ext}"), robot_width_real=args.robot_width_real)
    
    print(f"Done! Check the {args.out_dir} directory.")

if __name__ == "__main__":
    main()
