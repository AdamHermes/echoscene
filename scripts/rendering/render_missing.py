import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_obb_corners(x, z, l, w, angle_rad):
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append([x + rx, z + rz])
    return np.array(rotated)

def main():
    json_path = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json"
    out_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/render_imgs/echoscene"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    scene_ids = data.get("scene_ids", [])
    print(f"Total scenes: {len(scene_ids)}")
    
    class_labels_list = data.get("class_labels", [])
    translations_list = data.get("translations", [])
    sizes_list = data.get("sizes", [])
    angles_list = data.get("angles", [])
    objectness_list = data.get("objectness", [])
    
    cmap = plt.colormaps.get_cmap('tab20')
    
    rendered = 0
    skipped = 0
    
    for idx, scene_id in enumerate(scene_ids):
        out_path = os.path.join(out_dir, f"{scene_id}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue
            
        class_labels = class_labels_list[idx]
        translations = translations_list[idx]
        sizes = sizes_list[idx]
        angles = angles_list[idx]
        
        if objectness_list:
            obj_mask = objectness_list[idx]
        else:
            obj_mask = [[1.0]] * len(class_labels)
            
        fig, ax = plt.subplots(figsize=(9, 9))
        
        all_xs = []
        all_zs = []
        
        for i in range(len(class_labels)):
            if len(class_labels[i]) == 0:
                continue
            cls_idx = int(np.argmax(class_labels[i]))
            
            if isinstance(obj_mask[i], list):
                obj_m = obj_mask[i][0]
            else:
                obj_m = obj_mask[i]
                
            is_layout = (obj_m < 0.5) or (cls_idx == 14) 
            
            x = translations[i][0]
            z = translations[i][2]
            
            l = sizes[i][0]
            w = sizes[i][2]
            
            if is_layout and l < 0.15 and w < 0.15:
                continue
                
            name = "Floor/Layout" if is_layout else f"Class_{cls_idx}"
            color = "#bdc3c7" if is_layout else cmap(cls_idx % 20)
            alpha = 0.25 if is_layout else 0.7
            zorder = 0 if is_layout else 10
            
            angle_rad = angles[i][0]
            
            obb_corners = get_obb_corners(x, z, l, w, angle_rad)
            min_x, min_z = np.min(obb_corners, axis=0)
            max_x, max_z = np.max(obb_corners, axis=0)
            
            all_xs.extend(obb_corners[:, 0])
            all_zs.extend(obb_corners[:, 1])
            
            obb_polygon = patches.Polygon(
                obb_corners, closed=True, facecolor=color, 
                edgecolor='black', alpha=alpha, linewidth=1.5, zorder=zorder
            )
            ax.add_patch(obb_polygon)
            
            if not is_layout:
                ax.text(x, z, name, ha='center', va='center', 
                        fontsize=9, weight='bold', zorder=zorder+1,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            else:
                ax.text(min_x, min_z, name, ha='left', va='bottom', 
                        fontsize=8, color='#555555', zorder=zorder+1)
                    
        ax.set_aspect('equal')
        if len(all_xs) > 0:
            margin = 0.5
            ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
            ax.set_ylim(min(all_zs) - margin, max(all_zs) + margin)
        else:
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            
        ax.grid(True, linestyle=':', alpha=0.5)
        
        title = f"Scene: {scene_id}"
        ax.set_title(title, fontsize=12, weight='bold', pad=10)
        
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        rendered += 1
        
        if rendered % 50 == 0:
            print(f"Rendered {rendered} missing images so far...")
            
    print(f"Finished! Rendered {rendered} new images. Skipped {skipped} existing images.")

if __name__ == "__main__":
    main()
