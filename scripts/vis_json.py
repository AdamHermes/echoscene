import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Color map matching vis.py
COLOR_MAP = {
    "bed": "#3498db",
    "chair": "#e74c3c",
    "nightstand": "#9b59b6",
    "table": "#e67e22",
    "wardrobe": "#2ecc71",
    "lamp": "#f1c40f",
    "floor": "#bdc3c7",
    "sofa": "#e84393",
    "cabinet": "#00cec9",
    "desk": "#fdcb6e",
    "bookshelf": "#d63031",
    "tv_stand": "#1abc9c",
    "shelf": "#95a5a6",
    "_scene_": "#e74c3c"
}

def load_front_dataset_classes(front_dir):
    """
    Builds the exact classes_r mapping directly from the FRONT dataset files:
    mapping.json and classes_all.txt (following threedfront_dataset.py logic).
    """
    mapping_path = os.path.join(front_dir, 'mapping.json')
    classes_path = os.path.join(front_dir, 'classes_all.txt')
    
    if not os.path.exists(mapping_path) or not os.path.exists(classes_path):
        raise FileNotFoundError(f"Missing FRONT dataset files at '{front_dir}'")
        
    with open(mapping_path, 'r') as f:
        mapping_full2simple = json.load(f)
        
    with open(classes_path, 'r') as f:
        vocab_raw = f.readlines()
        
    vocab_simple = [mapping_full2simple[voc.strip('\n')]+'\n' for voc in vocab_raw]
    classes = dict(zip(sorted(list(set([voc.strip('\n') for voc in vocab_simple]))),
                        range(len(list(set(vocab_simple))))))
    classes_r = dict(zip(classes.values(), classes.keys()))
    
    print(f"Loaded {len(classes_r)} classes directly from FRONT dataset at '{front_dir}':")
    for idx, name in sorted(classes_r.items()):
        print(f"  [{idx:2d}] {name}")
        
    return classes_r

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

class JsonVisualizer:
    def __init__(self, json_path, front_dir=None):
        self.json_path = os.path.abspath(json_path)
        
        if front_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            front_dir = os.path.abspath(os.path.join(script_dir, "..", "FRONT"))
            if not os.path.exists(front_dir):
                front_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT"
                
        self.front_dir = front_dir
        self.classes_r = load_front_dataset_classes(self.front_dir)
        
        print(f"Loading {self.json_path}...")
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
            
        self.num_scenes = len(self.data.get("class_labels", []))
        if self.num_scenes == 0:
            print("No scenes found in JSON.")
            exit(1)
            
        print(f"Loaded {self.num_scenes} scenes from JSON.")
        self.current_idx = 0
        self.show_gpt_collision = False
        
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.subplots_adjust(bottom=0.15)
        
        ax_gpt = plt.axes([0.72, 0.02, 0.23, 0.05])
        self.btn_gpt = Button(ax_gpt, 'GPT Mode: OFF')
        self.btn_gpt.on_clicked(self.toggle_gpt)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def toggle_gpt(self, event):
        self.show_gpt_collision = not self.show_gpt_collision
        self.btn_gpt.label.set_text('GPT Mode: ON (Naive AABB)' if self.show_gpt_collision else 'GPT Mode: OFF')
        self.update_plot()

    def decode_object_name(self, cls_idx, size, trans):
        """
        Decodes the exact class name using FRONT dataset mapping.
        Floor is ALWAYS the large flat boundary box (height < 0.1m, area > 1.0m^2).
        _scene_ is the dummy token (tiny size < 0.2m or positioned at -5.6m).
        """
        l, h, w = size
        x, _, z = trans
        
        # 1. Check if it's the scene dummy token
        if (l < 0.2 and w < 0.2) or (x < -4.0 and z < -4.0):
            return "_scene_"
            
        # 2. Check if it's the floor (large flat horizontal box)
        if abs(h) < 0.1 and (l * w) > 1.0:
            return "floor"
            
        # 3. Check if one-hot index is the padding column
        if cls_idx in [0, 6, 14]:
            return "floor"
            
        return self.classes_r.get(cls_idx, f"class_{cls_idx}")

    def update_plot(self):
        self.ax.clear()
        
        # Load scene data
        scene_ids = self.data.get("scene_ids", [])
        scene_name = scene_ids[self.current_idx] if self.current_idx < len(scene_ids) else "Unknown"
        
        trans = np.array(self.data["translations"][self.current_idx])
        sizes = np.array(self.data["sizes"][self.current_idx])
        angles = np.array(self.data["angles"][self.current_idx])
        if angles.ndim == 2:
            angles = angles.squeeze(-1)
        if np.abs(angles).max() > 6.29:
            angles = np.radians(angles)
            
        classes = np.array(self.data["class_labels"][self.current_idx])
        max_cls = np.argmax(classes, axis=-1)
        
        # Identify floor index (largest flat object in scene)
        flat_indices = [j for j in range(len(sizes)) if abs(sizes[j][1]) < 0.1 and (sizes[j][0] * sizes[j][2]) > 0.5]
        floor_obj_idx = max(flat_indices, key=lambda j: sizes[j][0] * sizes[j][2]) if flat_indices else None
        
        for i in range(len(max_cls)):
            cls_idx = int(max_cls[i])
            if i == floor_obj_idx:
                name = "floor"
            else:
                name = self.decode_object_name(cls_idx, sizes[i], trans[i])
            
            x, _, z = trans[i]
            l, _, w = sizes[i]
            angle = angles[i]
            
            obb_corners = get_obb_corners(x, z, l, w, angle)
            
            if self.show_gpt_collision:
                # Naive unrotated AABB (the bug)
                min_x = x - l / 2
                max_x = x + l / 2
                min_z = z - w / 2
                max_z = z + w / 2
            else:
                # Correct rotated AABB
                min_x, min_z = np.min(obb_corners, axis=0)
                max_x, max_z = np.max(obb_corners, axis=0)
                
            if name == "_scene_":
                # Mark scene location with a red circle and text matching vis.py
                circle = patches.Circle((x, z), radius=0.08, color='red', alpha=0.8, zorder=5)
                self.ax.add_patch(circle)
                self.ax.text(x, z - 0.15, name, ha='center', va='top', 
                             color='red', fontsize=9, weight='bold', zorder=6,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=2))
                continue

            color = COLOR_MAP.get(name, "#1abc9c")
            alpha = 0.25 if name in ["floor", "room_boundary"] else 0.7
            
            # Draw OBB (Oriented Bounding Box)
            obb_polygon = patches.Polygon(
                obb_corners, closed=True, facecolor=color, 
                edgecolor='black', alpha=alpha, linewidth=1.5
            )
            self.ax.add_patch(obb_polygon)
            
            # Draw AABB (Dashed Red Line) for everything except floor/lamp
            if name not in ["floor", "lamp"]:
                aabb_rect = patches.Rectangle(
                    (min_x, min_z), max_x - min_x, max_z - min_z, 
                    linewidth=1, edgecolor='#e74c3c', facecolor='none', 
                    linestyle='--', alpha=0.6
                )
                self.ax.add_patch(aabb_rect)
            
            # Text label
            if name != "floor":
                self.ax.text(x, z, name, ha='center', va='center', 
                             fontsize=9, weight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                             
        # View settings
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Header text
        aabb_status = "GPT Mode (Naive)" if self.show_gpt_collision else "Normal Mode (Rotated AABB)"
        title = f"File: {os.path.basename(self.json_path)} | Scene [{self.current_idx + 1}/{self.num_scenes}]\n"
        title += f"ID: {scene_name} | {aabb_status}\n"
        title += "← / →: Switch Scene | G: Toggle GPT Mode"
        self.ax.set_title(title, fontsize=11, weight='bold', pad=10)
        
        # Legend matching vis.py
        aabb_label = 'Naive Unrotated AABB (GPT Mode)' if self.show_gpt_collision else 'Correct AABB Boundary'
        custom_lines = [
            patches.Patch(facecolor='#3498db', edgecolor='black', alpha=0.7, label='True OBB Shape'),
            plt.Line2D([0], [0], color='#e74c3c', lw=1, linestyle='--', label=aabb_label)
        ]
        self.ax.legend(handles=custom_lines, loc='upper right')
        
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % self.num_scenes
            self.update_plot()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % self.num_scenes
            self.update_plot()
        elif event.key in ['g', 'G']:
            self.toggle_gpt(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizes complete scene JSON predictions with FRONT dataset names.")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file (e.g. physcene_collision_input.json)")
    parser.add_argument("--front_dir", type=str, default=None, help="Optional path to FRONT dataset folder (defaults to FRONT dir)")
    args = parser.parse_args()
    
    vis = JsonVisualizer(args.json, front_dir=args.front_dir)
    plt.show()

