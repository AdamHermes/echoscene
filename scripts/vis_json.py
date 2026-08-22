import os
import glob
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

# Fallback coarse classes mapping for when scan_id is not found in dataset
COARSE_CLASSES_MAP = {
    0: '_scene_',
    1: 'bed',
    2: 'bookshelf',
    3: 'cabinet',
    4: 'chair',
    5: 'desk',
    6: 'floor',
    7: 'lamp',
    8: 'nightstand',
    9: 'shelf',
    10: 'sofa',
    11: 'table',
    12: 'tv_stand',
    13: 'wardrobe'
}

def load_dataset_index(front_dir):
    """
    Loads all relationships_*.json and mapping.json from FRONT directory
    to map scene_id -> list of fine-grained object names.
    """
    mapping_path = os.path.join(front_dir, 'mapping.json')
    mapping = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {mapping_path}: {e}")

    scan_to_objects = {}
    rel_files = glob.glob(os.path.join(front_dir, 'relationships_*.json'))
    for rf in rel_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                for s in data.get('scans', []):
                    scan_id = s.get('scan')
                    if scan_id and scan_id not in scan_to_objects:
                        scan_to_objects[scan_id] = list(s.get('objects', {}).values())
        except Exception as e:
            print(f"Warning: Failed to load {rf}: {e}")

    print(f"Indexed {len(scan_to_objects)} scenes from FRONT dataset at '{front_dir}'.")
    return scan_to_objects, mapping

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
            # Default to FRONT directory relative to script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            front_dir = os.path.abspath(os.path.join(script_dir, "..", "FRONT"))
            if not os.path.exists(front_dir):
                # Fallback to hardcoded absolute path
                front_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT"
                
        self.front_dir = front_dir
        self.scan_to_objects, self.mapping = load_dataset_index(self.front_dir)
        
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
        self.use_fine_grained = True
        
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.subplots_adjust(bottom=0.15)
        
        ax_gpt = plt.axes([0.72, 0.02, 0.23, 0.05])
        self.btn_gpt = Button(ax_gpt, 'GPT Mode: OFF')
        self.btn_gpt.on_clicked(self.toggle_gpt)
        
        ax_mode = plt.axes([0.45, 0.02, 0.25, 0.05])
        self.btn_mode = Button(ax_mode, 'Names: Fine-Grained')
        self.btn_mode.on_clicked(self.toggle_names)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def toggle_gpt(self, event):
        self.show_gpt_collision = not self.show_gpt_collision
        self.btn_gpt.label.set_text('GPT Mode: ON (Naive AABB)' if self.show_gpt_collision else 'GPT Mode: OFF')
        self.update_plot()

    def toggle_names(self, event):
        self.use_fine_grained = not self.use_fine_grained
        self.btn_mode.label.set_text('Names: Fine-Grained' if self.use_fine_grained else 'Names: Coarse')
        self.update_plot()

    def get_scene_object_names(self, scene_id, max_cls, sizes):
        """
        Determines the display name for each object in the scene.
        """
        n_objs = len(max_cls)
        names = []
        
        # Check if we have exact dataset ground truth object sequence for this scan_id
        if scene_id in self.scan_to_objects:
            gt_objs = list(self.scan_to_objects[scene_id])
            # The model appends _scene_ at the very end
            if len(gt_objs) == n_objs - 1:
                gt_objs.append('_scene_')
            elif len(gt_objs) < n_objs:
                # If sizes don't match, pad with _scene_ or unknown
                while len(gt_objs) < n_objs:
                    gt_objs.append('_scene_')
            
            for i in range(n_objs):
                name = gt_objs[i]
                if not self.use_fine_grained:
                    name = self.mapping.get(name, name)
                names.append(name)
            return names
        
        # Fallback to coarse mapping from one-hot class index
        for i in range(n_objs):
            cls_idx = max_cls[i]
            l, _, w = sizes[i]
            if cls_idx == 14: # Padding column used for floor and _scene_
                if (l * w) < 0.1 or (l < 0.2 and w < 0.2):
                    name = "_scene_"
                else:
                    name = "floor"
            else:
                name = COARSE_CLASSES_MAP.get(cls_idx, f"class_{cls_idx}")
            names.append(name)
            
        return names

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
        
        object_names = self.get_scene_object_names(scene_name, max_cls, sizes)
        
        for i in range(len(max_cls)):
            name = object_names[i]
            coarse_name = self.mapping.get(name, name)
            
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

            color = COLOR_MAP.get(coarse_name, "#1abc9c")
            alpha = 0.25 if name in ["floor", "room_boundary"] or coarse_name == "floor" else 0.7
            
            # Draw OBB (Oriented Bounding Box)
            obb_polygon = patches.Polygon(
                obb_corners, closed=True, facecolor=color, 
                edgecolor='black', alpha=alpha, linewidth=1.5
            )
            self.ax.add_patch(obb_polygon)
            
            # Draw AABB (Dashed Red Line) for everything except floor/lamp
            if coarse_name not in ["floor", "lamp"]:
                aabb_rect = patches.Rectangle(
                    (min_x, min_z), max_x - min_x, max_z - min_z, 
                    linewidth=1, edgecolor='#e74c3c', facecolor='none', 
                    linestyle='--', alpha=0.6
                )
                self.ax.add_patch(aabb_rect)
            
            # Text label
            if name != "floor" and coarse_name != "floor":
                self.ax.text(x, z, name, ha='center', va='center', 
                             fontsize=8, weight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                             
        # View settings
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Header text
        aabb_status = "GPT Mode (Naive)" if self.show_gpt_collision else "Normal Mode (Rotated AABB)"
        name_status = "Fine-Grained" if self.use_fine_grained else "Coarse"
        title = f"File: {os.path.basename(self.json_path)} | Scene [{self.current_idx + 1}/{self.num_scenes}]\n"
        title += f"ID: {scene_name} | {name_status} | {aabb_status}\n"
        title += "←/→: Switch Scene | G: Toggle GPT Mode | F: Toggle Coarse/Fine Names"
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
        elif event.key in ['f', 'F']:
            self.toggle_names(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizes complete scene JSON predictions with dataset ground-truth names.")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file (e.g. physcene_collision_input.json)")
    parser.add_argument("--front_dir", type=str, default=None, help="Optional path to FRONT dataset folder (defaults to FRONT dir)")
    args = parser.parse_args()
    
    vis = JsonVisualizer(args.json, front_dir=args.front_dir)
    plt.show()
