import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Button

def load_scene_from_log_file(file_path):
    objects = []

    color_map = {
        "bed": "#3498db",
        "chair": "#e74c3c",
        "nightstand": "#9b59b6",
        "table": "#e67e22",
        "wardrobe": "#2ecc71",
        "lamp": "#f1c40f",
        "floor": "#bdc3c7"
    }

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_parsing = False

    for line in lines:

        if "------" in line:
            start_parsing = True
            continue

        if "Pairwise overlap" in line:
            break

        if not start_parsing:
            continue

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 8:
            continue

        try:
            name = parts[0]

            l = float(parts[1])
            w = float(parts[3])

            x = float(parts[4])
            z = float(parts[6])

            angle_str = parts[7].replace("°", "")
            angle = float(angle_str)

            objects.append({
                "name": name,
                "l": l,
                "w": w,
                "x": x,
                "z": z,
                "angle": angle,
                "color": color_map.get(name, "#1abc9c")
            })

        except Exception as e:
            print("Parse error:", e)
            print("LINE:", line)

    return objects

def get_obb_corners(x, z, l, w, angle_deg):
    """Calculates the 4 corners of the Oriented Bounding Box (OBB)."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append([x + rx, z + rz])
    return np.array(rotated)

class SceneVisualizer:
    def __init__(self):
        # Scan for all txt files in the current folder
        self.files = sorted(glob.glob("*.txt"))
        if not self.files:
            print("No .txt log files found in the current directory!")
            exit(1)
            
        # Try to start at debug_bbox.txt if it exists
        self.current_idx = self.files.index("debug_bbox.txt") if "debug_bbox.txt" in self.files else 0
        
        self.show_gpt_collision = False
        
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.subplots_adjust(bottom=0.15)
        
        ax_gpt = plt.axes([0.75, 0.02, 0.2, 0.06])
        self.btn_gpt = Button(ax_gpt, 'GPT Mode: OFF')
        self.btn_gpt.on_clicked(self.toggle_gpt)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def toggle_gpt(self, event):
        self.show_gpt_collision = not self.show_gpt_collision
        self.btn_gpt.label.set_text('GPT Mode: ON (Naive AABB)' if self.show_gpt_collision else 'GPT Mode: OFF')
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        file_path = self.files[self.current_idx]
        objects = load_scene_from_log_file(file_path)
        
        for obj in objects:
            obb_corners = get_obb_corners(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
            
            if self.show_gpt_collision:
                # Naive unrotated AABB (the bug)
                min_x = obj["x"] - obj["l"]/2
                max_x = obj["x"] + obj["l"]/2
                min_z = obj["z"] - obj["w"]/2
                max_z = obj["z"] + obj["w"]/2
            else:
                # Correct rotated AABB
                min_x, min_z = np.min(obb_corners, axis=0)
                max_x, max_z = np.max(obb_corners, axis=0)
            
            if obj["name"] == "_scene_":
                # Mark scene location with a red circle and text
                circle = patches.Circle((obj["x"], obj["z"]), radius=0.08, color='red', alpha=0.8, zorder=5)
                self.ax.add_patch(circle)
                self.ax.text(obj["x"], obj["z"] - 0.15, obj["name"], ha='center', va='top', 
                             color='red', fontsize=9, weight='bold', zorder=6,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=2))
                continue

            alpha = 0.25 if obj["name"] == "floor" else 0.7
            
            # Draw OBB (Oriented Bounding Box)
            obb_polygon = patches.Polygon(
                obb_corners, closed=True, facecolor=obj["color"], 
                edgecolor='black', alpha=alpha, linewidth=1.5
            )
            self.ax.add_patch(obb_polygon)
            
            # Draw AABB (Dashed Red Line) for everything except floor/lamp
            if obj["name"] not in ["floor", "lamp"]:
                aabb_rect = patches.Rectangle(
                    (min_x, min_z), max_x - min_x, max_z - min_z, 
                    linewidth=1, edgecolor='#e74c3c', facecolor='none', 
                    linestyle='--', alpha=0.6
                )
                self.ax.add_patch(aabb_rect)
            
            # Label
            if obj["name"] != "floor":
                self.ax.text(obj["x"], obj["z"], obj["name"], ha='center', va='center', 
                             fontsize=9, weight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # View settings
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        # Header text info
        title = f"File [{self.current_idx + 1}/{len(self.files)}]: {file_path}\n"
        title += "Use ← or → arrow keys to switch logs"
        self.ax.set_title(title, fontsize=12, weight='bold', pad=10)
        
        # Legend
        aabb_label = 'Naive Unrotated AABB (GPT Mode)' if self.show_gpt_collision else 'Correct AABB Boundary'
        custom_lines = [
            patches.Patch(facecolor='#3498db', edgecolor='black', alpha=0.7, label='True OBB Shape'),
            plt.Line2D([0], [0], color='#e74c3c', lw=1, linestyle='--', label=aabb_label)
        ]
        self.ax.legend(handles=custom_lines, loc='upper right')
        
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.files)
            self.update_plot()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.files)
            self.update_plot()
        elif event.key == 'g':
            self.toggle_gpt(None)

if __name__ == "__main__":
    vis = SceneVisualizer()
    plt.show()