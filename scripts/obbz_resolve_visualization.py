import json
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Polygon
import sys
import os

def get_corners(translation, size, angle):
    x, y, z = translation
    dx, dy, dz = size
    theta = angle[0]
    
    c = math.cos(theta)
    s = math.sin(theta)
    
    hx, hz = dx / 2, dz / 2
    
    corners_local = [
        np.array([hx, hz]),
        np.array([-hx, hz]),
        np.array([-hx, -hz]),
        np.array([hx, -hz])
    ]
    
    corners = []
    for cx, cz in corners_local:
        nx = cx * c - cz * s + x
        nz = cx * s + cz * c + z
        corners.append(np.array([nx, nz]))
        
    return corners

def draw_scene(ax, translations, sizes, angles, labels, class_names, title):
    ax.set_title(title)
    ax.set_aspect('equal')
    
    for i in range(len(translations)):
        corners = get_corners(translations[i], sizes[i], angles[i])
        poly = Polygon(corners, fill=True, alpha=0.3, edgecolor='black', linewidth=1.5)
        ax.add_patch(poly)
        
        x, y, z = translations[i]
        
        # Determine the label cleanly
        label_text = str(i)
        if labels and i < len(labels):
            lbl = labels[i]
            if isinstance(lbl, list):
                if sum(lbl) > 0:
                    idx = int(np.argmax(lbl))
                    if class_names and idx < len(class_names):
                        label_text = class_names[idx]
                    else:
                        label_text = str(idx)
            else:
                if isinstance(lbl, int) and class_names and lbl < len(class_names):
                    label_text = class_names[lbl]
                else:
                    label_text = str(lbl)
                
        ax.text(x, z, label_text, ha='center', va='center', fontsize=8)
        
    ax.autoscale_view()

class Visualizer:
    def __init__(self, start_idx=0):
        input_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/physcene_collision_input_merged.json'
        output_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/physcene_collision_resolved.json'
        
        with open(input_file, 'r') as f:
            self.data_in = json.load(f)
        
        try:
            with open(output_file, 'r') as f:
                self.data_out = json.load(f)
        except FileNotFoundError:
            print(f"Error: {output_file} not found. Run solve_obb_collisions.py first.")
            sys.exit(1)
            
        self.num_scenes = len(self.data_in['scene_ids'])
        self.idx = start_idx
        
        self.class_names = []
        classes_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT/classes_all.txt'
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_plot()
        plt.show()

    def on_key(self, event):
        if event.key == 'right':
            self.idx = (self.idx + 1) % self.num_scenes
            self.update_plot()
        elif event.key == 'left':
            self.idx = (self.idx - 1) % self.num_scenes
            self.update_plot()

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        scene_id = self.data_in['scene_ids'][self.idx]
        labels = self.data_in.get('class_labels', [[] for _ in range(self.num_scenes)])[self.idx]
        
        draw_scene(self.ax1, 
                   self.data_in['translations'][self.idx], 
                   self.data_in['sizes'][self.idx], 
                   self.data_in['angles'][self.idx], 
                   labels,
                   self.class_names,
                   f"Before Resolution (Scene {self.idx}/{self.num_scenes-1}: {scene_id})")
                   
        draw_scene(self.ax2, 
                   self.data_out['translations'][self.idx], 
                   self.data_out['sizes'][self.idx], 
                   self.data_out['angles'][self.idx], 
                   labels,
                   self.class_names,
                   f"After Resolution (Scene {self.idx}/{self.num_scenes-1}: {scene_id})")
        
        self.fig.canvas.draw()

if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("Controls: Use 'Left' and 'Right' arrow keys to switch between scenes.")
    Visualizer(idx)
