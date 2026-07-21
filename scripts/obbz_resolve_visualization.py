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

def get_class_names():
    mapping_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT/mapping.json'
    classes_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT/classes_all.txt'
    
    if os.path.exists(mapping_file) and os.path.exists(classes_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        with open(classes_file, 'r') as f:
            vocab = [line.strip() for line in f if line.strip()]
            
        mapped_vocab = [mapping[voc] for voc in vocab if voc in mapping]
        unique_sorted = sorted(list(set(mapped_vocab)))
        return unique_sorted
    return []

def draw_scene(ax, translations, sizes, angles, labels, objectness, class_names, title):
    ax.set_title(title)
    ax.set_aspect('equal')
    
    for i in range(len(translations)):
        corners = get_corners(translations[i], sizes[i], angles[i])
        
        # Determine the label
        idx = -1
        if labels and i < len(labels):
            lbl = labels[i]
            if isinstance(lbl, list) and sum(lbl) > 0:
                idx = int(np.argmax(lbl))
            elif isinstance(lbl, int):
                idx = lbl
                
        obj_mask = 1.0
        if objectness and i < len(objectness):
            obj_mask = objectness[i][0] if isinstance(objectness[i], list) else objectness[i]
        
        is_layout = (idx in [0, 6, 14]) or (obj_mask < 0.5)
        color = 'lightgray' if is_layout else 'lightblue'
        zorder = 0 if is_layout else 10
        
        poly = Polygon(corners, facecolor=color, alpha=0.5, edgecolor='black', linewidth=1.5, zorder=zorder)
        ax.add_patch(poly)
        
        x, y, z = translations[i]
        
        label_text = str(i)
        if is_layout:
            label_text = 'Floor/Layout'
        elif idx >= 0 and class_names and idx < len(class_names):
            label_text = class_names[idx]
        elif idx >= 0:
            label_text = str(idx)
            
        ax.text(x, z, label_text, ha='center', va='center', fontsize=8, zorder=zorder+1)
        
    ax.autoscale_view()

class Visualizer:
    def __init__(self, start_idx=0, input_file=None, output_file=None):
        if input_file is None:
            input_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json'
        if output_file is None:
            output_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json'
        
        with open(input_file, 'r') as f:
            self.data_in = json.load(f)
        
        try:
            with open(output_file, 'r') as f:
                self.data_out = json.load(f)
        except FileNotFoundError:
            self.data_out = self.data_in
            print("Output file not found, displaying input for both plots. Please run solve_obb_collisions.py to generate resolution.")
            
        # Sort data_in
        scene_ids_in = self.data_in.get('scene_ids', [])
        list_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/test_rooms_list_utf8.txt'
        order_map = {}
        if os.path.exists(list_file):
            with open(list_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        order_map[parts[1].strip()] = i
                        
        if len(order_map) > 0:
            sort_indices_in = sorted(range(len(scene_ids_in)), key=lambda i: order_map.get(scene_ids_in[i], 999999))
        else:
            sort_indices_in = sorted(range(len(scene_ids_in)), key=lambda i: scene_ids_in[i])
            
        for k in self.data_in.keys():
            if isinstance(self.data_in[k], list) and len(self.data_in[k]) == len(scene_ids_in):
                self.data_in[k] = [self.data_in[k][i] for i in sort_indices_in]
                
        # Sort data_out INDEPENDENTLY
        scene_ids_out = self.data_out.get('scene_ids', [])
        if len(order_map) > 0:
            sort_indices_out = sorted(range(len(scene_ids_out)), key=lambda i: order_map.get(scene_ids_out[i], 999999))
        else:
            sort_indices_out = sorted(range(len(scene_ids_out)), key=lambda i: scene_ids_out[i])

        for k in self.data_out.keys():
            if isinstance(self.data_out[k], list) and len(self.data_out[k]) == len(scene_ids_out):
                self.data_out[k] = [self.data_out[k][i] for i in sort_indices_out]
            
        self.num_scenes = len(self.data_in['scene_ids'])
        self.idx = start_idx
        
        self.class_names = get_class_names()
        
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
        objectness = self.data_in.get('objectness', [[] for _ in range(self.num_scenes)])[self.idx]
        
        draw_scene(self.ax1, 
                   self.data_in['translations'][self.idx], 
                   self.data_in['sizes'][self.idx], 
                   self.data_in['angles'][self.idx], 
                   labels,
                   objectness,
                   self.class_names,
                   f"Before Resolution (Scene {self.idx}/{self.num_scenes-1}: {scene_id})")
                   
        draw_scene(self.ax2, 
                   self.data_out['translations'][self.idx], 
                   self.data_out['sizes'][self.idx], 
                   self.data_out['angles'][self.idx], 
                   labels,
                   objectness,
                   self.class_names,
                   f"After Resolution (Scene {self.idx}/{self.num_scenes-1}: {scene_id})")
        
        self.fig.canvas.draw()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help='Starting scene index')
    parser.add_argument('--input', type=str, default='/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json')
    parser.add_argument('--output', type=str, default='/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json')
    
    # Allows backwards compatibility for just passing a number as the first argument
    args, unknown = parser.parse_known_args()
    if unknown and unknown[0].isdigit():
        args.idx = int(unknown[0])
        
    print("Controls: Use 'Left' and 'Right' arrow keys to switch between scenes.")
    Visualizer(start_idx=args.idx, input_file=args.input, output_file=args.output)
