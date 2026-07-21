import json
import torch
import numpy as np
import sys
import os

# Ensure helpers module can be imported
sys.path.append('/Users/lehoangan/Documents/GitHub/ROOM/echoscene')
from helpers.resolve_collision import resolve_bbox_collisions_obb

def resolve_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    num_scenes = len(data['scene_ids'])
    print(f"Processing {num_scenes} scenes from {input_path}...")
    
    for i in range(num_scenes):
        print(f"Resolving scene {i+1}/{num_scenes}: {data['scene_ids'][i]}")
        
        sizes = torch.tensor(data['sizes'][i], dtype=torch.float32)
        translations = torch.tensor(data['translations'][i], dtype=torch.float32)
        angles_rad = torch.tensor(data['angles'][i], dtype=torch.float32)
        
        # Angles in JSON are radians, solver expects degrees
        angles_deg = angles_rad * (180.0 / np.pi)
        
        objectness = torch.tensor(data['objectness'][i], dtype=torch.float32)
        class_labels = np.array(data['class_labels'][i])
        
        boxes = torch.cat([sizes, translations], dim=-1)
        
        resolved_boxes = resolve_bbox_collisions_obb(
            boxes=boxes,
            angles_pred=angles_deg,
            objectness_mask=objectness,
            class_labels=class_labels,
            max_iter=500,
            push_eps=0.02,
            verbose=True
        )
        
        # Extract updated translations [x, y, z] from [l, h, w, x, y, z]
        data['translations'][i] = resolved_boxes[:, 3:].tolist()
        
    with open(output_path, 'w') as f:
        json.dump(data, f)
        
    print(f"Saved resolved bboxes to {output_path}")

if __name__ == '__main__':
    input_file = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input_sorted.json"
    output_file = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json"
    resolve_json(input_file, output_file)
