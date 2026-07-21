import json
import torch
import numpy as np
import sys
import os

# Ensure helpers module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
        objectness = torch.tensor(data['objectness'][i], dtype=torch.float32)
        
        # Convert angles from radians to degrees as expected by the function
        angles_deg = angles_rad * (180.0 / np.pi)
        
        # boxes shape: (N, 6) -> [l, h, w, x, y, z]
        boxes = torch.cat([sizes, translations], dim=-1)
        
        resolved_boxes = resolve_bbox_collisions_obb(
            boxes=boxes,
            angles_pred=angles_deg,
            objectness_mask=objectness,
            verbose=True
        )
        
        # Extract updated translations [x, y, z] from [l, h, w, x, y, z]
        data['translations'][i] = resolved_boxes[:, 3:].tolist()
        
    with open(output_path, 'w') as f:
        json.dump(data, f)
        
    print(f"Saved resolved bboxes to {output_path}")

if __name__ == '__main__':
    input_file = "physcene_collision_input_all.json"
    output_file = "resolved_physcene_collision_input.json"
    resolve_json(input_file, output_file)
