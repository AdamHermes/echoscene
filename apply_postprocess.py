import json
import torch
import numpy as np
import os
from helpers.resolve_collision import resolve_bbox_collisions_obb

def apply_postprocess_to_json(input_json_path, output_json_path):
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Copy all fields from the original data (including scene_ids, objectness, etc.)
    out_data = data.copy()
    # We will overwrite translations
    out_data["translations"] = []

    num_scenes = len(data["translations"])
    print(f"Found {num_scenes} scenes. Starting SAT/OBB post-processing...")

    for i in range(num_scenes):
        trans = data["translations"][i]
        sizes = data["sizes"][i]
        angles_rad = data["angles"][i]  # Expected in radians from eval_3dfront.py
        
        # Combine [l,h,w] and [x,y,z] into boxes tensor (N, 6)
        boxes_np = np.concatenate([np.array(sizes), np.array(trans)], axis=-1)
        boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)

        # Convert angles from radians back to degrees because resolve_bbox_collisions_obb expects DEGREES
        angles_np = np.array(angles_rad)
        angles_deg = np.degrees(angles_np)
        angles_tensor = torch.tensor(angles_deg, dtype=torch.float32)

        # Assuming all objects in the JSON should be checked
        # (if you want to skip floor/_scene_, you would generate an objectness_mask here based on class_labels)
        objectness_mask = torch.ones(boxes_tensor.shape[0], dtype=torch.bool)
        
        # Apply SAT/OBB resolution
        print(f"--- Scene {i+1}/{num_scenes} ---")
        boxes_resolved = resolve_bbox_collisions_obb(
            boxes=boxes_tensor, 
            angles_pred=angles_tensor, 
            objectness_mask=objectness_mask, 
            verbose=True
        )

        # Extract the resolved translations (x, y, z are indices 3, 4, 5)
        resolved_trans_np = boxes_resolved[:, 3:6].numpy()
        out_data["translations"].append(resolved_trans_np.tolist())

    print(f"Saving resolved outputs to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(out_data, f)
    print("Done! You can now run your metric evaluations on this new JSON.")

if __name__ == "__main__":
    input_file = 'output/final.json'  # Update if your file is named differently
    output_file = 'output/final_postprocessed.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure the path is correct.")
    else:
        apply_postprocess_to_json(input_file, output_file)
