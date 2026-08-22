"""
Evaluates Out-of-Bound (OB) objects in generated scenes.
This script checks if generated furniture objects are placed outside of the room boundary.
"""

import json
import numpy as np
from shapely.geometry import Polygon
import argparse
import os

def evaluate_out_of_bound(json_path, conf_thresh=0.0):
    """
    Evaluates the average number of out-of-bound (OB) objects per scene.

    An object is considered out-of-bound if its 2D bounding box footprint 
    is not fully contained within the room's bounding box footprint.

    Args:
        json_path (str): Path to the prediction JSON containing room layouts 
                         (translations, sizes, angles, and class_labels).
        conf_thresh (float): Confidence threshold for objectness (if applicable).

    Returns:
        float: Average number of out-of-bound objects per scene.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    tot_obj = 0
    tot_ob = 0
    num_scenes = len(data["class_labels"])
    processed_scenes = 0

    for i in range(num_scenes):
        trans = np.array(data["translations"][i])
        sizes = np.array(data["sizes"][i])
        angles = np.array(data["angles"][i])
        if angles.ndim == 2:
            angles = angles.squeeze(-1)
        if np.abs(angles).max() > 6.29:
            angles = np.radians(angles)

        classes = np.array(data["class_labels"][i])
        max_cls = np.argmax(classes, axis=-1)

        num_classes = classes.shape[-1]
        
        # Find room/floor object (it's the scene token which is the last class in this dataset)
        floor_idx = np.where(max_cls == (num_classes - 1))[0]
        if len(floor_idx) == 0:
            continue
            
        processed_scenes += 1
        
        # In case there are multiple tokens for the room, take the one with the largest area
        areas = [sizes[idx][0] * sizes[idx][2] for idx in floor_idx]
        idx_f = floor_idx[np.argmax(areas)]
        
        cos_f, sin_f = np.cos(angles[idx_f]), np.sin(angles[idx_f])
        dx_f = np.array([sizes[idx_f][0] / 2, sizes[idx_f][0] / 2, -sizes[idx_f][0] / 2, -sizes[idx_f][0] / 2])
        dz_f = np.array([sizes[idx_f][2] / 2, -sizes[idx_f][2] / 2, -sizes[idx_f][2] / 2, sizes[idx_f][2] / 2])
        floor_poly = Polygon(zip(trans[idx_f][0] + dx_f * cos_f - dz_f * sin_f, trans[idx_f][2] + dx_f * sin_f + dz_f * cos_f))

        # Valid furniture shouldn't include the room token
        valid_furniture = (max_cls != (num_classes - 1))
        
        if "objectness" in data and len(data["objectness"][i]) == len(valid_furniture):
            objness = np.array(data["objectness"][i]).squeeze()
            valid_furniture = valid_furniture & (objness > conf_thresh)

        valid_idx = np.where(valid_furniture)[0]
        
        for idx_i in valid_idx:
            cos_i, sin_i = np.cos(angles[idx_i]), np.sin(angles[idx_i])
            dx_i = np.array([sizes[idx_i][0] / 2, sizes[idx_i][0] / 2, -sizes[idx_i][0] / 2, -sizes[idx_i][0] / 2])
            dz_i = np.array([sizes[idx_i][2] / 2, -sizes[idx_i][2] / 2, -sizes[idx_i][2] / 2, sizes[idx_i][2] / 2])
            furn_poly = Polygon(zip(trans[idx_i][0] + dx_i * cos_i - dz_i * sin_i, trans[idx_i][2] + dx_i * sin_i + dz_i * cos_i))
            
            if not furn_poly.is_valid:
                continue

            # Check if out of bound: not entirely inside the floor
            intersection = floor_poly.intersection(furn_poly)
            # Allow a small threshold (e.g. 1e-4)
            if furn_poly.area - intersection.area > 1e-4:
                tot_ob += 1
                
            tot_obj += 1

    avg_ob_per_scene = tot_ob / processed_scenes if processed_scenes > 0 else 0
    print(f"Total number of scenes: {num_scenes} (processed {processed_scenes})")
    print(f"Total OB objects: {tot_ob}")
    print(f"Total furniture objects: {tot_obj}")
    print(f"Average #OB per scene: {avg_ob_per_scene}")
    
    return avg_ob_per_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate out-of-bound (OB) objects in generated room layouts.")
    parser.add_argument("--json", type=str, default="../physcene_collision_input_merged.json",
                        help="Path to the JSON file containing the room layout predictions.")
    args = parser.parse_args()
    evaluate_out_of_bound(args.json)
