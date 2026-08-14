import json
import numpy as np
from shapely.geometry import Polygon
import argparse
import os

def evaluate_out_of_bound(json_path, conf_thresh=0.0):
    with open(json_path, "r") as f:
        data = json.load(f)

    tot_obj = 0
    tot_ob = 0
    num_scenes = len(data["class_labels"])

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
        
        # Find floor object
        floor_idx = np.where(max_cls == (num_classes - 2))[0]
        if len(floor_idx) == 0:
            continue
        
        # Assuming one floor per scene, take the first one
        idx_f = floor_idx[0]
        cos_f, sin_f = np.cos(angles[idx_f]), np.sin(angles[idx_f])
        dx_f = np.array([sizes[idx_f][0] / 2, sizes[idx_f][0] / 2, -sizes[idx_f][0] / 2, -sizes[idx_f][0] / 2])
        dz_f = np.array([sizes[idx_f][2] / 2, -sizes[idx_f][2] / 2, -sizes[idx_f][2] / 2, sizes[idx_f][2] / 2])
        floor_poly = Polygon(zip(trans[idx_f][0] + dx_f * cos_f - dz_f * sin_f, trans[idx_f][2] + dx_f * sin_f + dz_f * cos_f))

        valid_furniture = (max_cls != (num_classes - 1)) & (max_cls != (num_classes - 2))
        
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
            # A tolerance can be used, e.g., if intersection area is less than furn_poly area
            intersection = floor_poly.intersection(furn_poly)
            # If the furniture is not fully inside the floor, it is out of bound.
            # Allow a small threshold (e.g. 1e-4)
            if furn_poly.area - intersection.area > 1e-4:
                tot_ob += 1
                
            tot_obj += 1

    avg_ob_per_scene = tot_ob / num_scenes if num_scenes > 0 else 0
    print(f"Total number of scenes: {num_scenes}")
    print(f"Total OB objects: {tot_ob}")
    print(f"Total furniture objects: {tot_obj}")
    print(f"Average #OB per scene: {avg_ob_per_scene}")
    
    return avg_ob_per_scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="../physcene_collision_input_merged.json")
    args = parser.parse_args()
    evaluate_out_of_bound(args.json)
