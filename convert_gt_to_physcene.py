import json
import os
import numpy as np

def convert_gt_to_physcene_format(
    box_json_path="FRONT/obj_boxes_all_test.json",
    rel_json_path="FRONT/relationships_all_test.json",
    mapping_path="FRONT/mapping.json",
    output_path="gt_physcene_input.json"
):
    print(f"Loading GT boxes from {box_json_path}...")
    with open(box_json_path, "r") as f:
        box_data = json.load(f)
        
    print(f"Loading relationships (for labels) from {rel_json_path}...")
    with open(rel_json_path, "r") as f:
        rel_data = json.load(f)
        
    print(f"Loading class mapping from {mapping_path}...")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        
    # Get unique coarse classes and sort them to create consistent one-hot indices
    unique_classes = sorted(list(set(mapping.values())))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    print(f"Found {num_classes} unique classes: {unique_classes}")

    # Build a lookup for scan_id -> {obj_id: fine_grained_class}
    scan_to_labels = {}
    for scan in rel_data["scans"]:
        scan_id = scan["scan"]
        scan_to_labels[scan_id] = scan["objects"]

    # Output structure
    output = {
        "class_labels": [],
        "translations": [],
        "sizes": [],
        "angles": [],
        "objfeats_32": [],
        "objectness": [],
        "scene_ids": []
    }

    print("Converting scenes...")
    for scan_id, scene_info in box_data.items():
        if scan_id == "scene_center":
            continue
            
        scene_center = np.array(scene_info.get("scene_center", [0, 0, 0]))
        
        c_labels = []
        c_trans = []
        c_sizes = []
        c_angles = []
        c_feats = []
        c_obj = []
        c_scene_ids = []

        labels_for_scene = scan_to_labels.get(scan_id, {})
        
        # We must add the `_scene_` token first if following the typical diffusion format, 
        # but for physcene it iterates over whatever is present.
        for obj_id_str, obj_data in scene_info.items():
            if obj_id_str == "scene_center":
                continue
                
            # Get label
            fine_label = labels_for_scene.get(obj_id_str, "table") # fallback
            coarse_label = mapping.get(fine_label, fine_label)
            class_idx = class_to_idx.get(coarse_label, 0)
            
            # One-hot encode
            one_hot = [0.0] * num_classes
            one_hot[class_idx] = 1.0
            
            # Extract param7
            param7 = obj_data["param7"]
            l, h, w, cx, cy, cz, angle = param7
            
            # Center translations
            cx -= scene_center[0]
            cy -= scene_center[1]
            cz -= scene_center[2]
            
            # Objectness: 0 for floor and _scene_, 1 for actual objects
            is_object = 0.0 if coarse_label in ["floor", "_scene_"] else 1.0
            
            c_labels.append(one_hot)
            c_trans.append([cx, cy, cz])
            c_sizes.append([l, h, w])
            c_angles.append([angle])
            c_feats.append([0.0] * 32) # Dummy features since it's GT
            c_obj.append([is_object])
            c_scene_ids.append(scan_id)
            
        output["class_labels"].append(c_labels)
        output["translations"].append(c_trans)
        output["sizes"].append(c_sizes)
        output["angles"].append(c_angles)
        output["objfeats_32"].append(c_feats)
        output["objectness"].append(c_obj)
        output["scene_ids"].append(scan_id)

    print(f"Processed {len(output['scene_ids'])} scenes.")
    
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    convert_gt_to_physcene_format()
