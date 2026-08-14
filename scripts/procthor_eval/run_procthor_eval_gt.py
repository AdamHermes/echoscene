import os
import sys
import json
import math
import argparse
import subprocess
import shutil

# Ensure current script directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from convert_echoscene_to_procthor import convert_to_procthor_json, CLASS_MAPPING

def prepare_gt_procthor_scenes(gt_json_path, out_dir, max_rooms=190, do_collision_resolution=False):
    """
    Converts Ground Truth 3D-FRONT json data to official ProcTHOR scene format.
    Optionally applies SAT collision resolution prior to ProcTHOR scene construction.
    """
    with open(gt_json_path, 'r') as f:
        gt_dict = json.load(f)

    scene_ids = list(gt_dict.keys())
    if max_rooms is not None and max_rooms > 0:
        scene_ids = scene_ids[:max_rooms]

    os.makedirs(out_dir, exist_ok=True)
    
    if do_collision_resolution:
        import torch
        sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../..")))
        from helpers.resolve_collision import resolve_bbox_collisions_obb

    converted_count = 0
    for sid in scene_ids:
        sc = gt_dict[sid]
        furn_objs = []
        floor_obj = None

        for k, obj in sc.items():
            if k == "scene_center" or not isinstance(obj, dict) or "param7" not in obj:
                continue
            p7 = obj["param7"]
            l, h, w, x, y, z, angle_rad = p7

            if obj.get("model_path") is None or h <= 0.01:
                floor_obj = {
                    "class": "floor",
                    "size": {"x": l, "y": h, "z": w},
                    "position": {"x": x, "y": y, "z": z},
                    "rotation": {"x": 0, "y": angle_rad, "z": 0}
                }
                continue

            model_path = obj.get("model_path", "")
            cls_name = "table"
            for k_name in CLASS_MAPPING.keys():
                if k_name in model_path.lower():
                    cls_name = k_name
                    break

            furn_objs.append({
                "class": cls_name,
                "size": [l, h, w],
                "position": [x, y, z],
                "rotation": angle_rad
            })

        if not furn_objs:
            continue

        if do_collision_resolution and len(furn_objs) > 1:
            boxes_np = np.array([[o["size"][0], o["size"][1], o["size"][2], o["position"][0], o["position"][1], o["position"][2]] for o in furn_objs], dtype=np.float32)
            angles_deg = torch.tensor([np.degrees(o["rotation"]) for o in furn_objs], dtype=torch.float32)
            objectness = torch.ones(len(furn_objs), dtype=torch.float32)

            resolved_boxes = resolve_bbox_collisions_obb(
                boxes=torch.tensor(boxes_np),
                angles_pred=angles_deg,
                objectness_mask=objectness,
                class_labels=None,
                max_iter=100,
                push_eps=0.02,
                verbose=False
            ).numpy()

            for idx, o in enumerate(furn_objs):
                o["position"][0] = float(resolved_boxes[idx, 3])
                o["position"][2] = float(resolved_boxes[idx, 5])

        # Format into convert_to_procthor_json expected dictionary
        formatted_objs = [{
            "class": o["class"],
            "size": {"x": o["size"][0], "y": o["size"][1], "z": o["size"][2]},
            "position": {"x": o["position"][0], "y": o["position"][1], "z": o["position"][2]},
            "rotation": {"x": 0, "y": o["rotation"], "z": 0}
        } for o in furn_objs]

        scene_data = {
            "objects": formatted_objs,
            "floor": floor_obj
        }

        house_json = convert_to_procthor_json(sid, scene_data)
        if house_json:
            house_json["objects_raw"] = formatted_objs
            out_file = os.path.join(out_dir, f"{sid}.json")
            with open(out_file, "w") as f:
                json.dump(house_json, f, indent=2)
            converted_count += 1

    print(f"Successfully created {converted_count} ProcTHOR scene JSONs in {out_dir}")
    return out_dir

def run_evaluation_suite(scenes_dir):
    print(f"\n=======================================================")
    print(f"Running ProcTHOR Evaluation on: {scenes_dir}")
    print(f"=======================================================")

    eval_walk_script = os.path.join(SCRIPT_DIR, "eval_walkability.py")
    eval_nav_script = os.path.join(SCRIPT_DIR, "eval_navigation.py")

    # 1. Walkability
    print("\n--- Running Walkability Evaluation ---")
    subprocess.run(["python", eval_walk_script, "--scenes_dir", scenes_dir], check=True)

    # 2. Navigability / Accessibility
    print("\n--- Running Navigability (Accessibility) Evaluation ---")
    subprocess.run(["python", eval_nav_script, "--scenes_dir", scenes_dir], check=True)

    # Read and summarize results
    walk_file = os.path.join(scenes_dir, "walkability_results.json")
    nav_file = os.path.join(scenes_dir, "navigation_results.json")

    avg_walkability = 0.0
    avg_navigability = 0.0

    if os.path.exists(walk_file):
        with open(walk_file, "r") as f:
            w_data = json.load(f)
            avg_walkability = w_data.get("summary", {}).get("average_walkability", 0.0)

    if os.path.exists(nav_file):
        with open(nav_file, "r") as f:
            n_data = json.load(f)
            sum_dict = n_data.get("summary", {})
            avg_navigability = sum_dict.get("average_accessibility_rate", sum_dict.get("overall_accessibility_rate", sum_dict.get("accessibility_rate", 0.0)))


    return avg_walkability, avg_navigability

if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(description="Run Walkability & Navigability on Ground Truth & Ground Truth PP.")
    parser.add_argument("--gt_json", type=str, default="FRONT/obj_boxes_all_test.json", help="Path to ground truth JSON file")
    parser.add_argument("--max_rooms", type=int, default=190, help="Maximum number of rooms to evaluate (default: 190, set 0 for all)")
    args = parser.parse_args()

    max_rooms = None if args.max_rooms <= 0 else args.max_rooms

    gt_path = args.gt_json
    if not os.path.isabs(gt_path):
        gt_path = os.path.abspath(os.path.join(SCRIPT_DIR, "../..", gt_path))

    out_base = os.path.abspath(os.path.join(SCRIPT_DIR, "../../output"))

    # 1. Ground Truth Raw
    gt_dir = os.path.join(out_base, "gt_eval", "procthor_scenes")
    print(f"\n[1/2] Converting Raw Ground Truth to ProcTHOR Scenes...")
    prepare_gt_procthor_scenes(gt_path, gt_dir, max_rooms=max_rooms, do_collision_resolution=False)
    gt_walk, gt_nav = run_evaluation_suite(gt_dir)

    # 2. Ground Truth Post-Processed (Collision Resolved)
    gt_pp_dir = os.path.join(out_base, "gt_pp_eval", "procthor_scenes")
    print(f"\n[2/2] Resolving Collisions and Converting Ground Truth PP to ProcTHOR Scenes...")
    prepare_gt_procthor_scenes(gt_path, gt_pp_dir, max_rooms=max_rooms, do_collision_resolution=True)
    gt_pp_walk, gt_pp_nav = run_evaluation_suite(gt_pp_dir)

    print("\n" + "=" * 75)
    print("      FINAL PROCTHOR EVALUATION RESULTS (GROUND TRUTH & GROUND TRUTH PP)     ")
    print("=" * 75)
    print(f" Target Evaluated Rooms                         : {max_rooms if max_rooms else 'All (370)'}")
    print(f" Ground Truth Raw            - Walkability     : {gt_walk:.4f} ({gt_walk * 100:.2f}%)")
    print(f" Ground Truth Raw            - Navigability    : {gt_nav:.4f} ({gt_nav * 100:.2f}%)")
    print("-" * 75)
    print(f" Ground Truth PP (Resolved)  - Walkability     : {gt_pp_walk:.4f} ({gt_pp_walk * 100:.2f}%)")
    print(f" Ground Truth PP (Resolved)  - Navigability    : {gt_pp_nav:.4f} ({gt_pp_nav * 100:.2f}%)")
    print("=" * 75)


