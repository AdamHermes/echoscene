import os
import json
import argparse
import numpy as np
from shapely.geometry import Polygon

def evaluate_furniture_collisions(json_path, max_rooms=None, conf_thresh=0.0):
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    scene_ids = data.get("scene_ids", [f"scene_{i}" for i in range(len(data.get("class_labels", [])))])
    if max_rooms is not None and max_rooms > 0:
        scene_ids = scene_ids[:max_rooms]

    num_scenes = len(scene_ids)
    tot_obj, col_obj, col_scene = 0, 0, 0
    cat_stats = {}

    for i in range(num_scenes):
        scene_id = scene_ids[i]
        cat = scene_id.split("-")[0].lower() if "-" in scene_id else "all"
        if cat not in cat_stats:
            cat_stats[cat] = {"obj_col": 0, "obj_tot": 0, "scene_col": 0, "scene_tot": 0}
        cat_stats[cat]["scene_tot"] += 1

        trans = np.array(data["translations"][i])
        sizes = np.array(data["sizes"][i])  # Full extents in meters
        angles = np.array(data["angles"][i])
        if angles.ndim == 2:
            angles = angles.squeeze(-1)
        if np.abs(angles).max() > 6.29:
            angles = np.radians(angles)

        classes = np.array(data["class_labels"][i])
        max_cls = np.argmax(classes, axis=-1)

        # Exclude background class (last index) and floor/outer_room class (second last index)
        num_classes = classes.shape[-1]
        valid_furniture = (max_cls != (num_classes - 1)) & (max_cls != (num_classes - 2))

        if "objectness" in data and len(data["objectness"][i]) == len(valid_furniture):
            objness = np.array(data["objectness"][i]).squeeze()
            valid_furniture = valid_furniture & (objness > conf_thresh)

        valid_idx = np.where(valid_furniture)[0]
        n_valid = len(valid_idx)
        tot_obj += n_valid
        cat_stats[cat]["obj_tot"] += n_valid

        if n_valid <= 1:
            continue

        collided = np.zeros(n_valid, dtype=bool)
        for bi in range(n_valid):
            idx_i = valid_idx[bi]
            for bj in range(bi + 1, n_valid):
                idx_j = valid_idx[bj]

                # Vertical height check
                y1_min, y1_max = trans[idx_i][1] - sizes[idx_i][1] / 2.0, trans[idx_i][1] + sizes[idx_i][1] / 2.0
                y2_min, y2_max = trans[idx_j][1] - sizes[idx_j][1] / 2.0, trans[idx_j][1] + sizes[idx_j][1] / 2.0
                if y1_max <= y2_min or y2_max <= y1_min:
                    continue

                # 2D xz footprint check
                cos_i, sin_i = np.cos(angles[idx_i]), np.sin(angles[idx_i])
                dx_i = np.array([sizes[idx_i][0] / 2, sizes[idx_i][0] / 2, -sizes[idx_i][0] / 2, -sizes[idx_i][0] / 2])
                dz_i = np.array([sizes[idx_i][2] / 2, -sizes[idx_i][2] / 2, -sizes[idx_i][2] / 2, sizes[idx_i][2] / 2])
                p1 = Polygon(zip(trans[idx_i][0] + dx_i * cos_i - dz_i * sin_i, trans[idx_i][2] + dz_i * sin_i + dz_i * cos_i))

                cos_j, sin_j = np.cos(angles[idx_j]), np.sin(angles[idx_j])
                dx_j = np.array([sizes[idx_j][0] / 2, sizes[idx_j][0] / 2, -sizes[idx_j][0] / 2, -sizes[idx_j][0] / 2])
                dz_j = np.array([sizes[idx_j][2] / 2, -sizes[idx_j][2] / 2, -sizes[idx_j][2] / 2, sizes[idx_j][2] / 2])
                p2 = Polygon(zip(trans[idx_j][0] + dx_j * cos_j - dz_j * sin_j, trans[idx_j][2] + dz_j * sin_j + dz_j * cos_j))

                if not p1.is_valid or not p2.is_valid:
                    continue

                if p1.intersects(p2) and p1.intersection(p2).area > 1e-4:
                    collided[bi] = True
                    collided[bj] = True

        n_col = collided.sum()
        col_obj += n_col
        cat_stats[cat]["obj_col"] += n_col
        if n_col > 0:
            col_scene += 1
            cat_stats[cat]["scene_col"] += 1

    co_rate = col_obj / tot_obj if tot_obj > 0 else 0.0
    cs_rate = col_scene / num_scenes if num_scenes > 0 else 0.0

    print("=" * 75)
    print(f"          PHYSICAL SCENE COLLISION EVALUATION (Col Scene & Col Obj)          ")
    print("=" * 75)
    print(f" Target File                               : {json_path}")
    print(f" Overall ColObj   (Object Collision Rate) : {co_rate:.4f} ({col_obj}/{tot_obj} objects)")
    print(f" Overall ColScene (Scene Collision Rate)  : {cs_rate:.4f} ({col_scene}/{num_scenes} scenes)")
    print("-" * 75)
    hdr_cat, hdr_co, hdr_cs, hdr_sc = "Category", "ColObj", "ColScene", "Collided Scenes / Total"
    print(f" {hdr_cat:<18} | {hdr_co:<8} | {hdr_cs:<8} | {hdr_sc}")
    print("-" * 75)
    for cat in sorted(cat_stats.keys()):
        st = cat_stats[cat]
        co = st["obj_col"] / st["obj_tot"] if st["obj_tot"] > 0 else 0.0
        cs = st["scene_col"] / st["scene_tot"] if st["scene_tot"] > 0 else 0.0
        col_str = f"{st['scene_col']}/{st['scene_tot']}"
        print(f" {cat:<18} | {co:<8.4f} | {cs:<8.4f} | {col_str}")
    print("=" * 75)

    return {
        "col_obj": co_rate,
        "col_scene": cs_rate,
        "tot_obj": tot_obj,
        "col_obj_cnt": col_obj,
        "num_scenes": num_scenes,
        "col_scene_cnt": col_scene
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ColObj and ColScene for 3D indoor scene predictions.")
    parser.add_argument("--json", type=str, required=True, help="Path to prediction JSON file (e.g. physcene_collision_input.json)")
    parser.add_argument("--max_rooms", type=int, default=190, help="Maximum number of rooms/scenes to evaluate (default: 190, set to 0 for all)")
    args = parser.parse_args()

    max_rooms = None if args.max_rooms <= 0 else args.max_rooms
    evaluate_furniture_collisions(args.json, max_rooms=max_rooms)
