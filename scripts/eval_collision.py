import os
import json
import argparse
import numpy as np
from shapely.geometry import Polygon

def footprint(t, s, a, legacy=False, yaw_sign=1.0):
    """xz footprint polygon of a box with full extents s, centre t, yaw a."""
    a = yaw_sign * a
    c, si = np.cos(a), np.sin(a)
    dx = np.array([s[0] / 2, s[0] / 2, -s[0] / 2, -s[0] / 2])
    dz = np.array([s[2] / 2, -s[2] / 2, -s[2] / 2, s[2] / 2])
    if legacy:  # pre-2026-10-18 typo: dz*sin instead of dx*sin
        return Polygon(zip(t[0] + dx * c - dz * si, t[2] + dz * si + dz * c))
    return Polygon(zip(t[0] + dx * c - dz * si, t[2] + dx * si + dz * c))


def evaluate_furniture_collisions(json_path, max_rooms=None, conf_thresh=0.0, legacy=False, yaw_sign=1.0):
    """legacy=True reproduces the pre-2026-10-18 numbers of the prediction-JSON path, which
    (1) excluded class index num_classes-2 (= wardrobe in the 15-class encoding, not a floor
    class), (2) treated translation y as the box centre (it is the bottom), and (3) built the
    footprint z coordinate as z + dz*sin + dz*cos instead of z + dx*sin + dz*cos.
    yaw_sign=-1 mirrors the yaw (see docs/navigability_v2.md, "Orientation fix")."""
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    tot_obj, col_obj, col_scene = 0, 0, 0
    cat_stats = {}

    # Detect if file is Ground Truth format (dict mapping scene_id -> obj_dict)
    is_gt_format = False
    if isinstance(data, dict) and "scene_ids" not in data and "class_labels" not in data:
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            is_gt_format = True

    if is_gt_format:
        scene_ids = list(data.keys())
        if max_rooms is not None and max_rooms > 0:
            scene_ids = scene_ids[:max_rooms]
        num_scenes = len(scene_ids)

        for scene_id in scene_ids:
            sc = data[scene_id]
            cat = scene_id.split("-")[0].lower() if "-" in scene_id else "all"
            if cat not in cat_stats:
                cat_stats[cat] = {"obj_col": 0, "obj_tot": 0, "scene_col": 0, "scene_tot": 0}
            cat_stats[cat]["scene_tot"] += 1

            furn_boxes = []
            for k, obj in sc.items():
                if k == "scene_center" or not isinstance(obj, dict) or "param7" not in obj:
                    continue
                p7 = obj["param7"]
                if obj.get("model_path") is None or p7[1] <= 0.01:
                    continue
                furn_boxes.append(p7)

            n_valid = len(furn_boxes)
            tot_obj += n_valid
            cat_stats[cat]["obj_tot"] += n_valid

            if n_valid <= 1:
                continue

            collided = np.zeros(n_valid, dtype=bool)
            for bi in range(n_valid):
                l_i, h_i, w_i, x_i, y_i, z_i, ang_i = furn_boxes[bi]
                for bj in range(bi + 1, n_valid):
                    l_j, h_j, w_j, x_j, y_j, z_j, ang_j = furn_boxes[bj]

                    if (y_i + h_i) <= y_j or (y_j + h_j) <= y_i:
                        continue

                    cos_i, sin_i = np.cos(ang_i), np.sin(ang_i)
                    dx_i = np.array([l_i / 2, l_i / 2, -l_i / 2, -l_i / 2])
                    dz_i = np.array([w_i / 2, -w_i / 2, -w_i / 2, w_i / 2])
                    p1 = Polygon(zip(x_i + dx_i * cos_i - dz_i * sin_i, z_i + dx_i * sin_i + dz_i * cos_i))

                    cos_j, sin_j = np.cos(ang_j), np.sin(ang_j)
                    dx_j = np.array([l_j / 2, l_j / 2, -l_j / 2, -l_j / 2])
                    dz_j = np.array([w_j / 2, -w_j / 2, -w_j / 2, w_j / 2])
                    p2 = Polygon(zip(x_j + dx_j * cos_j - dz_j * sin_j, z_j + dx_j * sin_j + dz_j * cos_j))

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

    else:
        # Standard prediction JSON format
        scene_ids = data.get("scene_ids", [f"scene_{i}" for i in range(len(data.get("class_labels", [])))])
        if max_rooms is not None and max_rooms > 0:
            scene_ids = scene_ids[:max_rooms]
        num_scenes = len(scene_ids)

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

            num_classes = classes.shape[-1]
            if legacy:
                valid_furniture = (max_cls != (num_classes - 1)) & (max_cls != (num_classes - 2))
            else:
                # Last index = floor (and the tiny '_scene_' token); num_classes-2 is the wardrobe
                valid_furniture = max_cls != (num_classes - 1)

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
                    if legacy:
                        y1_min, y1_max = trans[idx_i][1] - sizes[idx_i][1] / 2.0, trans[idx_i][1] + sizes[idx_i][1] / 2.0
                        y2_min, y2_max = trans[idx_j][1] - sizes[idx_j][1] / 2.0, trans[idx_j][1] + sizes[idx_j][1] / 2.0
                    else:  # translation y is the bottom of the box
                        y1_min, y1_max = trans[idx_i][1], trans[idx_i][1] + sizes[idx_i][1]
                        y2_min, y2_max = trans[idx_j][1], trans[idx_j][1] + sizes[idx_j][1]
                    if y1_max <= y2_min or y2_max <= y1_min:
                        continue

                    # 2D xz footprint check
                    p1 = footprint(trans[idx_i], sizes[idx_i], angles[idx_i], legacy, yaw_sign)
                    p2 = footprint(trans[idx_j], sizes[idx_j], angles[idx_j], legacy, yaw_sign)

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
        "col_scene_cnt": col_scene,
        "categories": cat_stats
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ColObj and ColScene for 3D indoor scene predictions or Ground Truth.")
    parser.add_argument("--json", type=str, required=True, help="Path to prediction or GT JSON file")
    parser.add_argument("--max_rooms", type=int, default=190, help="Maximum number of rooms/scenes to evaluate (default: 190, set to 0 for all)")
    parser.add_argument("--legacy", action="store_true",
                        help="Reproduce pre-2026-10-18 numbers (wardrobes excluded, y as centre, footprint typo)")
    parser.add_argument("--yaw_sign", type=float, default=1.0,
                        help="-1 mirrors the yaw (orientation fix, see docs/navigability_v2.md)")
    args = parser.parse_args()

    max_rooms = None if args.max_rooms <= 0 else args.max_rooms
    evaluate_furniture_collisions(args.json, max_rooms=max_rooms, legacy=args.legacy, yaw_sign=args.yaw_sign)
