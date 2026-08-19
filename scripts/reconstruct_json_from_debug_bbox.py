#!/usr/bin/env python3
"""
Reconstruct PhyScene JSON from EchoScene debug_bbox.txt

This script parses the human-readable bounding box dump in `debug_bbox.txt`
and reconstructs the official `physcene_collision_input.json` format used
by downstream evaluation scripts (collision evaluation, relational accuracy,
collision resolution / +PP, and ProcTHOR walkability/navigation conversion).

Usage:
    python scripts/reconstruct_json_from_debug_bbox.py \
        --debug_bbox /Volumes/ExternalSSD/current_works/work_XX/extracted/2050/debug_bbox.txt \
        --out_json   /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json

    # Or with room_type (default: 'all')
    python scripts/reconstruct_json_from_debug_bbox.py \
        --debug_bbox path/to/debug_bbox.txt \
        --out_json   path/to/physcene_collision_input.json \
        --dataset_root FRONT/
"""

import os
import sys
import json
import argparse
import numpy as np

# Canonical class mapping for 3D-FRONT room_type='all' with mapping_full2simple
DEFAULT_CLASS_MAPPING = {
    '_scene_': 0,
    'bed': 1,
    'bookshelf': 2,
    'cabinet': 3,
    'chair': 4,
    'desk': 5,
    'floor': 6,
    'lamp': 7,
    'nightstand': 8,
    'shelf': 9,
    'sofa': 10,
    'table': 11,
    'tv_stand': 12,
    'wardrobe': 13,
}

def load_class_mapping(dataset_root=None, room_type="all"):
    """
    Attempt to load class mapping dynamically from FRONT/ dataset files.
    Falls back to DEFAULT_CLASS_MAPPING if files are not accessible.
    """
    if dataset_root and os.path.exists(dataset_root):
        mapping_file = os.path.join(dataset_root, "mapping.json")
        cat_file = os.path.join(dataset_root, f"classes_{room_type}.txt")
        if os.path.exists(mapping_file) and os.path.exists(cat_file):
            try:
                with open(mapping_file, "r") as f:
                    mapping_full2simple = json.load(f)
                with open(cat_file, "r") as f:
                    cats = [line.strip() for line in f if line.strip()]
                simple_cats = sorted(list(set([mapping_full2simple.get(c, c) for c in cats])))
                classes = {name: idx for idx, name in enumerate(simple_cats)}
                return classes
            except Exception as e:
                print(f"[Warning] Failed to dynamically load mapping from {dataset_root}: {e}. Using default.")
    return DEFAULT_CLASS_MAPPING


def parse_debug_bbox_file(debug_bbox_path):
    """
    Parses scenes and objects from debug_bbox.txt.
    Returns list of dicts: [{'scene_id': str, 'num_objects': int, 'objects': list of dicts}]
    """
    if not os.path.exists(debug_bbox_path):
        raise FileNotFoundError(f"debug_bbox file not found at: {debug_bbox_path}")

    with open(debug_bbox_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Split scenes by standard header delimiter
    raw_sections = text.split("============================================================\nSCENE: ")
    if len(raw_sections) <= 1:
        # Fallback split without leading newline
        raw_sections = text.split("SCENE: ")

    parsed_scenes = []

    for section in raw_sections:
        section = section.strip()
        if not section or section.startswith("=") or section.startswith("-"):
            continue

        lines = section.split("\n")
        header_line = lines[0].strip()

        # Parse header: e.g. "MasterBedroom-33296  |  9 objects"
        if "|" in header_line:
            parts = header_line.split("|")
            scene_id = parts[0].strip()
            num_objs_str = parts[1].replace("objects", "").replace("object", "").strip()
            try:
                expected_num_objs = int(num_objs_str)
            except ValueError:
                expected_num_objs = None
        else:
            scene_id = header_line.strip()
            expected_num_objs = None

        # Find object table header
        obj_table_idx = -1
        for idx, line in enumerate(lines):
            line_s = line.strip()
            if line_s.startswith("Obj") and ("l" in line_s) and ("h" in line_s) and ("w" in line_s):
                obj_table_idx = idx
                break

        if obj_table_idx == -1:
            continue

        # Skip separator line (e.g. '---------------------')
        start_row = obj_table_idx + 1
        if start_row < len(lines) and lines[start_row].strip().startswith("-"):
            start_row += 1

        obj_entries = []
        for line in lines[start_row:]:
            line_s = line.strip()
            if not line_s or line_s.startswith("=") or line_s.startswith("Pairwise") or line_s.startswith("!"):
                break
            tokens = line_s.split()
            if len(tokens) >= 8:
                try:
                    name = tokens[0]
                    l = float(tokens[1])
                    h = float(tokens[2])
                    w = float(tokens[3])
                    x = float(tokens[4])
                    y = float(tokens[5])
                    z = float(tokens[6])
                    deg_str = tokens[7].replace("°", "").replace("deg", "")
                    yaw_deg = float(deg_str)
                    obj_entries.append({
                        "name": name,
                        "size": [l, h, w],
                        "translation": [x, y, z],
                        "angle_deg": yaw_deg
                    })
                    if expected_num_objs is not None and len(obj_entries) == expected_num_objs:
                        break
                except ValueError:
                    continue

        if obj_entries:
            parsed_scenes.append({
                "scene_id": scene_id,
                "expected_objects": expected_num_objs,
                "objects": obj_entries
            })

    return parsed_scenes


def build_physcene_json(parsed_scenes, class_mapping):
    """
    Builds the dictionary matching physcene_collision_input.json schema.
    """
    n_classes = len(class_mapping)

    scene_ids = []
    class_labels = []
    translations = []
    sizes = []
    angles = []
    objfeats_32 = []
    objectness = []

    for sc in parsed_scenes:
        sid = sc["scene_id"]
        objs = sc["objects"]

        sc_labels = []
        sc_trans = []
        sc_sizes = []
        sc_angles = []
        sc_objfeats = []
        sc_objness = []

        for obj in objs:
            name = obj["name"]
            l, h, w = obj["size"]
            x, y, z = obj["translation"]
            deg = obj["angle_deg"]
            rad = deg / 180.0 * np.pi

            # One-hot vector with (n_classes + 1) dimensions
            # Last column (index n_classes) is reserved for empty/padding (_scene_, floor)
            one_hot = [0.0] * (n_classes + 1)
            cls_idx = class_mapping.get(name)

            if name in ["_scene_", "floor"] or cls_idx is None:
                one_hot[n_classes] = 1.0
                sc_objness.append([0.0])
            else:
                one_hot[cls_idx] = 1.0
                sc_objness.append([1.0])

            sc_labels.append(one_hot)
            sc_sizes.append([l, h, w])
            sc_trans.append([x, y, z])
            sc_angles.append([rad])
            sc_objfeats.append([0.0] * 32)

        scene_ids.append(sid)
        class_labels.append(sc_labels)
        translations.append(sc_trans)
        sizes.append(sc_sizes)
        angles.append(sc_angles)
        objfeats_32.append(sc_objfeats)
        objectness.append(sc_objness)

    return {
        "class_labels": class_labels,
        "translations": translations,
        "sizes": sizes,
        "angles": angles,
        "objfeats_32": objfeats_32,
        "objectness": objectness,
        "scene_ids": scene_ids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct physcene_collision_input.json from EchoScene debug_bbox.txt"
    )
    parser.add_argument(
        "--debug_bbox",
        type=str,
        required=True,
        help="Path to input debug_bbox.txt file"
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Path to save the reconstructed physcene_collision_input.json file"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="FRONT",
        help="Path to FRONT dataset root directory (for class mappings)"
    )
    parser.add_argument(
        "--room_type",
        type=str,
        default="all",
        help="Room type used during generation (default: 'all')"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Reconstructing PhyScene JSON from debug_bbox.txt")
    print("=" * 70)
    print(f"Input debug_bbox  : {args.debug_bbox}")
    print(f"Output JSON       : {args.out_json}")

    class_mapping = load_class_mapping(args.dataset_root, args.room_type)
    print(f"Loaded {len(class_mapping)} classes: {list(class_mapping.keys())}")

    parsed_scenes = parse_debug_bbox_file(args.debug_bbox)
    print(f"Parsed {len(parsed_scenes)} scene entries from debug_bbox.txt.")

    if not parsed_scenes:
        print("[Error] No valid scenes parsed from debug_bbox.txt!")
        sys.exit(1)

    result_json = build_physcene_json(parsed_scenes, class_mapping)

    # Verification checks
    n_scenes = len(result_json["scene_ids"])
    for key in result_json.keys():
        assert len(result_json[key]) == n_scenes, f"Length mismatch for key {key}"

    total_objects = sum(len(objs) for objs in result_json["sizes"])
    furniture_objects = sum(sum(int(x[0]) for x in objn) for objn in result_json["objectness"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f)

    print("-" * 70)
    print(f"Successfully saved reconstructed JSON to: {args.out_json}")
    print(f"  Total Scenes     : {n_scenes}")
    print(f"  Total BBoxes     : {total_objects}")
    print(f"  Furniture Objects: {furniture_objects}")
    print(f"  Floor / Layout   : {total_objects - furniture_objects}")
    print("=" * 70)


if __name__ == "__main__":
    main()
