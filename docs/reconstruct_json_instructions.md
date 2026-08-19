# Reconstructing Output JSON from `debug_bbox.txt`

## Overview

When an evaluation zip is missing `physcene_collision_input.json` (for example, if generation was partially stopped or only `debug_bbox.txt` / rendered meshes were archived), you can reconstruct the official JSON using:

```bash
scripts/reconstruct_json_from_debug_bbox.py
```

This script parses the human-readable bounding box dump in `debug_bbox.txt` and reconstructs the exact `physcene_collision_input.json` schema needed by all downstream benchmarks (Collision, Relational Accuracy, Collision Resolution / +PP, and ProcTHOR Walkability/Navigation).

---

## Usage

From the `echoscene/` root:

```bash
python scripts/reconstruct_json_from_debug_bbox.py \
    --debug_bbox /Volumes/ExternalSSD/current_works/work_XX/extracted/2050/debug_bbox.txt \
    --out_json   /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json
```

### Optional Arguments

- `--dataset_root`: Path to the `FRONT/` dataset folder (default: `FRONT`) used to dynamically load class vocabulary mappings.
- `--room_type`: Room type used during generation (default: `all`).

---

## What the Reconstructed JSON Contains

The output JSON strictly conforms to the PhyScene format produced by `eval_3dfront.py`:

| Key | Description | Type / Shape |
|---|---|---|
| `scene_ids` | Scene identifiers (e.g. `MasterBedroom-33296`) | `List[str]` |
| `class_labels` | 15-class one-hot vectors (14 object classes + padding column at index 14 for `_scene_` and `floor`) | `(N, 15)` list of floats |
| `objectness` | Object confidence (`1.0` for furniture, `0.0` for floor/`_scene_`) | `(N, 1)` list of floats |
| `sizes` | Bounding box dimensions `[length, height, width]` in meters | `(N, 3)` list of floats |
| `translations` | Bounding box center coordinates `[x, y, z]` in meters | `(N, 3)` list of floats |
| `angles` | Yaw rotation in **radians** (converted from degrees) | `(N, 1)` list of floats |
| `objfeats_32` | 32-dim feature vectors (zeros) | `(N, 32)` list of floats |

---

## Downstream Workflow

Once reconstructed, proceed with the standard evaluation pipeline:

1. **Collision Resolution (+PP)**:
   ```bash
   python scripts/collision/resolve_collision_json.py \
       --in_file  /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json \
       --out_file /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_resolved.json
   ```

2. **Collision Evaluation**:
   ```bash
   python scripts/eval_collision.py --json /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json --max_rooms 0
   python scripts/eval_collision.py --json /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_resolved.json --max_rooms 0
   ```

3. **Relational Accuracy**:
   ```bash
   python scripts/relational/evaluate_relational_accuracy.py --json /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json
   python scripts/relational/evaluate_relational_accuracy.py --json /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_resolved.json
   ```

4. **ProcTHOR Conversion & Evaluation**:
   ```bash
   python scripts/procthor_eval/convert_echoscene_to_procthor.py --full \
       --bbox_path /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_input.json \
       --out_dir   /Volumes/ExternalSSD/current_works/work_XX/2050/procthor_scenes_RAW

   python scripts/procthor_eval/convert_echoscene_to_procthor.py --full \
       --bbox_path /Volumes/ExternalSSD/current_works/work_XX/2050/physcene_collision_resolved.json \
       --out_dir   /Volumes/ExternalSSD/current_works/work_XX/2050/procthor_scenes_PP
   ```
