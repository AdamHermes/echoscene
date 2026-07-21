# Scene Collision Resolution Pipeline

This guide explains how to properly resolve physically impossible object collisions and layout bounding constraints in generated 3D scenes. The resolution uses the official PyTorch optimization loops with injected geometric rules for absolute layout containment and specific furniture exemption (like lamps).

## Prerequisites

All operations happen within the `echoscene/` root folder. 
Make sure you have all your input JSON data in the appropriate directories. The following steps assume your input JSON is located at: `to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json`.

## Step 1: Data Alignment and Sorting

Before any collisions can be resolved or accurately visualized side-by-side, the raw scene data must be identically aligned to the validation list order (`test_rooms_list_utf8.txt`). 

Run the sort script to produce an aligned input file:
```bash
cd scripts
python sort_json.py
```
**Output**: `to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input_sorted.json`

*(Note: Never modify the original `physcene_collision_input.json` in place.)*

## Step 2: PyTorch Optimization Resolution

The core collision solver is executed by `resolve_collision_json.py`, which heavily utilizes the PyTorch-based algorithm inside `helpers/resolve_collision.py`.

**Custom rules applied during the resolution loop:**
1. **Layout Containment (No Oscillations)**: The algorithm finds the single largest `Floor/Layout` footprint in each scene (preventing disjoint layout oscillations) and perfectly mathematically restricts all objects inside of it. If an object is placed outside the floor boundaries, the solver slides it precisely back within the room perimeter.
2. **Lamp Exception (Idx 7)**: Lamps are uniquely skipped during the object-to-object pushing logic, allowing them to hang near or above other furniture without forcefully knocking everything else away.
3. **Vertical Checking**: Standard vertical alignment checks ensure objects above or below one another don't falsely register as collisions in the top-down 2D SAT matrix.

Run the resolution process (can take a minute or two to converge over all iterations):
```bash
cd ..
python resolve_collision_json.py
```
**Output**: `to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json`

## Step 3: Visual Validation

Once the output is generated, you can visually compare the original scrambled/colliding scenes against the freshly optimized ones. 

The visualizer script creates a side-by-side Matplotlib plot. It correctly colors layouts in gray (whether they are strictly index 14, index 0, or just have low objectness) and colors furniture bounding boxes in blue. It automatically sorts the raw inputs dynamically in memory so the left and right screens perfectly match up scene-by-scene.

```bash
cd scripts
python obbz_resolve_visualization.py
```

*Optional: To jump straight to a specific scene index (e.g. index 3), just pass the number:*
```bash
python obbz_resolve_visualization.py 3
```
