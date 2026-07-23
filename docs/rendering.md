# Rendering and Dataset Post-Processing Utilities

This document outlines the usage of standalone utility scripts designed to help manage, render, and refine generated EchoScene outputs without needing to rerun the heavy diffusion models.

All rendering scripts are located in the `scripts/rendering/` directory (and `scripts/loss_visualization/` for loss diagnostics).

## 1. Rendering Missing GLB Outputs

**Script:** `scripts/rendering/render_missing_glb.py`

Sometimes the main evaluation pipeline (`eval_3dfront.py`) successfully generates `.glb` files for scenes but may timeout or fail when generating the top-down 256x256 `.png` snapshots. These 256x256 renders are absolutely necessary for calculating FID and KID metrics.

This script parses a target directory containing generated scenes, detects which scenes are missing their top-down images in the `render_imgs` directory, and cleanly generates them using `pyrender` and `trimesh`.

**Usage:**
```bash
python scripts/rendering/render_missing_glb.py
```
*Note: For macOS compatibility, this script automatically handles disabling the EGL backend.*

## 2. Removing Lamps from Generated Datasets

**Script:** `scripts/rendering/remove_lamp_from_folders.py`

When evaluating the fidelity of scene layouts using FID and KID metrics, the standard practice (used in common baselines) is to evaluate the rooms without lamps or stools (`without_lamp=True`). If you previously ran the model and generated the dataset with `without_lamp=False`, you can use this script to retroactively strip the lamp geometries from the output datasets.

This script reads the raw base meshes and the corresponding layout JSON files (`physcene_collision_input.json` in both the original and the collision-resolved post-processed dataset folders), strips out any objects corresponding to category ID 7 (lamps), and reconstructs the `.glb` files and `.png` top-down snapshots with perfectly matched scaling, translation, and rotation.

**Behavior:**
- The script preserves the original generated folders.
- It creates two new destination folders:
  - `complete_released_full_model_without_lamp`
  - `complete_released_full_model_post_processed_without_lamp`
- It ensures that the newly created GLB files and image renders exactly match the layout structure of the rooms (just without lamps), making them ready for strict FID/KID evaluations.

**Usage:**
```bash
python scripts/rendering/remove_lamp_from_folders.py
```

## 3. Fast JSON-based Top-Down Visualization & Comparison

**Script:** `scripts/rendering/vis_json_compare.py`

Instead of spinning up heavy `pyrender` or `trimesh` environments to render GLB meshes, this tool directly parses the model's generated `JSON` layout files and plots top-down 2D scenes using `matplotlib`. This is extremely fast and useful for comparing raw outputs vs post-processed outputs side-by-side.

**Features:**
- Highlights the room boundary with a solid red line based on floor constraints.
- Accurately maps class IDs to labels via the original object meshes.
- Supports side-by-side comparison or `--individual` export mode for perfectly tight, label-only image cropping without text titles.
- Can rotate the layout 90-degrees sideways using the `--sideways` flag.
- Excludes lamp visualisations.

**Usage (Single Scene Side-by-Side):**
```bash
python scripts/rendering/vis_json_compare.py \
    --scene_id LivingDiningRoom-2583 \
    --json1 to_be_merged/complete_released_full_model_without_lamp/vis/2050/physcene_collision_input.json \
    --json2 to_be_merged/complete_released_full_model_post_processed_without_lamp/vis/2050/physcene_collision_resolved.json \
    --old_mesh_dir to_be_merged/complete_released_full_model/vis/2050/echoscene/object_meshes \
    --out compare/LivingDiningRoom-2583.png
```

**Usage (All Scenes Individual Mode):**
```bash
python scripts/rendering/vis_json_compare.py \
    --scene_id all \
    --json1 baseline/vis/2050/physcene_collision_input.json \
    --json2 baseline_post_processed/vis/2050/physcene_collision_input.json \
    --old_mesh_dir baseline/vis/2050/echoscene/object_meshes \
    --out_dir compare_baseline \
    --individual
```


## 4. Standalone Image Renderer (from GLB)

**Script:** `scripts/rendering/render_images_only.py`

This script provides standalone capabilities for re-rendering 256x256 `.png` snapshots directly from the generated `.glb` meshes in `render_imgs`. It bypasses the generation loop entirely.

**Features:**
- Can render individual scenes or loop across all scenes.
- Now includes a `--without_lamp` flag. If this flag is provided, the script ignores rendering meshes that correspond to lamps, ensuring the final PNG matches standard FID evaluation constraints.

**Usage:**
```bash
python scripts/rendering/render_images_only.py \
    --in_dir to_be_merged/complete_released_full_model_post_processed_without_lamp/vis/2050 \
    --without_lamp
```


## 5. Physical Guidance Loss Visualizations

**Script:** `scripts/loss_visualization/visualize_losses.py`

This debugging and diagnostic script generates specialized 2D top-down visualizations that visually map exactly how the physical guidance losses (used in the diffusion guidance loop) are computed and applied to the layout bounding boxes.

**Generates three distinct visualizations per scene:**
1. **Outer Boundary Loss (`_outer_loss.png`)**: Measures the L1 distance an object extends beyond the floor boundary. The parts of the objects hanging outside the floor plan are painted with a red alpha mask that dynamically scales based on how far out they protrude.
2. **Collision Loss (`_collision_loss.png`)**: Computes the exact 2D Intersection over Union (IoU) of any overlapping furniture bounding boxes. Overlapping sectors are painted in red, with opacity scaling linearly with the severity of the IoU penetration.
3. **Walkable / Reachability Loss (`_walkable_loss.png`)**: Displays a Gaussian heatmap (`sigma=0.5`) centered at `(0,0)`. The faces of objects caught in the center of the room are dynamically blended with red to signify the reachability penalty assigned to their position.

**Usage:**
```bash
python scripts/loss_visualization/visualize_losses.py \
    --scene_id LivingDiningRoom-2583 \
    --json to_be_merged/complete_released_full_model_without_lamp/vis/2050/physcene_collision_input.json \
    --old_mesh_dir to_be_merged/complete_released_full_model/vis/2050/echoscene/object_meshes \
    --out_dir compare/loss_visualizations
```
