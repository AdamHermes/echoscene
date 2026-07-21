# Rendering and Dataset Post-Processing Utilities

This document outlines the usage of standalone utility scripts designed to help manage, render, and refine generated EchoScene outputs without needing to rerun the heavy diffusion models.

All rendering scripts are located in the `scripts/rendering/` directory.

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
