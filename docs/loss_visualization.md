# Loss Visualization System in EchoScene

This document provides a guide for generating, interpreting, and re-running loss visualizations in **EchoScene**. The system produces comprehensive 2D spatial figures and loss breakdown scorecards for all physical guidance constraints: **Room Outer Loss**, **Collision Loss**, **Multi-Component Walkable Loss**, and **Relational Guidance Loss**.

---

## 1. Overview of Visualized Losses

| Loss Type | Output Filename | Description |
|---|---|---|
| **Outer Loss** | `{scene_id}_outer_loss.png` | Highlights object bounding boxes that extend past floor plan / room boundary walls (red filled L1 distance overlap). |
| **Collision Loss** | `{scene_id}_collision_loss.png` | Highlights intersecting furniture pairs (red filled 3D IoU overlap regions). |
| **Edge-Gaussian Walkable Loss** | `{scene_id}_walkable_loss.png` / `{scene_id}_walkable_loss_edge_gaussian.png` | Standalone figure showing pure 2D continuous Edge-Gaussian density field radiating from object boundaries across the floor. |
| **Object Gaussian Walkable Loss** | `{scene_id}_walkable_loss_object_gaussian.png` | Standalone figure showing oriented 2D Gaussian density fields centered at each object bounding box location (`gausv1`). |
| **Center Penalty Walkable Loss** | `{scene_id}_walkable_loss_center_penalty.png` | Standalone figure showing pure radial Gaussian density field centered at room origin $(0, 0)$. |
| **Pathfinding Walkable Loss** | `{scene_id}_walkable_loss_pathfinding.png` | Standalone figure showing 2D free-space reachability map. |
| **Relational Guidance Loss** | `{scene_id}_relational_loss.png` | Standalone 2D Spatial Scene Graph Diagram showing directional spatial constraints on room layout. |

---

## 2. Walkable Guidance Loss (Standalone Individual Visualizations)

Each loss component is rendered in its own dedicated, standalone image without visual clutter (no dot-connecting lines or extraneous circles):

1. **Edge-Gaussian Walkable Loss (`{scene_id}_walkable_loss.png` / `{scene_id}_walkable_loss_edge_gaussian.png`)**
   - Continuous 2D Edge-Gaussian sum radiating directly from rotated rectangular OBB edges onto the room floor plane.
   - Smooth density map indicating how furniture proximity impacts walkable floor clearance.

2. **Center Penalty Walkable Loss (`{scene_id}_walkable_loss_center_penalty.png`)**
   - Pure radial Gaussian heatmap ($e^{-(x^2+z^2)/\sigma}$) centered at room origin $(0, 0)$ with $\sigma=0.5$.
   - Pushes central furniture outward to keep the room center open.

3. **Pathfinding Walkable Loss (`{scene_id}_walkable_loss_pathfinding.png`)**
   - Renders floor grid eroded by agent width ($0.35\text{m}$).
   - Color-codes connected walkable free-space components.

---

## 3. Relational Guidance Loss Visualization

Relational Guidance Loss actively steers objects during diffusion sampling to satisfy scene graph spatial relations (`left`, `right`, `front`, `behind`, `close by`, `standing on`, `above`). The figure (`{scene_id}_relational_loss.png`) features:

- Room floorplan and object OBBs with labels.
- **Green solid arrows**: Satisfied spatial relations ($\text{loss} = 0$).
- **Red dashed arrows**: Violated spatial relations ($\text{loss} > 0$) with penalty callout badges.

---

## 4. How to Re-Run the Visualizations

### Method 1: Easy Shell Script Runner

From the `echoscene` directory, run the launcher script:

```bash
cd /Users/lehoangan/Documents/GitHub/ROOM/echoscene
./scripts/loss_visualization/run_loss_vis.sh LivingDiningRoom-2583
```

This automatically runs the pipeline for both:
- **Input Scene Layout**: saved in `loss_vis_2583_input/`
- **Resolved Scene Layout**: saved in `loss_vis_2583/`

### Method 2: Direct Python Script Execution

You can run `visualize_losses.py` directly with custom parameters:

```bash
cd /Users/lehoangan/Documents/GitHub/ROOM/echoscene

# For Input Scene:
python3 scripts/loss_visualization/visualize_losses.py \
  --scene_id LivingDiningRoom-2583 \
  --json physcene_collision_input_merged.json \
  --old_mesh_dir baseline/vis/2050/echoscene/object_meshes \
  --out_dir loss_vis_2583_input

# For Resolved Scene:
python3 scripts/loss_visualization/visualize_losses.py \
  --scene_id LivingDiningRoom-2583 \
  --json baseline/physcene_collision_resolved.json \
  --old_mesh_dir baseline/vis/2050/echoscene/object_meshes \
  --out_dir loss_vis_2583
```

---

## 5. File & Code Architecture

- **Main Visualization Python Script**: [`scripts/loss_visualization/visualize_losses.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/loss_visualization/visualize_losses.py)
- **Shell Runner Script**: [`scripts/loss_visualization/run_loss_vis.sh`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/loss_visualization/run_loss_vis.sh)
- **Loss Computation Backend**: [`model/networks/diffusion_layout/physical_guidance.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/model/networks/diffusion_layout/physical_guidance.py)
