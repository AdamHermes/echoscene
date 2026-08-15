# Loss Visualization System in EchoScene

This document provides a guide for generating, interpreting, and re-running loss visualizations in **EchoScene**. The system produces comprehensive 2D spatial figures and loss breakdown scorecards for all physical guidance constraints: **Room Outer Loss**, **Collision Loss**, **Multi-Component Walkable Loss**, and **Relational Guidance Loss**.

---

## 1. Overview of Visualized Losses

| Loss Type | Output Filename | Description |
|---|---|---|
| **Outer Loss** | `{scene_id}_outer_loss.png` | Highlights object bounding boxes that extend past floor plan / room boundary walls (red filled L1 distance overlap). |
| **Collision Loss** | `{scene_id}_collision_loss.png` | Highlights intersecting furniture pairs (red filled 3D IoU overlap regions). |
| **Multi-Component Walkable Loss** | `{scene_id}_walkable_loss.png` | **2×2 Multi-Panel Breakdown** detailing all 4 sub-components of Walkable Guidance Loss. |
| **Relational Guidance Loss** | `{scene_id}_relational_loss.png` | **2-Panel Figure**: (A) 2D Spatial Scene Graph Diagram + (B) Relation Scorecard Table. |

---

## 2. Walkable Guidance Loss (Multi-Component Breakdown)

Walkable Guidance Loss ensures that generated rooms are physically navigable by humans/robots. The combined figure (`{scene_id}_walkable_loss.png`) features a 2×2 grid layout visualizing each sub-component:

1. **(A) Component 1: Reachability & Dijkstra Pathfinding**
   - Renders 256×256 floor grid eroded by agent width ($0.35\text{m}$).
   - Color-codes connected walkable islands (green, teal, yellow, purple).
   - If disconnected islands exist, executes Dijkstra shortest path algorithm connecting disconnected regions through blocked space, displaying the path in **cyan/yellow** with path step penalty boxes.

2. **(B) Component 2: Room Center Penalty**
   - Radial Gaussian heatmap ($e^{-(x^2+z^2)/\sigma}$) centered at room origin $(0, 0)$ with $\sigma=0.5$.
   - Concentric distance rings ($0.5\text{m}, 1.0\text{m}, 1.5\text{m}$) and red origin-to-furniture distance vectors.
   - Pushes central furniture outward to keep the room center open.

3. **(C) Component 3a: Floor Edge-Gaussian Heatmap**
   - Continuous 2D Edge-Gaussian sum radiating from rotated rectangular OBB edges onto the room floor plane.
   - Density map indicating how furniture edges reduce walkable floor clearance.

4. **(D) Component 3b: Pairwise OBB Edge Repulsion**
   - Individual filled 2D Edge-Gaussian fields for each ground-level object.
   - **Cyan/Red pairwise repulsion links** connecting furniture pairs whose edge influence zones overlap ($G_i(x_j, z_j) > 0.02$), annotated with exact penalty badges.

---

## 3. Relational Guidance Loss Visualization

Relational Guidance Loss actively steers objects during diffusion sampling to satisfy scene graph spatial relations (`left`, `right`, `front`, `behind`, `close by`, `standing on`, `above`). The figure (`{scene_id}_relational_loss.png`) features:

1. **(A) 2D Spatial Scene Graph Diagram**:
   - Room floorplan and object OBBs with labels.
   - **Green solid arrows**: Satisfied spatial relations ($\text{loss} = 0$).
   - **Red/Orange dashed arrows**: Violated spatial relations ($\text{loss} > 0$) with penalty callout badges.
   - **Proximity dotted circles**: Distance rings for `close by` relations ($\le 0.45\text{m}$).

2. **(B) Relational Guidance Loss Scorecard & Detailed Table**:
   - Top summary card displaying **Total Relational Guidance Loss**, average loss per relation, and percentage of satisfied vs. violated constraints.
   - Detailed breakdown table listing:
     - `Subject` $\to$ `Predicate` $\to$ `Object`
     - Measured metric vs. target threshold (e.g. $z_s - z_o = +1.85\text{m} \ge +0.05\text{m}$)
     - Status (`OK` Green / `VIOLATED` Red)
     - Computed loss value.

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
