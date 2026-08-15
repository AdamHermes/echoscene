#!/bin/bash
# Runner script for Loss Visualizations (Outer, Collision, Walkable Multi-Component, Relational)

SCENE_ID=${1:-"LivingDiningRoom-2583"}
INPUT_JSON=${2:-"physcene_collision_input_merged.json"}
RESOLVED_JSON=${3:-"baseline/physcene_collision_resolved.json"}
MESH_DIR=${4:-"baseline/vis/2050/echoscene/object_meshes"}

echo "=========================================================="
echo "Generating Loss Visualizations for Scene: $SCENE_ID"
echo "=========================================================="

# 1. Generate Input Scene Loss Visualizations
echo "Generating Input Scene Loss Visualizations -> loss_vis_2583_input..."
python3 scripts/loss_visualization/visualize_losses.py \
  --scene_id "$SCENE_ID" \
  --json "$INPUT_JSON" \
  --old_mesh_dir "$MESH_DIR" \
  --out_dir "loss_vis_2583_input"

# 2. Generate Resolved Scene Loss Visualizations
echo "Generating Resolved Scene Loss Visualizations -> loss_vis_2583..."
python3 scripts/loss_visualization/visualize_losses.py \
  --scene_id "$SCENE_ID" \
  --json "$RESOLVED_JSON" \
  --old_mesh_dir "$MESH_DIR" \
  --out_dir "loss_vis_2583"

echo "=========================================================="
echo "Done! Generated loss visualizations in:"
echo "  - loss_vis_2583_input"
echo "  - loss_vis_2583"
echo "=========================================================="
