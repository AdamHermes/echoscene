import os, sys, json
import numpy as np

REPO_ROOT = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene'
sys.path.insert(0, REPO_ROOT)

from scripts.eval_collision import evaluate_furniture_collisions
from scripts.relational.eval_room_types import evaluate_folder_for_room_type, load_3dfront_datasets, locate_physcene_input_json

models = [
    ('Current Best Sig (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/to_be_merged/complete_released_full_model'),
    ('Current Best Sig (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/to_be_merged/complete_released_full_model_post_processed'),
    ('Baseline (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline'),
    ('Baseline (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline_post_processed/vis'),
    ('Work 28 (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3'),
    ('Work 28 (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3_pp'),
    ('Work 27 (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/real_num27'),
    ('Work 27 (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/real_num27_pp')
]

print("Loading 3D-FRONT ground truth datasets...")
front_root = os.path.join(REPO_ROOT, 'FRONT')
ds_dict, scan_maps = load_3dfront_datasets(front_root)

summary_data = []

for name, m_dir in models:
    json_path = locate_physcene_input_json(m_dir)
    print(f"\nEvaluating {name}...")
    print(f"JSON path: {json_path}")
    
    # 1. Collision Evaluation
    col_res = evaluate_furniture_collisions(json_path, max_rooms=190)
    col_obj = col_res['col_obj'] if col_res else 0.0
    col_scene = col_res['col_scene'] if col_res else 0.0
    
    # 2. Relational Accuracy Evaluation
    rel_res = evaluate_folder_for_room_type(
        folder_path=m_dir,
        room_type='all',
        start_idx=0,
        end_idx=190,
        ds_dict=ds_dict,
        scan_maps=scan_maps,
        front_root=front_root
    )
    rel_acc = rel_res['total_acc'] if rel_res else 0.0
    rel_mom = rel_res['means_of_means'] if rel_res else 0.0
    
    summary_data.append({
        'name': name,
        'col_obj': col_obj,
        'col_scene': col_scene,
        'rel_acc': rel_acc,
        'rel_mom': rel_mom
    })

print("\n" + "=" * 80)
print("INTERMEDIATE EVALUATION SUMMARY (Collision & Relational Accuracy)")
print("=" * 80)
print(f"{'Model Name':<25} | {'Col Obj':<8} | {'Col Scene':<10} | {'Rel Acc (Micro)':<15} | {'Rel Acc (Macro)':<15}")
print("-" * 80)
for row in summary_data:
    print(f"{row['name']:<25} | {row['col_obj']:<8.4f} | {row['col_scene']:<10.4f} | {row['rel_acc']*100:<15.2f}% | {row['rel_mom']*100:<15.2f}%")
print("=" * 80)
