import os, sys, json
import numpy as np

REPO_ROOT = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene'
sys.path.insert(0, REPO_ROOT)

from scripts.eval_collision import evaluate_furniture_collisions
from scripts.relational.eval_room_types import evaluate_folder_for_room_type, load_3dfront_datasets, locate_physcene_input_json

baseline_models = [
    ('Baseline (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline'),
    ('Baseline (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline_post_processed/vis')
]

front_root = os.path.join(REPO_ROOT, 'FRONT')
ds_dict, scan_maps = load_3dfront_datasets(front_root)

results = []

for name, m_dir in baseline_models:
    json_path = locate_physcene_input_json(m_dir)
    p_dir = os.path.join(os.path.dirname(json_path), 'procthor_scenes')
    
    print(f"\n=========================================================================")
    print(f"  EVALUATING {name}")
    print(f"  Directory: {m_dir}")
    print(f"  JSON Path: {json_path}")
    print(f"=========================================================================")
    
    # 1. Collision metrics
    col_res = evaluate_furniture_collisions(json_path, max_rooms=190)
    
    # 2. Walkability
    w_json = os.path.join(p_dir, 'walkability_results.json')
    walkability = np.nan
    if os.path.exists(w_json):
        with open(w_json, 'r') as f:
            w_data = json.load(f)
            walkability = w_data.get('summary', {}).get('average_walkability', np.nan)
            
    # 3. Navigability / Accessibility
    n_json = os.path.join(p_dir, 'navigation_results.json')
    navigability = np.nan
    if os.path.exists(n_json):
        with open(n_json, 'r') as f:
            n_data = json.load(f)
            navigability = n_data.get('summary', {}).get('average_accessibility_rate', np.nan)
            
    # 4. Relational Accuracy
    rel_res = evaluate_folder_for_room_type(
        folder_path=m_dir,
        room_type='all',
        start_idx=0,
        end_idx=190,
        ds_dict=ds_dict,
        scan_maps=scan_maps,
        front_root=front_root
    )
    
    results.append({
        'name': name,
        'col_obj': col_res['col_obj'],
        'col_obj_cnt': col_res['col_obj_cnt'],
        'tot_obj': col_res['tot_obj'],
        'col_scene': col_res['col_scene'],
        'col_scene_cnt': col_res['col_scene_cnt'],
        'num_scenes': col_res['num_scenes'],
        'walkability': walkability,
        'navigability': navigability,
        'rel_acc': rel_res['total_acc'],
        'rel_mom': rel_res['means_of_means'],
        'rel_details': rel_res
    })

print("\n" + "=" * 90)
print("                       BASELINE RECALCULATION SUMMARY TABLE")
print("=" * 90)
print(f"{'Metric':<40} | {'Baseline (Raw)':<20} | {'Baseline (PP)':<20}")
print("-" * 90)

raw = results[0]
pp = results[1]

print(f"{'Col Obj (Object Collision Rate)':<40} | {raw['col_obj']:.4f} ({raw['col_obj_cnt']}/{raw['tot_obj']}){'':<4} | {pp['col_obj']:.4f} ({pp['col_obj_cnt']}/{pp['tot_obj']})")
print(f"{'Col Scene (Scene Collision Rate)':<40} | {raw['col_scene']:.4f} ({raw['col_scene_cnt']}/{raw['num_scenes']}){'':<4} | {pp['col_scene']:.4f} ({pp['col_scene_cnt']}/{pp['num_scenes']})")
print(f"{'Walkability Score (NavMesh Area %)':<40} | {raw['walkability']*100:.2f}%{'':<13} | {pp['walkability']*100:.2f}%")
print(f"{'Navigability (Object Accessibility %)':<40} | {raw['navigability']*100:.2f}%{'':<13} | {pp['navigability']*100:.2f}%")
print(f"{'Relational Accuracy (Micro Total)':<40} | {raw['rel_acc']*100:.2f}%{'':<13} | {pp['rel_acc']*100:.2f}%")
print(f"{'Relational Accuracy (Macro Means)':<40} | {raw['rel_mom']*100:.2f}%{'':<13} | {pp['rel_mom']*100:.2f}%")
print("=" * 90)
