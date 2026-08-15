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

front_root = os.path.join(REPO_ROOT, 'FRONT')
ds_dict, scan_maps = load_3dfront_datasets(front_root)

all_results = []

for name, m_dir in models:
    json_path = locate_physcene_input_json(m_dir)
    p_dir = os.path.join(os.path.dirname(json_path), 'procthor_scenes')
    
    # 1. Collision metrics
    col_res = evaluate_furniture_collisions(json_path, max_rooms=190)
    col_obj = col_res['col_obj'] if col_res else np.nan
    col_scene = col_res['col_scene'] if col_res else np.nan
    
    # 2. Walkability metrics
    w_json = os.path.join(p_dir, 'walkability_results.json')
    walkability = np.nan
    if os.path.exists(w_json):
        with open(w_json, 'r') as f:
            w_data = json.load(f)
            walkability = w_data.get('summary', {}).get('average_walkability', np.nan)
            
    # 3. Navigability / Accessibility metrics
    n_json = os.path.join(p_dir, 'navigation_results.json')
    navigability = np.nan
    if os.path.exists(n_json):
        with open(n_json, 'r') as f:
            n_data = json.load(f)
            navigability = n_data.get('summary', {}).get('average_accessibility_rate', np.nan)
            
    # 4. Relational Accuracy metrics
    rel_res = evaluate_folder_for_room_type(
        folder_path=m_dir,
        room_type='all',
        start_idx=0,
        end_idx=190,
        ds_dict=ds_dict,
        scan_maps=scan_maps,
        front_root=front_root
    )
    
    rel_acc = rel_res['total_acc'] if rel_res else np.nan
    rel_mom = rel_res['means_of_means'] if rel_res else np.nan
    
    all_results.append({
        'name': name,
        'col_obj': col_obj,
        'col_scene': col_scene,
        'walkability': walkability,
        'navigability': navigability,
        'rel_acc': rel_acc,
        'rel_mom': rel_mom,
        'lr': rel_res.get('lr_mean', np.nan) if rel_res else np.nan,
        'fb': rel_res.get('fb_mean', np.nan) if rel_res else np.nan,
        'bism': rel_res.get('bism_mean', np.nan) if rel_res else np.nan,
        'tash': rel_res.get('tash_mean', np.nan) if rel_res else np.nan,
        'stand': rel_res.get('stand_mean', np.nan) if rel_res else np.nan,
        'close': rel_res.get('close_mean', np.nan) if rel_res else np.nan,
        'symm': rel_res.get('symm_mean', np.nan) if rel_res else np.nan
    })

print("\n" + "=" * 125)
print("                                COMPLETE MASTER BENCHMARK EVALUATION REPORT")
print("=" * 125)
header = f"{'Model Name':<25} | {'Col Obj':<8} | {'Col Scene':<10} | {'Walkability':<12} | {'Navigability':<12} | {'RelAcc (Micro)':<15} | {'RelAcc (Macro)':<15}"
print(header)
print("-" * 125)
for r in all_results:
    co = f"{r['col_obj']:.4f}" if not np.isnan(r['col_obj']) else "N/A"
    cs = f"{r['col_scene']:.4f}" if not np.isnan(r['col_scene']) else "N/A"
    wa = f"{r['walkability']*100:.2f}%" if not np.isnan(r['walkability']) else "N/A"
    na = f"{r['navigability']*100:.2f}%" if not np.isnan(r['navigability']) else "N/A"
    ra = f"{r['rel_acc']*100:.2f}%" if not np.isnan(r['rel_acc']) else "N/A"
    rm = f"{r['rel_mom']*100:.2f}%" if not np.isnan(r['rel_mom']) else "N/A"
    
    print(f"{r['name']:<25} | {co:<8} | {cs:<10} | {wa:<12} | {na:<12} | {ra:<15} | {rm:<15}")
print("=" * 125)

# Save JSON report
with open(os.path.join(REPO_ROOT, 'master_benchmark_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
