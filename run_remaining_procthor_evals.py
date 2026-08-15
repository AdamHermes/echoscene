import os, sys, subprocess

REPO_ROOT = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene'
sys.path.insert(0, REPO_ROOT)

target_dirs = [
    ('Baseline (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline/vis/2050/procthor_scenes'),
    ('Work 28 (Raw)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3/2050/procthor_scenes'),
    ('Work 28 (PP)', '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3_pp/vis/2050/procthor_scenes')
]

for name, p_dir in target_dirs:
    print(f"\n=======================================================")
    print(f"Running ProcTHOR evaluations for {name}...")
    print(f"Directory: {p_dir}")
    print(f"=======================================================")
    
    # 1. Walkability
    w_json = os.path.join(p_dir, 'walkability_results.json')
    if not os.path.exists(w_json):
        print(f"Running Walkability on {name}...")
        subprocess.run(['python', 'eval_walkability.py', '--scenes_dir', p_dir], check=True)
    else:
        print(f"Walkability results already exist for {name}.")

    # 2. Navigation / Accessibility
    n_json = os.path.join(p_dir, 'navigation_results.json')
    if not os.path.exists(n_json):
        print(f"Running Navigation/Accessibility on {name}...")
        subprocess.run(['python', 'eval_navigation.py', '--scenes_dir', p_dir], check=True)
    else:
        print(f"Navigation results already exist for {name}.")

print("\nALL REMAINING PROCTHOR EVALUATIONS COMPLETE!")
