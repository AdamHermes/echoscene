import os
import subprocess
import shutil

folders = [
    "./output/baseline/vis/2050",
    "./output/physcene_guidance/vis/2050",
    "./output/released_full_model/vis/2050"
]

for folder in folders:
    print(f"\n======================================")
    print(f"Processing {folder}...")
    final_json = os.path.join(folder, "final.json")
    out_dir = os.path.join(folder, "procthor_scenes")
    
    # 1. Delete the procthor folders if they exist
    if os.path.exists(out_dir):
        print(f"Deleting existing folder: {out_dir}")
        shutil.rmtree(out_dir)
        
    # 2. Convert echoscene to procthor
    print("Converting scenes to ProcTHOR format...")
    subprocess.run(["python", "scripts/procthor_eval/convert_echoscene_to_procthor.py", "--bbox_path", final_json, "--out_dir", out_dir], check=True)
    
    # 3. Evaluate walkability with AI2-THOR Unity engine
    print("Running Unity walkability evaluation...")
    subprocess.run(["python", "scripts/procthor_eval/eval_walkability.py", "--scenes_dir", out_dir], check=True)
    
print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY!")
