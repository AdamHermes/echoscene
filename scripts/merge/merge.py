import json
import shutil
import os
from pathlib import Path

KEYS = ["class_labels", "translations", "sizes", "angles",
        "objfeats_32", "objectness", "scene_ids"]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def merge_physcene_jsons(paths, dedupe_by_scene_id=True):
    merged = {k: [] for k in KEYS}
    seen_scene_ids = set()
    n_skipped_dupes = 0

    for path in paths:
        if not os.path.exists(path):
            continue
        data = load_json(path)
        n_scenes = len(data["scene_ids"])
        for k in KEYS:
            if k not in data:
                raise ValueError(f"{path} is missing key '{k}'")
            if len(data[k]) != n_scenes:
                raise ValueError(
                    f"{path}: key '{k}' has {len(data[k])} entries, "
                    f"but scene_ids has {n_scenes}. File is inconsistent."
                )
        print(f"Loaded {path}: {n_scenes} scenes")

        for i in range(n_scenes):
            scan_id = data["scene_ids"][i]
            if dedupe_by_scene_id and scan_id in seen_scene_ids:
                n_skipped_dupes += 1
                continue
            seen_scene_ids.add(scan_id)
            for k in KEYS:
                merged[k].append(data[k][i])

    print(f"\nMerged total: {len(merged['scene_ids'])} scenes")
    if n_skipped_dupes:
        print(f"Skipped {n_skipped_dupes} duplicate scene_id(s)")
    return merged

def main():
    base_dir = Path("/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged")
    out_dir = base_dir / "complete_released_full_model"
    
    # 1. Merge files by copying
    physcene_paths = []
    final_paths = []
    
    # Sort folders numerically to ensure correct merging order
    def extract_start(f):
        # f.name is like 'released_full_model_50_100'
        try:
            return int(f.name.split('_')[-2])
        except:
            return 0
            
    folders = sorted([f for f in base_dir.glob("released_full_model_*") if f.is_dir()], key=extract_start)
    
    for folder in folders:
        print(f"Processing folder: {folder.name}")
        for root, dirs, files in os.walk(folder):
            rel_path = os.path.relpath(root, folder)
            target_dir = out_dir / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                src_file = Path(root) / file
                target_file = target_dir / file
                
                if rel_path == "vis/2050":
                    if file == "physcene_collision_input.json":
                        physcene_paths.append(str(src_file))
                        continue
                    elif file in ["final.json", "echoscene_collision_summary.json"]:
                        # Skip these redundant files
                        continue
                    elif file.endswith(".json") or file.endswith(".txt"):
                        # Keep only the first occurrence for summary files
                        if not target_file.exists():
                            shutil.copy2(src_file, target_file)
                        continue
                
                # For non-json files or jsons in other folders
                if not target_file.exists():
                    shutil.copy2(src_file, target_file)

    # 2. Merge physcene_collision_input.json
    out_physcene = out_dir / "vis" / "2050" / "physcene_collision_input.json"
    if physcene_paths:
        print("\nMerging physcene_collision_input.json...")
        merged_physcene = merge_physcene_jsons(physcene_paths, dedupe_by_scene_id=False)
        with open(out_physcene, "w") as f:
            json.dump(merged_physcene, f)
        print(f"Wrote merged file to: {out_physcene}")

    # 3. Check rooms
    print("\n--- Room Check ---")
    rooms_list_file = base_dir / "test_rooms_list_utf8.txt"
    if not rooms_list_file.exists():
        os.system(f"iconv -f UTF-16LE -t UTF-8 '{base_dir / 'test_rooms_list.txt'}' > '{rooms_list_file}'")
    
    expected_rooms = []
    try:
        with open(rooms_list_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Find the header row "Index | Name" to determine where to start
            start_idx = 0
            for i, line in enumerate(lines):
                if "Index" in line and "Name" in line:
                    start_idx = i + 2
                    break
                    
            for line in lines[start_idx:]:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        room_name = parts[1].strip()
                        expected_rooms.append(room_name)
    except Exception as e:
        print(f"Could not read test_rooms_list_utf8.txt: {e}")
    
    echoscene_dir = out_dir / "vis" / "2050" / "echoscene"
    if echoscene_dir.exists():
        found_rooms = set()
        for f in os.listdir(echoscene_dir):
            if f.endswith("_echoscene.glb"):
                room_name = f.replace("_echoscene.glb", "")
                found_rooms.add(room_name)
        
        missing = [r for r in expected_rooms if r not in found_rooms]
        print(f"Total expected rooms: {len(expected_rooms)}")
        print(f"Total found rooms: {len(found_rooms)}")
        if missing:
            print(f"Missing {len(missing)} rooms:")
            for m in missing:
                print(f"  - {m}")
        else:
            print("All rooms are present!")
    else:
        print("echoscene directory not found to check rooms.")

if __name__ == "__main__":
    main()
