import json
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


# ==== CẤU HÌNH ====
base_dir = Path("/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged")

in_paths = list(base_dir.rglob("released_full_model_*/vis/2050/physcene_collision_input.json"))
in_paths = [str(p) for p in in_paths]

out_path = base_dir / "physcene_collision_input_merged.json"

for p in in_paths:
    if not Path(p).exists():
        raise FileNotFoundError(f"Input file not found: {p}")

print(f"Found {len(in_paths)} files to merge.")

# QUAN TRỌNG: dedupe_by_scene_id=False vì các run là 5 base khác nhau
# chạy trên cùng tập scene, ta muốn GIỮ LẠI hết, không loại trùng
merged = merge_physcene_jsons(in_paths, dedupe_by_scene_id=False)

with open(out_path, "w") as f:
    json.dump(merged, f)

print(f"\nWrote merged file to: {out_path}")