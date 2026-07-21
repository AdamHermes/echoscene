import json
import os

txt_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/test_rooms_list.txt'
in_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json'
out_file = '/Users/lehoangan/Documents/GitHub/ROOM/echoscene/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input_sorted.json'

with open(txt_file, 'r', encoding='utf-16') as f:
    lines = [l.strip() for l in f if l.strip()]

target_order = []
for line in lines[2:]:
    parts = line.split('|')
    if len(parts) == 2:
        target_order.append(parts[1].strip())

print(f"Target order length: {len(target_order)}")

with open(in_file, 'r') as f:
    data = json.load(f)

print(f"Original JSON scenes count: {len(data['scene_ids'])}")

scene_to_idx = {sid: i for i, sid in enumerate(data['scene_ids'])}

sorted_data = {k: [] for k in data.keys()}

for sid in target_order:
    if sid in scene_to_idx:
        idx = scene_to_idx[sid]
        for k in data.keys():
            sorted_data[k].append(data[k][idx])
    else:
        print(f"Warning: {sid} not found in JSON.")

print(f"Saving sorted JSON to {out_file} (Original file is untouched)")
with open(out_file, 'w') as f:
    json.dump(sorted_data, f)
print("Done!")
