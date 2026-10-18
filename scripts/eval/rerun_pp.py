"""Re-run the trunk Collision Resolver on a raw layout JSON with fixed settings.

Every run is post-processed by the same code (echoscene/helpers/resolve_collision.py,
max_iter=500, push_eps=0.02), so PP-displacement comparisons between runs are controlled.

usage: python rerun_pp.py <raw physcene_collision_input.json> <out resolved.json>
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from helpers.resolve_collision import resolve_bbox_collisions_obb

src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
t0 = time.time()
for i in range(len(data['scene_ids'])):
    sizes = torch.tensor(data['sizes'][i], dtype=torch.float32)
    trans = torch.tensor(data['translations'][i], dtype=torch.float32)
    angles_deg = torch.tensor(data['angles'][i], dtype=torch.float32) * (180.0 / np.pi)
    resolved = resolve_bbox_collisions_obb(
        boxes=torch.cat([sizes, trans], dim=-1),
        angles_pred=angles_deg,
        objectness_mask=torch.tensor(data['objectness'][i], dtype=torch.float32),
        class_labels=np.array(data['class_labels'][i]),
        max_iter=500, push_eps=0.02, verbose=False)
    data['translations'][i] = resolved[:, 3:].tolist()
json.dump(data, open(dst, 'w'))
print(f'{src} -> {dst}: {len(data["scene_ids"])} scenes in {time.time() - t0:.0f}s')
