"""Write placements.json (and placements_resolved.json) for a PhyScene-format run, for the qualitative viewer.

Furniture is retrieved and placed exactly as export_glb.py / PhyScene's evaluator does (objfeat+size retrieval, dataset
scale, centred on its bbox, ATISS rotation, translation), stored in InstructScene's placement format:
v_world = (v * scale - center) . R(theta) + translation, R = [[c,0,-s],[0,1,0],[s,0,c]].
Run in the PhyScene dir with the physcene env.

usage: python dump_placements.py <run_dir> [<run_dir> ...]   (reads result.json, result_resolved.json)
"""
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
objects = pickle.load(open("data/pickled_data/threed_future_model_livingroom.pkl", "rb"))
classes = np.array(json.load(open("data/preprocessed_data/LivingRoom/dataset_stats.txt"))["class_labels"])
_bounds = {}


def bounds(path):
    """min/max of the obj's vertices, read straight from the 'v' lines (no trimesh processing)."""
    if path not in _bounds:
        with open(path, "rb") as f:
            v = np.array([l.split()[1:4] for l in f if l.startswith(b"v ")], dtype=float)
        _bounds[path] = (v.min(0), v.max(0))
    return _bounds[path]


for run in sys.argv[1:]:
    for src, dst in (("result.json", "placements.json"), ("result_resolved.json", "placements_resolved.json")):
        p = os.path.join(run, src)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        out = []
        for i, sid in enumerate(d["scene_ids"]):
            cl, tr = np.array(d["class_labels"][i]), np.array(d["translations"][i])
            sz, ang = np.array(d["sizes"][i]), np.array(d["angles"][i]).reshape(-1)
            feat = np.array(d["objfeats_32"][i])
            objs = []
            for j in range(len(tr)):
                label = str(classes[cl[j].argmax(-1)])
                if label == "empty":
                    continue
                f = objects.get_closest_furniture_to_objfeats_and_size(label, feat[j], sz[j])
                lo, hi = bounds(f.raw_model_path)
                s = float(f.scale) if np.ndim(f.scale) == 0 else np.asarray(f.scale, float)
                center = ((lo + hi) / 2 * s).tolist()
                objs.append(dict(jid=f.model_jid, label=label, mesh=os.path.abspath(f.raw_model_path),
                                 scale=(np.ones(3) * s).tolist(), center=center, theta=float(ang[j]),
                                 translation=tr[j].tolist()))
            out.append(dict(scene_uid=sid, objects=objs))
        json.dump(out, open(os.path.join(run, dst), "w"))
        print("wrote", os.path.join(run, dst), len(out), "scenes", flush=True)
