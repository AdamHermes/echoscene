"""Mesh-level 3D collision rate for a layout JSON plus the generated object meshes.

Each generated mesh (`<scene>/<class>_<cat>_<instance>.obj`) is fitted into its predicted box exactly as
scripts/rendering/render_images_only.py does for the FID renders, so raw and post-processed layouts reuse the
same shapes. Two objects collide if any vertex of one lies inside the other's (watertight) mesh, the
point-in-mesh rule of PhyScene's `calc_overlap` (overlap_type=mesh), checked in both directions. Inside-ness is
the generalised winding number (> 0.5). An axis-aligned bbox test filters pairs first, and only vertices inside
the other mesh's bbox are tested. Floor/non-object entries are skipped; every other object counts (as in
eval_collision.py).

usage: python scripts/eval/eval_mesh_collision.py --json <layout.json> --mesh_dir <.../object_meshes> [--out per_scene.json]
"""
import os, sys, glob, json, argparse
from multiprocessing import Pool
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rendering'))
from render_images_only import fit_shapes_to_box_v2


def winding_inside(points, tris, chunk=200000):
    """Generalised winding number of each point w.r.t. a closed triangle soup; > 0.5 means inside."""
    w = np.zeros(len(points))
    step = max(1, chunk // len(tris))
    for s in range(0, len(points), step):
        p = points[s:s + step, None, None, :]
        a, b, c = (tris[None, :, k, :] - p[:, :, 0, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        det = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        div = la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la \
            + np.einsum('ijk,ijk->ij', c, a) * lb
        w[s:s + step] = np.arctan2(det, div).sum(1) / (2 * np.pi)
    return np.abs(w) > 0.5


def any_inside(src, dst):
    lo, hi = dst.bounds
    v = src.vertices[np.all((src.vertices > lo) & (src.vertices < hi), axis=1)]
    return len(v) > 0 and bool(winding_inside(v, dst.triangles).any())


def scene(args):
    sid, cats, sizes, trans, angles, objness, mesh_dir = args
    meshes, inst = [], 1
    for j, c in enumerate(cats):
        if objness[j] != 1.0:
            continue
        f = glob.glob(os.path.join(mesh_dir, sid, f'*_{c}_{inst}.obj'))
        inst += 1
        if not f:
            continue
        m = trimesh.load(f[0], force='mesh', process=False)
        box = torch.tensor(np.concatenate([sizes[j], trans[j], [np.degrees(angles[j])]])).float()
        _, m = fit_shapes_to_box_v2(m, box, degrees=True)
        meshes.append((os.path.basename(f[0]).split('_')[0], m))
    n = len(meshes)
    hit = np.zeros(n, bool)
    for i in range(n):
        for j in range(i + 1, n):
            if hit[i] and hit[j]:
                continue
            (lo1, hi1), (lo2, hi2) = meshes[i][1].bounds, meshes[j][1].bounds
            if np.any(hi1 <= lo2) or np.any(hi2 <= lo1):
                continue
            if any_inside(meshes[j][1], meshes[i][1]) or any_inside(meshes[i][1], meshes[j][1]):
                hit[i] = hit[j] = True
    return sid, n, int(hit.sum()), [m[0] for m in meshes], hit.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True); ap.add_argument('--mesh_dir', required=True)
    ap.add_argument('--out', default=None); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    d = json.load(open(a.json))
    jobs = []
    for i, sid in enumerate(d['scene_ids']):
        cats = np.argmax(np.array(d['class_labels'][i]), 1)
        ang = np.array(d['angles'][i]).reshape(-1)
        obj = [o[0] for o in d['objectness'][i]]
        jobs.append((sid, cats, np.array(d['sizes'][i]), np.array(d['translations'][i]), ang, obj, a.mesh_dir))
    with Pool(a.workers) as p:
        res = p.map(scene, jobs, chunksize=4)
    per = {r[0]: dict(n=r[1], collided=r[2], labels=r[3], hit=r[4]) for r in res}
    rooms = {}
    for sid, r in per.items():
        for key in ('all', ''.join(ch for ch in sid.split('-')[0] if ch.isalpha()).lower()):
            s = rooms.setdefault(key, [0, 0, 0, 0])
            s[0] += r['n']; s[1] += r['collided']; s[2] += 1; s[3] += r['collided'] > 0
    print(f'Mesh collision ({a.json})')
    print(f'{"room":18s} | ColObj  | ColScene | objects | scenes')
    for k in ['all'] + sorted(x for x in rooms if x != 'all'):
        o, co, sc, cs = rooms[k]
        print(f'{k:18s} | {co / max(o, 1):.4f}  | {cs / max(sc, 1):.4f}   | {co}/{o} | {cs}/{sc}')
    if a.out:
        json.dump(per, open(a.out, 'w'))


if __name__ == '__main__':
    main()
