"""Per-scene versions of the controlled-rerun metrics, for paired bootstrap CIs.

ColObj / ColScene twice: `*_legacy` replicates scripts/eval_collision.py exactly (reproduces the
published numbers, including its three bugs, see legacy_collision); the plain keys are corrected
(wardrobes included, y = box bottom, correct footprint) and `*_yawfix` additionally mirrors the yaw. scripts/eval_ob.py (OOB: footprint not inside the floor footprint) and
scripts/relational/evaluate_relational_accuracy.py (per-relation accuracy from validate_constrains),
but kept per scene. Also the PP displacement (raw -> PP) per scene, as in scripts/eval/pp_damage.py.

--yaw_sign -1 recomputes ColObj/OOB with the footprint rotation mirrored (see
docs/navigability_v2.md, "Orientation fix"); the default +1 reproduces the existing evaluators.

  python scripts/eval/per_scene_metrics.py --pp pp.json --raw raw.json --out scenes.json [--no_rel]
"""
import os
import sys
import json
import argparse

import numpy as np
from shapely.geometry import Polygon

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def footprint(t, s, a, sign):
    a = sign * a
    c, si = np.cos(a), np.sin(a)
    dx = np.array([s[0] / 2, s[0] / 2, -s[0] / 2, -s[0] / 2])
    dz = np.array([s[2] / 2, -s[2] / 2, -s[2] / 2, s[2] / 2])
    return Polygon(zip(t[0] + dx * c - dz * si, t[2] + dx * si + dz * c))


def legacy_collision(data):
    """Exact replica of the scripts/eval_collision.py 'standard prediction JSON' path, which the
    published ColObj/ColScene numbers used: classes nc-1 and nc-2 excluded (nc-2 = 13 is the
    wardrobe, not a floor class), y treated as the box centre, and the footprint z coordinate
    computed as z + dz*sin + dz*cos (typo for dx*sin)."""
    out = {}
    for i, sid in enumerate(data['scene_ids']):
        T = np.array(data['translations'][i]); S = np.array(data['sizes'][i])
        A = np.array(data['angles'][i]).reshape(-1)
        if np.abs(A).max() > 6.29:
            A = np.radians(A)
        cls = np.argmax(np.array(data['class_labels'][i]), -1)
        nc = len(data['class_labels'][i][0])
        valid = (cls != nc - 1) & (cls != nc - 2)
        if 'objectness' in data and len(data['objectness'][i]) == len(valid):
            valid &= np.array(data['objectness'][i]).squeeze() > 0.0
        idx = np.where(valid)[0]

        def poly(k):
            c, si = np.cos(A[k]), np.sin(A[k])
            dx = np.array([S[k][0] / 2, S[k][0] / 2, -S[k][0] / 2, -S[k][0] / 2])
            dz = np.array([S[k][2] / 2, -S[k][2] / 2, -S[k][2] / 2, S[k][2] / 2])
            return Polygon(zip(T[k][0] + dx * c - dz * si, T[k][2] + dz * si + dz * c))
        col = np.zeros(len(idx), bool)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i1, i2 = idx[a], idx[b]
                if T[i1][1] + S[i1][1] / 2 <= T[i2][1] - S[i2][1] / 2 or T[i2][1] + S[i2][1] / 2 <= T[i1][1] - S[i1][1] / 2:
                    continue
                p, q = poly(i1), poly(i2)
                if p.is_valid and q.is_valid and p.intersects(q) and p.intersection(q).area > 1e-4:
                    col[a] = col[b] = True
        out[sid] = {'col_obj_legacy': float(col.mean()) if len(col) else 0.0, 'col_scene_legacy': float(col.any()),
                    'n_col_legacy': int(col.sum()), 'n_furn_legacy': len(idx)}
    return out


def geometry(data, sign=1.0):
    out = {}
    for i, sid in enumerate(data['scene_ids']):
        cls = np.argmax(np.array(data['class_labels'][i]), -1)
        nc = len(data['class_labels'][i][0])
        T = np.array(data['translations'][i]); S = np.array(data['sizes'][i])
        A = np.array(data['angles'][i]).reshape(-1)
        objn = np.array(data['objectness'][i]).ravel() if 'objectness' in data else np.ones(len(cls))
        furn = [j for j in range(len(cls)) if cls[j] != nc - 1 and objn[j] > 0.5]   # floor/scene tokens out, wardrobes in
        polys = {j: footprint(T[j], S[j], A[j], sign) for j in furn}
        col = np.zeros(len(furn), bool)
        for a in range(len(furn)):
            for b in range(a + 1, len(furn)):
                i1, i2 = furn[a], furn[b]
                if T[i1][1] + S[i1][1] <= T[i2][1] or T[i2][1] + S[i2][1] <= T[i1][1]:
                    continue
                p, q = polys[i1], polys[i2]
                if p.is_valid and q.is_valid and p.intersects(q) and p.intersection(q).area > 1e-4:
                    col[a] = col[b] = True
        # OOB as eval_ob: floor = largest flat box, tiny / parked tokens ignored
        flat = [j for j in range(len(S)) if abs(S[j][1]) < 0.1 and S[j][0] * S[j][2] > 0.5]
        ob = 0; n_ob = 0
        if flat:
            f = max(flat, key=lambda j: S[j][0] * S[j][2])
            fp = footprint(T[f], S[f], A[f], sign)
            for j in range(len(S)):
                if j == f or (S[j][0] < 0.2 and S[j][2] < 0.2) or (T[j][0] < -4 and T[j][2] < -4) or objn[j] <= 0:
                    continue
                p = footprint(T[j], S[j], A[j], sign)
                if not p.is_valid:
                    continue
                n_ob += 1
                ob += (p.area - fp.intersection(p).area) > 1e-4
        out[sid] = {'col_obj': float(col.mean()) if len(col) else 0.0, 'col_scene': float(col.any()),
                    'n_col': int(col.sum()), 'n_furn': len(furn), 'oob': int(ob)}
    return out


def displacement(raw, pp):
    out = {}
    for i, sid in enumerate(pp['scene_ids']):
        j = raw['scene_ids'].index(sid)
        o = np.array(pp['objectness'][i]).ravel() > 0.5
        a = np.array(raw['translations'][j])[o][:, [0, 2]]
        b = np.array(pp['translations'][i])[o][:, [0, 2]]
        d = np.linalg.norm(b - a, axis=1)
        out[sid] = {'pp_move_cm': float(d.mean() * 100) if len(d) else 0.0,
                    'pp_gt30_frac': float((d > 0.3).mean()) if len(d) else 0.0}
    return out


def relations(data):
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, 'scripts', 'relational'))
    import torch
    import evaluate_relational_accuracy as era
    ds = era.ThreedFrontDatasetSceneGraph(root=os.path.join(REPO, 'FRONT'), split='val_scans', use_scene_rels=True,
                                          with_changes=False, eval=True, eval_type='none', with_CLIP=False,
                                          use_SDF=False, large=False, room_type='all')
    scan = {s: k for k, s in enumerate(ds.scans)}
    keys = ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller',
            'standing on', 'close by', 'symmetrical to', 'total']
    out = {}
    for i, sid in enumerate(data['scene_ids']):
        if sid not in scan:
            continue
        item = ds[scan[sid]]
        tr = item['decoder']['triples']
        tr = tr if isinstance(tr, torch.Tensor) else torch.tensor(tr, dtype=torch.long)
        boxes = torch.cat([torch.tensor(data['sizes'][i], dtype=torch.float32),
                           torch.tensor(data['translations'][i], dtype=torch.float32)], -1)
        ang = torch.tensor(data['angles'][i], dtype=torch.float32) * (180.0 / np.pi)
        tr = tr[(tr[:, 0] < len(boxes)) & (tr[:, 2] < len(boxes))]
        acc = era.validate_constrains(tr, boxes, ang, None, ds.vocab, {k: [] for k in keys})
        rec = {}
        for k in keys:
            rec['rel_' + k.replace(' ', '_')] = float(np.mean(acc[k])) if len(acc[k]) else float('nan')
        pairs = [('left', 'right'), ('front', 'behind'), ('bigger', 'smaller'), ('taller', 'shorter')]
        parts = [np.nanmean([rec['rel_' + a], rec['rel_' + b]]) for a, b in pairs]
        parts += [rec['rel_standing_on'], rec['rel_close_by'], rec['rel_symmetrical_to']]
        rec['rel_macro'] = float(np.nanmean(parts)) if np.isfinite(parts).any() else float('nan')
        rec['n_rel'] = len(acc['total'])
        out[sid] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pp', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_rel', action='store_true')
    args = ap.parse_args()
    pp, raw = json.load(open(args.pp)), json.load(open(args.raw))
    scenes = {}
    for tag, d in (('pp', pp), ('raw', raw)):
        for sid, rec in legacy_collision(d).items():
            scenes.setdefault(sid, {}).update({f'{tag}_{k}': v for k, v in rec.items()})
        for sign, suf in ((1.0, ''), (-1.0, '_yawfix')):
            for sid, rec in geometry(d, sign).items():
                scenes.setdefault(sid, {}).update({f'{tag}_{k}{suf}': v for k, v in rec.items()})
        if not args.no_rel:
            for sid, rec in relations(d).items():
                scenes[sid].update({f'{tag}_{k}': v for k, v in rec.items()})
    for sid, rec in displacement(raw, pp).items():
        scenes[sid].update(rec)
    json.dump({'scenes': scenes}, open(args.out, 'w'), indent=1)
    print(f'{len(scenes)} scenes -> {args.out}')


if __name__ == '__main__':
    main()
