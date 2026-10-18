"""Function-aware accessibility metrics for converted ProcTHOR scenes (offline).

eval_connectivity.py asks "can the agent get within 0.75 m of each object". These metrics
ask whether the object is *usable*:

  func_access     object reachable from the EDT spawn AND the agent can stand in the
                  object's usable zone (a band --zone_depth deep along its usable side):
                    front only : wardrobe cabinet bookshelf shelf nightstand desk sofa tv_stand
                    long sides : bed (either side)
                    any side   : table chair lamp
  front_clear     fraction of the usable zone that is inside the room and not covered by
                  another floor-level footprint (best side for multi-side classes)
  front_blocked   share of objects whose usable zone is < 50 % clear
  comfy_reach     reachable from the spawn through a corridor >= --comfy_width wide
                  (agent radius = comfy_width / 2; default 0.8 m, an accessible-route width)
  bottleneck_w    widest-path corridor width (m) from the spawn to each object, 0 if unreachable
                  (radius swept 0.20..0.60 m in 0.05 m steps)
  reach_rate      the eval_connectivity rule (0.75 m, radius 0.2) on the same footprints
  *_aw            spawn-independent versions of reach_rate / func_access: expected value for a
                  start drawn uniformly from the free floor (each free component weighted by
                  its area), so a scene is not judged by which region the EDT spawn falls in

Orientation: footprints are rebuilt from objects_raw with --yaw_sign (default -1). With -1 the
local -z axis of wall-hugging furniture points at the nearest wall for 629/695 objects
(EchoScene+PP), i.e. +z is the front; with the converter's +1 sign there is no consistent
front. Object centres are taken from the converter's lid polygons (sign independent).

  python scripts/procthor_eval/eval_functional.py --scenes_dir <procthor_scenes> --out x.json
"""
import os
import json
import math
import argparse

import numpy as np
from scipy import ndimage

from eval_connectivity import list_scene_files, edt_spawn

FRONT = {'wardrobe', 'cabinet', 'bookshelf', 'shelf', 'nightstand', 'desk', 'sofa', 'tv_stand'}
LONG_SIDES = {'bed'}
ANY_SIDE = {'table', 'chair', 'lamp'}
RADII = np.round(np.arange(0.20, 0.601, 0.05), 2)


def load_objects(path, yaw_sign=-1.0, obstacle_height=1.0):
    with open(path) as f:
        house = json.load(f)
    l, w = house['room_dims']['l'], house['room_dims']['w']
    walls = {wl['id']: wl for wl in house['walls']}
    objs = []
    for i, r in enumerate(house.get('objects_raw', [])):
        lid = walls.get(f'furn|{i}|lid')
        if lid is None:
            continue
        c = np.array([[p['x'], p['z']] for p in lid['polygon']]).mean(0)
        min_y = walls[f'furn|{i}|0']['polygon'][0]['y']
        a = yaw_sign * r['rotation']['y']
        hx, hz = r['size']['x'] / 2, r['size']['z'] / 2
        ca, sa = math.cos(a), math.sin(a)
        corners = np.array([[c[0] + lx * ca - lz * sa, c[1] + lx * sa + lz * ca]
                            for lx, lz in [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]])
        objs.append({'idx': i, 'cls': r['class'], 'center': c, 'corners': corners, 'hx': hx, 'hz': hz,
                     'cos': ca, 'sin': sa, 'blocking': min_y < obstacle_height})
    return l, w, objs


def local(o, X, Z):
    dx, dz = X - o['center'][0], Z - o['center'][1]
    return dx * o['cos'] + dz * o['sin'], -dx * o['sin'] + dz * o['cos']


def footprint_dist(o, X, Z):
    lx, lz = local(o, X, Z)
    return np.hypot(np.maximum(np.abs(lx) - o['hx'], 0), np.maximum(np.abs(lz) - o['hz'], 0))


def zones(o, X, Z, depth):
    """(mask, full band area) for each usable side of an object. Band area outside the room
    is missing from the mask and therefore counts as blocked."""
    lx, lz = local(o, X, Z)
    hx, hz = o['hx'], o['hz']
    inx, inz = np.abs(lx) <= hx, np.abs(lz) <= hz
    front = inx & (lz > hz) & (lz <= hz + depth)
    back = inx & (lz < -hz) & (lz >= -hz - depth)
    right = inz & (lx > hx) & (lx <= hx + depth)
    left = inz & (lx < -hx) & (lx >= -hx - depth)
    fb, lr = 2 * hx * depth, 2 * hz * depth  # full analytic band areas
    if o['cls'] in FRONT:
        return [(front, fb)]
    if o['cls'] in LONG_SIDES:
        return [(left, lr), (right, lr)] if hz >= hx else [(front, fb), (back, fb)]
    return [(front, fb), (back, fb), (left, lr), (right, lr)]


def evaluate_scene(path, args):
    l, w, objs = load_objects(path, args.yaw_sign)
    res = args.res
    nx, nz = max(1, int(math.ceil(l / res))), max(1, int(math.ceil(w / res)))
    X, Z = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(nz) + 0.5) * res, indexing='ij')
    blk = [o for o in objs if o['blocking']]
    nan = float('nan')
    out = {'n_objects': len(blk)}
    if not blk:
        return dict(out, func_access=nan, front_clear=nan, front_blocked=nan, comfy_reach=nan,
                    bottleneck_w=nan, reach_rate=nan, reach_aw=nan, func_access_aw=nan)

    dists = [footprint_dist(o, X, Z) for o in blk]
    wall_d = np.minimum.reduce([X, l - X, Z, w - Z])
    occ = np.minimum.reduce(dists) <= 0
    clearance = np.minimum(wall_d, np.minimum.reduce(dists))
    sp = edt_spawn(l, w, blk)
    if sp is not None and sp[2] >= RADII[0]:
        si = (min(int(sp[0] / res), nx - 1), min(int(sp[1] / res), nz - 1))
    else:
        si = np.unravel_index(np.argmax(clearance), clearance.shape)

    # widest-path bottleneck: largest radius at which the object is still reachable
    bott = np.zeros(len(blk))
    reach02 = np.zeros(len(blk), bool)
    spawn_comp02 = None
    for r in RADII:
        free = clearance >= r
        if not free[si]:  # the spawn (EDT max) itself no longer fits: nothing wider is reachable
            break
        lab, _ = ndimage.label(free, structure=np.ones((3, 3)))
        comp = lab == lab[si]
        if r == RADII[0]:
            spawn_comp02 = comp
        for k, d in enumerate(dists):
            if (comp & (d <= args.reach_dist)).any():
                bott[k] = r
                if r == RADII[0]:
                    reach02[k] = True

    # spawn-independent: expected accessibility for a uniformly random start in the free floor
    free02 = clearance >= RADII[0]
    lab02, ncomp = ndimage.label(free02, structure=np.ones((3, 3)))
    total = free02.sum()
    reach_aw = func_aw = 0.0
    zone_masks = [zones(o, X, Z, args.zone_depth) for o in blk]
    for c in range(1, ncomp + 1):
        comp = lab02 == c
        wgt = comp.sum() / total
        reach_aw += wgt * np.mean([(comp & (d <= args.reach_dist)).any() for d in dists])
        func_aw += wgt * np.mean([any((zm & comp).any() for zm, _ in zm_list) for zm_list in zone_masks])

    func, clear = [], []
    for k, o in enumerate(blk):
        others = [d for j, d in enumerate(dists) if j != k]
        other_occ = (np.minimum.reduce(others) <= 0) if others else np.zeros_like(occ)
        best_clear, ok = 0.0, False
        for zm, area in zones(o, X, Z, args.zone_depth):
            c = float((zm & ~other_occ).sum() * res ** 2 / max(area, 1e-9))
            best_clear = max(best_clear, min(c, 1.0))
            if spawn_comp02 is not None and (zm & spawn_comp02).any():
                ok = True
        func.append(ok)
        clear.append(best_clear)
    clear = np.array(clear)
    out.update(func_access=float(np.mean(func)), front_clear=float(clear.mean()),
               front_blocked=float((clear < 0.5).mean()),
               comfy_reach=float((bott >= args.comfy_width / 2 - 1e-9).mean()),
               bottleneck_w=float((2 * bott).mean()), reach_rate=float(reach02.mean()),
               reach_aw=float(reach_aw), func_access_aw=float(func_aw))
    return out


METRICS = ['func_access', 'func_access_aw', 'reach_aw', 'front_clear', 'front_blocked', 'comfy_reach', 'bottleneck_w', 'reach_rate']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes_dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--zone_depth', type=float, default=0.6)
    ap.add_argument('--reach_dist', type=float, default=0.75)
    ap.add_argument('--comfy_width', type=float, default=0.8)
    ap.add_argument('--yaw_sign', type=float, default=-1.0)
    args = ap.parse_args()
    scenes = {fn[:-5]: evaluate_scene(os.path.join(args.scenes_dir, fn), args)
              for fn in list_scene_files(args.scenes_dir)}
    summ = {'instance_count': len(scenes), 'params': vars(args)}
    for m in METRICS:
        v = np.array([s[m] for s in scenes.values()], float)
        summ[m] = float(np.nanmean(v))
    out = args.out or os.path.join(args.scenes_dir, 'functional_results.json')
    with open(out, 'w') as f:
        json.dump({'summary': summ, 'scenes': scenes}, f, indent=2)
    print(json.dumps({m: round(summ[m], 4) for m in METRICS}), '->', out)


if __name__ == '__main__':
    main()
