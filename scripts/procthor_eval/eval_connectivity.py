"""Offline global-connectivity / navigation metrics for converted ProcTHOR scenes.

Answers "does a NavMesh point exist near each object" -> "can an agent actually get there
from one start point, and between objects, and how long is the route".

Works directly on the scene JSONs written by convert_echoscene_to_procthor.py (no AI2-THOR
needed). Geometry is the exact rotated footprint of every furniture piece that blocks the
NavMesh in AI2-THOR (the `furn|i|lid` wall polygons, bottom height < --obstacle_height,
same rule as `furniture_boxes`).

Configuration space: a cell centre p (raster --res, default 0.05 m) is free iff its exact
Euclidean distance to every footprint and to every room wall is >= --agent_radius. This is
the 2D equivalent of AI2-THOR's capsule check behind GetReachablePositions (gridSize 0.25).

Per-scene metrics (all in [0, 1] unless noted):
  free_ratio          free C-space area / room area
  lcc_ratio           largest 8-connected free component / free area (1 = one region)
  n_components        number of free components with area >= one 0.25 m grid cell
  reach_rate          objects reachable from the EDT spawn: some free cell in the spawn's
                      component within --reach_dist (0.75 m) of the footprint edge
                      (same rule as eval_navigation.py, but on the true C-space)
  reach_rate_lattice  same rule, but the reachable set is an AI2-THOR-like 4-neighbour flood
                      fill on the 0.25 m lattice (reproduces GetReachablePositions counts on
                      most scenes; see docs/navigability_v2.md)
  pair_success        object pairs (i<j) whose 0.75 m approach regions share a component,
                      i.e. a shortest path between them exists
  detour_ratio        mean over connected pairs (Euclidean >= --min_pair_dist) of
                      shortest-path length / straight-line distance between approach points
                      (8-connected Dijkstra = A* with a zero heuristic, then line-of-sight
                      string pulling so the grid does not inflate lengths)
  phy_walkable        PhyScene R_walkable: largest connected walkable area / walkable area
  phy_walkable_code   same, but reproducing the released code literally (returns after the
                      first label, see docs/navigability_v2.md)
  phy_reach           PhyScene R_reach: sum over components of (#objects touching it /
                      #objects) * component area / walkable area

A scene with no free cell at all scores 0 on every metric (never skipped). Scenes with no
blocking objects get reach/pair metrics = NaN and are excluded from those means only.

Usage:
  python scripts/procthor_eval/eval_connectivity.py --scenes_dir <procthor_scenes> [--out x.json]
  python scripts/procthor_eval/eval_connectivity.py --scenes_dir ... --thor_check 20
"""
import os
import json
import math
import argparse

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

GRID_CELL_AREA = 0.25 * 0.25


# ----------------------------------------------------------------------------- geometry
def list_scene_files(scenes_dir):
    return sorted(f for f in os.listdir(scenes_dir)
                  if f.endswith('.json') and not f.startswith('._')
                  and not f.startswith(('walkability_', 'navigation_', 'connectivity_')))


def load_scene(path, obstacle_height=1.0):
    """Return room (l, w) and a list of obstacles {corners(4x2), min_y, center, size, angle, raw}."""
    with open(path) as f:
        house = json.load(f)
    l = house['room_dims']['l']
    w = house['room_dims']['w']
    raw = house.get('objects_raw', [])
    walls = {wl['id']: wl for wl in house.get('walls', [])}

    objects = []
    idx = 0
    while f'furn|{idx}|lid' in walls:
        lid = walls[f'furn|{idx}|lid']['polygon']
        corners = np.array([[p['x'], p['z']] for p in lid])
        min_y = walls[f'furn|{idx}|0']['polygon'][0]['y']
        r = raw[idx] if idx < len(raw) else None
        objects.append({
            'idx': idx, 'corners': corners, 'min_y': min_y,
            'center': corners.mean(0),
            'size_xz': (r['size']['x'], r['size']['z']) if r else None,
            'angle': r['rotation']['y'] if r else 0.0,
            'center_y': r['position']['y'] if r else 0.0,
            'blocking': min_y < obstacle_height,
        })
        idx += 1

    if not objects:  # older JSONs without per-object walls: fall back to AABBs
        for i, (x0, x1, z0, z1) in enumerate(house.get('furniture_boxes', [])):
            c = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
            objects.append({'idx': i, 'corners': c, 'min_y': 0.0, 'center': c.mean(0),
                            'size_xz': (x1 - x0, z1 - z0), 'angle': 0.0, 'center_y': 0.0,
                            'blocking': True})
    return l, w, objects


def dist_to_rect(px, pz, corners):
    """Exact Euclidean distance from points to a (rotated) rectangle; 0 inside."""
    c0, c1, _, c3 = corners
    u = c1 - c0
    v = c3 - c0
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-9 or lv < 1e-9:
        return np.hypot(px - corners[:, 0].mean(), pz - corners[:, 1].mean())
    u, v = u / lu, v / lv
    center = corners.mean(0)
    dx, dz = px - center[0], pz - center[1]
    a = np.abs(dx * u[0] + dz * u[1]) - lu / 2
    b = np.abs(dx * v[0] + dz * v[1]) - lv / 2
    return np.hypot(np.maximum(a, 0), np.maximum(b, 0))


def edt_spawn(l, w, obstacles, grid=0.25):
    """Lattice point (multiples of `grid` inside the room) with the largest exact clearance to
    walls and obstacle footprints, i.e. the EDT maximum of the free floor snapped to the
    AI2-THOR grid. Returns (x, z, clearance) or None if the room is smaller than one cell."""
    gx = np.arange(grid, l, grid)
    gz = np.arange(grid, w, grid)
    if not len(gx) or not len(gz):
        return None
    GX, GZ = np.meshgrid(gx, gz, indexing='ij')
    d = np.minimum.reduce([GX, l - GX, GZ, w - GZ])
    for o in obstacles:
        d = np.minimum(d, dist_to_rect(GX, GZ, o['corners']))
    i = np.unravel_index(np.argmax(d), d.shape)
    return float(GX[i]), float(GZ[i]), float(d[i])


class Raster:
    def __init__(self, l, w, objects, res=0.05, agent_radius=0.3):
        self.l, self.w, self.res, self.r = l, w, res, agent_radius
        nx = max(1, int(math.ceil(l / res)))
        nz = max(1, int(math.ceil(w / res)))
        xs = (np.arange(nx) + 0.5) * res
        zs = (np.arange(nz) + 0.5) * res
        self.X, self.Z = np.meshgrid(xs, zs, indexing='ij')
        self.blockers = [o for o in objects if o['blocking']]
        wall_d = np.minimum.reduce([self.X, l - self.X, self.Z, w - self.Z])
        self.obj_d = [dist_to_rect(self.X, self.Z, o['corners']) for o in self.blockers]
        obst_d = np.minimum.reduce(self.obj_d) if self.obj_d else np.full_like(self.X, np.inf)
        self.clearance = np.minimum(wall_d, obst_d)       # EDT of the free floor (exact)
        self.free = self.clearance >= agent_radius
        self.labels, self.ncomp = ndimage.label(self.free, structure=np.ones((3, 3)))
        self.comp_area = ndimage.sum(np.ones_like(self.labels), self.labels,
                                     index=np.arange(1, self.ncomp + 1)) * res * res

    def spawn(self, snap=0.25):
        """EDT maximum of the free floor, restricted to the AI2-THOR gridSize lattice."""
        if not self.free.any():
            return None
        if snap:
            sp = edt_spawn(self.l, self.w, self.blockers, snap)
            if sp is not None and sp[2] >= self.r:
                return sp[0], sp[1]
        i = np.unravel_index(np.argmax(self.clearance), self.clearance.shape)
        return float(self.X[i]), float(self.Z[i])

    def lattice_reachable(self, sp, grid=0.25, sub=5):
        """AI2-THOR-like flood fill: 4-neighbour moves on the gridSize lattice anchored at the
        spawn, a move is allowed iff the swept segment keeps >= agent_radius clearance
        (checked at grid/sub steps). Returns an (N, 2) array of reached lattice points."""
        h = grid / sub
        x0 = sp[0] - math.floor(sp[0] / grid) * grid  # lattice offset so the spawn is a node
        z0 = sp[1] - math.floor(sp[1] / grid) * grid
        fx = x0 + np.arange(-sub, int((self.l - x0) / h) + sub + 1) * h
        fz = z0 + np.arange(-sub, int((self.w - z0) / h) + sub + 1) * h
        FX, FZ = np.meshgrid(fx, fz, indexing='ij')
        c = np.minimum.reduce([FX, self.l - FX, FZ, self.w - FZ])
        for o in self.blockers:
            c = np.minimum(c, dist_to_rect(FX, FZ, o['corners']))
        ok = c >= self.r
        node = ok[::sub, ::sub]
        nxl, nzl = node.shape
        # horizontal/vertical move feasibility between lattice nodes
        mx = np.ones((nxl - 1, nzl), bool)
        mz = np.ones((nxl, nzl - 1), bool)
        for t in range(sub + 1):
            mx &= ok[t:t + sub * (nxl - 1):sub, ::sub][:nxl - 1]
            mz &= ok[::sub, t:t + sub * (nzl - 1):sub][:, :nzl - 1]
        si = int(round((sp[0] - fx[0]) / grid))
        sj = int(round((sp[1] - fz[0]) / grid))
        if not (0 <= si < nxl and 0 <= sj < nzl) or not node[si, sj]:
            return np.zeros((0, 2))
        seen = np.zeros_like(node)
        seen[si, sj] = True
        stack = [(si, sj)]
        while stack:
            i, j = stack.pop()
            for ni, nj, m in ((i + 1, j, i + 1 < nxl and mx[i, j]), (i - 1, j, i > 0 and mx[i - 1, j]),
                              (i, j + 1, j + 1 < nzl and mz[i, j]), (i, j - 1, j > 0 and mz[i, j - 1])):
                if m and not seen[ni, nj]:
                    seen[ni, nj] = True
                    stack.append((ni, nj))
        ii, jj = np.nonzero(seen)
        return np.column_stack([fx[0] + ii * grid, fz[0] + jj * grid])

    def cell_of(self, x, z):
        return (min(int(x / self.res), self.X.shape[0] - 1),
                min(int(z / self.res), self.X.shape[1] - 1))


# ----------------------------------------------------------------------------- paths
def grid_graph(free, res):
    """8-connected graph over free cells (no corner cutting). Returns csr graph and cell->node map."""
    nx, nz = free.shape
    idx = -np.ones((nx + 2, nz + 2), dtype=np.int64)
    n = int(free.sum())
    idx[1:-1, 1:-1][free] = np.arange(n)
    pad = np.zeros((nx + 2, nz + 2), bool)
    pad[1:-1, 1:-1] = free
    rows, cols, wts = [], [], []
    for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        nb = pad[1 + di:nx + 1 + di, 1 + dj:nz + 1 + dj]
        ok = free & nb
        if di and dj:
            ok &= pad[1 + di:nx + 1 + di, 1:nz + 1] & pad[1:nx + 1, 1 + dj:nz + 1 + dj]
        a = idx[1:-1, 1:-1][ok]
        b = idx[1 + di:nx + 1 + di, 1 + dj:nz + 1 + dj][ok]
        rows.append(a); cols.append(b)
        wts.append(np.full(len(a), res * (math.sqrt(2) if di and dj else 1.0)))
    rows = np.concatenate(rows); cols = np.concatenate(cols); wts = np.concatenate(wts)
    g = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
    return g, idx[1:-1, 1:-1]


def line_free(free, a, b):
    """Check that the segment between cells a and b only crosses free cells (supercover sampling)."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    t = np.linspace(0, 1, n + 1)
    ii = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
    jj = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
    return bool(free[ii, jj].all())


def smoothed_length(free, cells, res):
    """Greedy string pulling of a cell path; returns metric length."""
    if len(cells) < 2:
        return 0.0
    out = [cells[0]]
    k = 0
    while k < len(cells) - 1:
        j = len(cells) - 1
        while j > k + 1 and not line_free(free, cells[k], cells[j]):
            j -= 1
        out.append(cells[j])
        k = j
    pts = np.array(out, dtype=float) * res
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def unravel_path(pred, src, dst, inv):
    path = []
    cur = dst
    while cur != src and cur >= 0:
        path.append(inv[cur])
        cur = pred[cur]
    if cur != src:
        return None
    path.append(inv[src])
    return path[::-1]


# ----------------------------------------------------------------------------- PhyScene
def physcene_metrics(l, w, objects, robot_width=0.3, image_size=256):
    """Re-implementation of PhyScene scripts/eval/walkable_metric.py (R_walkable) and
    utils/overlap.py::calc_wall_overlap (R_reach) on our room/boxes. Room is centred at the
    origin (PhyScene subtracts the floor centroid); sizes here are full extents."""
    import cv2
    verts = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    scale = np.abs(verts).max() + 0.2
    rw = int(robot_width / scale * image_size / 2)

    def to_img(p):
        return int(p[0] / scale * image_size / 2) + image_size / 2, int(p[1] / scale * image_size / 2) + image_size / 2

    def base():
        img = np.zeros((image_size, image_size, 3), np.uint8)
        pts = np.array([to_img(v) for v in verts], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [pts], (255, 0, 0))
        if rw > 0:
            img[:, :, 0] = cv2.erode(img[:, :, 0], np.ones((rw, rw)), iterations=1)
        return img

    def box_pts(o):
        c = o['center'] - np.array([l / 2, w / 2])
        center = to_img(c)
        sx, sz = o['size_xz']
        size = (int(sx / 2 / scale * image_size / 2) * 2, int(sz / 2 / scale * image_size / 2) * 2)
        return np.intp(cv2.boxPoints(((center[0], center[1]), size, -o['angle'] / np.pi * 180)))

    def components(img):
        wm = img[:, :, 0].copy()
        n, lab, _, _ = cv2.connectedComponentsWithStats(wm, connectivity=8)
        return wm, n, lab

    # R_walkable: boxes with centre height < 1.5 m
    img = base()
    for o in objects:
        if o['center_y'] < 1.5:
            p = box_pts(o)
            cv2.drawContours(img, [p], 0, (0, 255, 0), max(rw, 1))
            cv2.fillPoly(img, [p], (0, 255, 0))
    wm, n, lab = components(img)
    total = (lab != 0).sum()
    if n <= 1 or total == 0:
        walk = walk_code = 0.0
    else:
        areas = np.bincount(lab.ravel())[1:]
        walk = areas.max() / total
        walk_code = areas[0] / total   # released code returns inside the loop at label 1

    # R_reach: all boxes, per-box masks dilated by robot width + 3x3
    img = base()
    masks = []
    for o in objects:
        p = box_pts(o)
        cv2.drawContours(img, [p], 0, (0, 255, 0), max(rw, 1))
        cv2.fillPoly(img, [p], (0, 255, 0))
        m = np.zeros((image_size, image_size, 3), np.uint8)
        cv2.drawContours(m, [p], 0, (0, 255, 0), max(rw, 1))
        cv2.fillPoly(m, [p], (0, 255, 0))
        masks.append(ndimage.binary_dilation(m[:, :, 1] == 255, np.ones((3, 3), bool)))
    wm, n, lab = components(img)
    total = (lab != 0).sum()
    if not masks:
        reach = float('nan')
    elif total == 0:
        reach = 0.0
    else:
        reach = 0.0
        for k in range(1, n):
            comp = lab == k
            touching = sum(bool((m & comp).any()) for m in masks)
            reach += touching / len(masks) * comp.sum() / total
    return float(walk), float(walk_code), float(reach)


# ----------------------------------------------------------------------------- per scene
def evaluate_scene(path, args):
    l, w, objects = load_scene(path, args.obstacle_height)
    R = Raster(l, w, objects, res=args.res, agent_radius=args.agent_radius)
    blockers = R.blockers
    n_obj = len(blockers)
    nan = float('nan')
    out = {'room_area': l * w, 'n_objects': n_obj}

    big = R.comp_area >= GRID_CELL_AREA
    free_area = float(R.free.sum()) * args.res ** 2
    out['free_ratio'] = free_area / (l * w) if l * w > 0 else 0.0
    out['n_components'] = int(big.sum())
    out['lcc_ratio'] = float(R.comp_area.max() / free_area) if free_area > 0 else 0.0

    try:
        out['phy_walkable'], out['phy_walkable_code'], out['phy_reach'] = physcene_metrics(
            l, w, objects, robot_width=args.phy_robot_width)
    except ImportError:
        out['phy_walkable'] = out['phy_walkable_code'] = out['phy_reach'] = nan

    sp = R.spawn()
    out['spawn'] = sp
    if sp is None:  # no navigable floor at all
        out.update(reach_rate=0.0 if n_obj else nan, n_reachable=0,
                   reach_rate_lattice=0.0 if n_obj else nan, n_lattice_reachable=0,
                   pair_success=0.0 if n_obj > 1 else nan, n_pairs=n_obj * (n_obj - 1) // 2,
                   detour_ratio=nan, n_detour_pairs=0, status='no_free_space')
        return out

    spawn_lab = R.labels[R.cell_of(*sp)]
    if spawn_lab == 0:  # lattice snap landed outside C-space: use the raw EDT max
        i = np.unravel_index(np.argmax(R.clearance), R.clearance.shape)
        spawn_lab = R.labels[i]

    # approach region of each object: free cells within reach_dist of its footprint
    approach = [R.free & (d <= args.reach_dist) for d in R.obj_d]
    obj_comps = [set(np.unique(R.labels[a])) - {0} for a in approach]
    reachable = [spawn_lab in c for c in obj_comps]
    out['n_reachable'] = int(sum(reachable))
    out['reach_rate'] = (sum(reachable) / n_obj) if n_obj else nan
    out['unreachable_objects'] = [blockers[i]['idx'] for i in range(n_obj) if not reachable[i]]

    # same rule on an AI2-THOR-like 0.25 m lattice flood fill (offline eval_navigation.py)
    lat = R.lattice_reachable(sp)
    if n_obj:
        lat_ok = [len(lat) > 0 and bool((dist_to_rect(lat[:, 0], lat[:, 1], o['corners']) <= args.reach_dist).any())
                  for o in blockers]
        out['reach_rate_lattice'] = sum(lat_ok) / n_obj
    else:
        out['reach_rate_lattice'] = nan
    out['n_lattice_reachable'] = int(len(lat))

    # pairwise paths
    pairs = [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)]
    out['n_pairs'] = len(pairs)
    if not pairs:
        out.update(pair_success=nan, detour_ratio=nan, n_detour_pairs=0, status='ok')
        return out

    g, idx = grid_graph(R.free, args.res)
    inv_t = [tuple(c) for c in np.argwhere(R.free)]  # node -> (i, j), row-major like idx

    def approach_cell(k, comp):
        cand = approach[k] & (R.labels == comp)
        if not cand.any():
            return None
        d = np.where(cand, R.obj_d[k], np.inf)
        # nearest to the footprint, ties broken towards the object centre
        dc = np.hypot(R.X - blockers[k]['center'][0], R.Z - blockers[k]['center'][1])
        key = np.where(cand, d * 1e3 + dc, np.inf)
        return np.unravel_index(np.argmin(key), key.shape)

    success, detours, sp_cache = 0, [], {}
    for i, j in pairs:
        shared = obj_comps[i] & obj_comps[j]
        if not shared:
            continue
        success += 1
        comp = max(shared, key=lambda c: R.comp_area[c - 1])
        a, b = approach_cell(i, comp), approach_cell(j, comp)
        eu = math.hypot(a[0] - b[0], a[1] - b[1]) * args.res
        if eu < args.min_pair_dist:
            continue
        na, nb = idx[a], idx[b]
        if na not in sp_cache:
            sp_cache[na] = dijkstra(g, directed=False, indices=na, return_predecessors=True)
        dist, pred = sp_cache[na]
        if not np.isfinite(dist[nb]):
            continue
        cells = unravel_path(pred, na, nb, inv_t)
        length = smoothed_length(R.free, cells, args.res) if cells else dist[nb]
        detours.append(max(length, eu) / eu)
    out['pair_success'] = success / len(pairs)
    out['detour_ratio'] = float(np.mean(detours)) if detours else nan
    out['n_detour_pairs'] = len(detours)
    out['status'] = 'ok'
    return out


# ----------------------------------------------------------------------------- AI2-THOR cross-check
def thor_check(scenes_dir, files, args):
    from ai2thor.controller import Controller
    controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)
    rows = []
    for fn in files:
        path = os.path.join(scenes_dir, fn)
        with open(path) as f:
            house = json.load(f)
        l, w, objects = load_scene(path, args.obstacle_height)
        R = Raster(l, w, objects, res=args.res, agent_radius=args.agent_radius)
        sp = R.spawn()
        row = {'scene': fn[:-5], 'spawn': sp}
        try:
            controller.reset(scene=house)
        except Exception as e:
            row['error'] = f'load: {e}'
            rows.append(row); continue
        if sp is None:
            row['error'] = 'no_free_space'; rows.append(row); continue
        controller.step(action="Teleport", position={"x": sp[0], "y": 0.9, "z": sp[1]}, forceAction=True)
        ev = controller.step(action="GetReachablePositions")
        pts = ev.metadata["actionReturn"] if ev.metadata["lastActionSuccess"] else []
        pts = [p for p in (pts or []) if 0 <= p['x'] <= l and 0 <= p['z'] <= w]
        thor = {(round(p['x'] / 0.25), round(p['z'] / 0.25)) for p in pts}
        # offline: lattice points in the spawn component
        spawn_lab = R.labels[R.cell_of(*sp)]
        mine = set()
        for gx in np.arange(0.25, l, 0.25):
            for gz in np.arange(0.25, w, 0.25):
                c = R.cell_of(gx, gz)
                if R.labels[c] == spawn_lab and spawn_lab > 0:
                    mine.add((round(gx / 0.25), round(gz / 0.25)))
        inter = len(thor & mine)
        union = len(thor | mine)
        row.update(n_thor=len(thor), n_offline=len(mine), iou=inter / union if union else 1.0)
        # shortest paths spawn -> each object's nearest THOR point (as plot_path.py)
        sp_paths = []
        for o in R.blockers:
            if not pts:
                break
            goal = min(pts, key=lambda p: math.hypot(p['x'] - o['center'][0], p['z'] - o['center'][1]))
            e = controller.step(action="GetShortestPathToPoint", target=goal)
            ok = e.metadata["lastActionSuccess"]
            plen = None
            if ok:
                c = e.metadata["actionReturn"]["corners"]
                plen = sum(math.hypot(c[k + 1]['x'] - c[k]['x'], c[k + 1]['z'] - c[k]['z']) for k in range(len(c) - 1))
            sp_paths.append({'obj': o['idx'], 'ok': ok, 'len': plen,
                             'euclid': math.hypot(goal['x'] - sp[0], goal['z'] - sp[1])})
        row['paths'] = sp_paths
        rows.append(row)
        print(f"{row['scene']}: THOR {row['n_thor']} pts, offline {row['n_offline']}, IoU {row['iou']:.3f}")
    controller.stop()
    return rows


# ----------------------------------------------------------------------------- main
METRICS = ['free_ratio', 'lcc_ratio', 'n_components', 'reach_rate', 'reach_rate_lattice', 'pair_success', 'detour_ratio',
           'phy_walkable', 'phy_walkable_code', 'phy_reach']


def summarize(scenes):
    s = {'instance_count': len(scenes)}
    for m in METRICS:
        v = np.array([r[m] for r in scenes.values()], dtype=float)
        v = v[np.isfinite(v)]
        s[m] = float(v.mean()) if len(v) else None
        s[m + '_n'] = int(len(v))
    n_obj = sum(r['n_objects'] for r in scenes.values())
    s['reach_rate_pooled'] = sum(r.get('n_reachable', 0) for r in scenes.values()) / n_obj if n_obj else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes_dir', required=True)
    ap.add_argument('--out', default=None, help='default: <scenes_dir>/connectivity_results.json')
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--agent_radius', type=float, default=0.2,
                    help='0.2 matches AI2-THOR (min wall clearance of GetReachablePositions points is 0.207 m)')
    ap.add_argument('--reach_dist', type=float, default=0.75)
    ap.add_argument('--obstacle_height', type=float, default=1.0)
    ap.add_argument('--min_pair_dist', type=float, default=0.5,
                    help='skip pairs closer than this for the detour ratio')
    ap.add_argument('--phy_robot_width', type=float, default=0.3)
    ap.add_argument('--thor_check', type=int, default=0,
                    help='also compare N scenes against AI2-THOR GetReachablePositions/GetShortestPathToPoint')
    ap.add_argument('--thor_out', default=None)
    args = ap.parse_args()

    files = list_scene_files(args.scenes_dir)
    if args.thor_check:
        rng = np.random.RandomState(0)
        pick = sorted(rng.choice(files, min(args.thor_check, len(files)), replace=False))
        rows = thor_check(args.scenes_dir, pick, args)
        out = args.thor_out or os.path.join(args.scenes_dir, 'connectivity_thor_check.json')
        with open(out, 'w') as f:
            json.dump(rows, f, indent=2)
        ious = [r['iou'] for r in rows if 'iou' in r]
        print(f'mean IoU offline vs THOR reachable set: {np.mean(ious):.3f} over {len(ious)} scenes -> {out}')
        return

    scenes = {}
    for k, fn in enumerate(files):
        scenes[fn[:-5]] = evaluate_scene(os.path.join(args.scenes_dir, fn), args)
        if (k + 1) % 50 == 0:
            print(f'{k + 1}/{len(files)}')
    summary = summarize(scenes)
    summary['params'] = vars(args)
    out = args.out or os.path.join(args.scenes_dir, 'connectivity_results.json')
    with open(out, 'w') as f:
        json.dump({'summary': summary, 'scenes': scenes}, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'params'}, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
