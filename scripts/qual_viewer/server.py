"""Qualitative-figure browser for SteerScene: side-by-side 3D views of the same test scene under several methods.

Runs on the machine that holds the scenes. Each backbone in the config is a set of runs (raw layout, and after the
collision resolver) plus the panels to show. Three kinds of run are supported:
  echoscene   EchoScene layout JSON + the generated meshes (zip or folder); each mesh is fitted into its box exactly
              as in the FID renders and the mesh collision eval.
  placements  InstructScene-style placements.json (jid, scale, center, theta, translation per object) + the 3D-FUTURE
              meshes they use, and a floor JSON (floor_xz, floor_faces). PhyScene runs are converted to this format.
  boxes       a layout JSON of boxes only (scene_ids, class_labels, translations, sizes, angles), e.g. DiffuScene.
On first start the server builds a per-scene index of what goes wrong in each run: colliding objects (mesh-level for
echoscene when an eval_mesh_collision.py --out file is given, else rotated-box overlap), out-of-room objects (rotated
footprint not inside the floor) and objects the resolver moved. The viewer ranks scenes with it. Scene files are sent
as GLB, pre-built in the background and cached on disk.

usage: python server.py --config config_lhanubu.json [--port 8765] [--host 127.0.0.1]
then open http://localhost:8765 (from another machine: ssh -L 8765:localhost:8765 <host>).
Only needs numpy + trimesh (python >= 3.8).
"""
import argparse, io, json, os, re, sys, threading, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from multiprocessing import Pool
from urllib.parse import parse_qs, urlparse, unquote
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_RE = re.compile(r'(?:^|/)object_meshes/([^/]+)/([^/]+)\.obj$')


# ---------------------------------------------------------------- geometry (same as render_images_only / eval_mesh_collision)

def rotation_3dfront(y_deg):
    y = np.deg2rad(y_deg)
    return np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])


def fit_to_box(m, l, h, w, px, py, pz, angle_rad):
    """render_images_only.fit_shapes_to_box_v2: bottom-centre the shape, rotate -90 deg about y, scale to (l, h, w), place."""
    bb = m.bounding_box
    bottom = bb.bounds[0] + bb.extents / 2
    bottom[1] = bb.bounds[0][1]
    m.apply_transform(np.dot(trimesh.transformations.translation_matrix(-bottom),
                             trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])))
    v = np.asarray(m.vertices)
    m.apply_scale(1 / (v.max(0) - v.min(0)))
    m.apply_scale([l, h, w])
    T = np.eye(4)
    T[:3, :3] = np.linalg.inv(rotation_3dfront(np.degrees(angle_rad)))
    T[:3, 3] = [px, py, pz]
    m.apply_transform(T)
    return m


def winding_inside(points, tris, chunk=200000):
    w = np.zeros(len(points))
    step = max(1, chunk // len(tris))
    for s in range(0, len(points), step):
        p = points[s:s + step, None, None, :]
        a, b, c = (tris[None, :, k, :] - p[:, :, 0, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        div = la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la \
            + np.einsum('ijk,ijk->ij', c, a) * lb
        w[s:s + step] = np.arctan2(det, div).sum(1) / (2 * np.pi)
    return np.abs(w) > 0.5


def any_inside(src, dst):
    lo, hi = dst.bounds
    v = src.vertices[np.all((src.vertices > lo) & (src.vertices < hi), axis=1)]
    return len(v) > 0 and bool(winding_inside(v, dst.triangles).any())


def footprint(size, t, a):
    """Rotated XZ footprint corners of a box with full size, same convention as scripts/eval_ob.py."""
    dx = np.array([1, 1, -1, -1]) * size[0] / 2
    dz = np.array([1, -1, -1, 1]) * size[2] / 2
    c, s = np.cos(a), np.sin(a)
    return np.stack([t[0] + dx * c - dz * s, t[2] + dx * s + dz * c], 1)


def poly_area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def clip(subject, clipper):
    """Sutherland-Hodgman: intersection of two convex polygons (either orientation)."""
    sign = 1.0 if np.cross(clipper[1] - clipper[0], clipper[2] - clipper[1]) > 0 else -1.0
    out = list(subject)
    for k in range(len(clipper)):
        a, b = clipper[k], clipper[(k + 1) % len(clipper)]
        inside = lambda p: sign * np.cross(b - a, p - a) >= 0
        src, out = out, []
        for i in range(len(src)):
            p, q = src[i], src[(i + 1) % len(src)]
            if inside(q):
                if not inside(p):
                    out.append(seg_x(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(seg_x(p, q, a, b))
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def seg_x(p, q, a, b):
    d1, d2 = q - p, b - a
    den = np.cross(d1, d2)
    return p if abs(den) < 1e-12 else p + d1 * (np.cross(a - p, d2) / den)


def box_collisions(objs):
    """Box-level collisions as in eval_collision.py: rotated footprints overlap by > 1e-4 m^2 and heights overlap."""
    hit = set()
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if a['y'][1] <= b['y'][0] or b['y'][1] <= a['y'][0]:
                continue
            pa, pb = np.array(a['foot']), np.array(b['foot'])
            if np.any(pa.max(0) <= pb.min(0)) or np.any(pb.max(0) <= pa.min(0)):
                continue
            inter = clip(pa, pb)
            if len(inter) >= 3 and poly_area(inter) > 1e-4:
                hit.add(a['inst']); hit.add(b['inst'])
    return sorted(hit)


def outside_floor(foot, tris, tol=1e-3):
    """True if any footprint corner lies outside every floor triangle (tolerance tol metres)."""
    for p in foot:
        ok = False
        for t in tris:
            a, b, c = np.asarray(t, float)
            if point_in_tri(p, a, b, c, tol):
                ok = True
                break
        if not ok:
            return True
    return False


def point_in_tri(p, a, b, c, tol):
    for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
        e = v - u
        n = np.linalg.norm(e)
        side = np.cross(e, p - u) / max(n, 1e-12)
        ref = np.cross(e, w - u)
        if (side * np.sign(ref)) < -tol:
            return False
    return True


def room_of(uid):
    tail = uid.split('_')[-1] if '_' in uid else uid
    return ''.join(ch for ch in tail.split('-')[0] if ch.isalpha()) or 'scene'


def read_bytes(spec):
    """'path' or 'archive.zip!member' -> bytes."""
    if '!' in spec:
        z, member = spec.split('!', 1)
        with zipfile.ZipFile(z) as f:
            return f.read(member)
    with open(spec, 'rb') as f:
        return f.read()


def as_list(x):
    return x if isinstance(x, list) else [x]


_BOUNDS = {}


def obj_bounds(path):
    """min/max of an obj's vertices, from its 'v' lines only."""
    if path not in _BOUNDS:
        with open(path, 'rb') as f:
            v = np.array([l.split()[1:4] for l in f if l.startswith(b'v ')], dtype=float)
        _BOUNDS[path] = (v.min(0), v.max(0))
    return _BOUNDS[path]


# ---------------------------------------------------------------- run adapters

class MeshSource:
    """object_meshes/<scene>/<label>_<cat>_<inst>.obj inside a zip or a folder."""

    def __init__(self, path):
        self.path, self.files = path, {}
        if path.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        else:
            names = [os.path.join(os.path.abspath(r), f) for r, _, fs in os.walk(path) for f in fs]
        for n in names:
            m = MESH_RE.search(n.replace(os.sep, '/'))
            if m:
                self.files.setdefault(m.group(1), {})[m.group(2)] = n
        self._local = threading.local()

    def find(self, sid, cat, inst):
        suffix = '_%d_%d' % (cat, inst)
        for stem, name in self.files.get(sid, {}).items():
            if stem.endswith(suffix):
                return stem.split('_')[0], name
        return None, None

    def load(self, name):
        if self.path.endswith('.zip'):
            z = getattr(self._local, 'zip', None)
            if z is None:
                z = self._local.zip = zipfile.ZipFile(self.path)
            data = z.read(name)
        else:
            with open(name, 'rb') as f:
                data = f.read()
        return trimesh.load(io.BytesIO(data), file_type='obj', force='mesh', process=False)


class EchoRun:
    """EchoScene layout JSON + generated meshes."""
    mesh_level = True

    def __init__(self, layout, meshes):
        self.d = json.loads(read_bytes(layout))
        self.ms = MeshSource(meshes)
        self.idx = {s: k for k, s in enumerate(self.d['scene_ids'])}

    def ids(self):
        return list(self.d['scene_ids'])

    def _boxes(self, i):
        d = self.d
        cats = np.argmax(np.array(d['class_labels'][i]), 1)
        obj = [o[0] for o in d['objectness'][i]]
        ang = np.array(d['angles'][i], dtype=float).reshape(-1)
        if np.abs(ang).max() > 6.29:
            ang = np.radians(ang)
        return cats, obj, np.array(d['sizes'][i], dtype=float), np.array(d['translations'][i], dtype=float), ang

    def _objects(self, sid):
        """Objects with a mesh, in eval_mesh_collision.py's order: (box index, inst, label, member)."""
        cats, obj, _, _, _ = self._boxes(self.idx[sid])
        out, inst = [], 1
        for j, c in enumerate(cats):
            if obj[j] != 1.0:
                continue
            label, name = self.ms.find(sid, int(c), inst)
            inst += 1
            if name:
                out.append((j, inst - 1, label, name))
        return out

    def scene(self, sid):
        _, _, sizes, trans, ang = self._boxes(self.idx[sid])
        flat = [j for j in range(len(sizes)) if abs(sizes[j][1]) < 0.1 and sizes[j][0] * sizes[j][2] > 0.5]
        tris = None
        if flat:
            f = max(flat, key=lambda j: sizes[j][0] * sizes[j][2])
            fp = footprint(sizes[f], trans[f], ang[f]).round(4).tolist()
            tris = [[fp[0], fp[1], fp[2]], [fp[0], fp[2], fp[3]]]
        objs = [dict(inst=inst, label=label, foot=footprint(sizes[j], trans[j], ang[j]).tolist(),
                     y=[trans[j][1], trans[j][1] + sizes[j][1]], t=[trans[j][0], trans[j][2]])
                for j, inst, label, _ in self._objects(sid)]
        return dict(room=room_of(sid), floor=tris, objects=objs)

    def meshes(self, sid):
        _, _, sizes, trans, ang = self._boxes(self.idx[sid])
        return [(inst, label, fit_to_box(self.ms.load(name), *sizes[j], *trans[j], ang[j]))
                for j, inst, label, name in self._objects(sid)]


def place_matrix(o):
    """v_world = (v * scale - center) . R + translation as a 4x4 acting on column vectors."""
    t = o['theta']
    R = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
    M = np.eye(4)
    M[:3, :3] = R.T @ np.diag(o['scale'])
    M[:3, 3] = np.array(o['translation']) - R.T @ np.array(o['center'])
    return M


class PlacementRun:
    """InstructScene-style placements (+ floor JSON) and the 3D-FUTURE meshes they use. Scenes are matched across runs
    by position, so every run of a backbone must list the same test scenes in the same order."""
    mesh_level = False

    def __init__(self, layout, floor, mesh_root):
        self.scenes, self.floors = [], []
        for f in as_list(layout):
            self.scenes += json.load(open(f))
        for f in as_list(floor or []):
            fl = json.load(open(f))
            self.floors += [dict(xz=a, faces=b) for a, b in zip(fl['floor_xz'], fl['floor_faces'])]
        self.mesh_root = mesh_root
        self.sid = ['%04d_%s' % (k, s['scene_uid']) for k, s in enumerate(self.scenes)]
        self.idx = {s: k for k, s in enumerate(self.sid)}

    def ids(self):
        return list(self.sid)

    def path(self, o):
        p = os.path.join(self.mesh_root, o['jid'], 'raw_model.obj') if self.mesh_root else o['mesh']
        return p if os.path.exists(p) or not o.get('mesh') else o['mesh']

    def scene(self, sid):
        k = self.idx[sid]
        tris = None
        if k < len(self.floors):
            xz = np.array(self.floors[k]['xz'])
            tris = [[xz[a].round(4).tolist() for a in f] for f in self.floors[k]['faces']]
        objs = []
        for n, o in enumerate(self.scenes[k]['objects'], 1):
            lo, hi = obj_bounds(self.path(o))
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            w = (place_matrix(o) @ np.c_[corners, np.ones(8)].T).T[:, :3]
            foot = w[[0, 1, 5, 4], :][:, [0, 2]]  # y = lo corners: (lo,lo), (lo,hi), (hi,hi), (hi,lo) in x,z
            objs.append(dict(inst=n, label=o.get('label', 'object'), foot=foot.tolist(),
                             y=[float(w[:, 1].min()), float(w[:, 1].max())], t=[o['translation'][0], o['translation'][2]]))
        return dict(room=room_of(self.scenes[k]['scene_uid']), floor=tris, objects=objs)

    def meshes(self, sid):
        out = []
        for n, o in enumerate(self.scenes[self.idx[sid]]['objects'], 1):
            m = trimesh.load(self.path(o), file_type='obj', force='mesh', process=False)
            m.apply_transform(place_matrix(o))
            out.append((n, o.get('label', 'object'), m))
        return out


class BoxRun:
    """Boxes only (scene_ids, class_labels, translations, sizes, angles); e.g. DiffuScene (half sizes)."""
    mesh_level = False

    def __init__(self, layout, classes=None, half_sizes=False, skip=('empty', 'floor', 'start', 'end')):
        self.d, self.src = {}, []
        for n, f in enumerate(as_list(layout)):
            part = json.load(open(f))
            for k, v in part.items():
                self.d.setdefault(k, []).extend(v)
            self.src += [n] * len(part['scene_ids'])
        # classes: one list, or one list per layout file (rooms with different class sets)
        self.classes = classes if classes and isinstance(classes[0], list) else [classes] * len(as_list(layout))
        self.half, self.skip = half_sizes, set(skip)
        self.sid = ['%04d_%s' % (k, s) for k, s in enumerate(self.d['scene_ids'])]
        self.idx = {s: k for k, s in enumerate(self.sid)}

    def ids(self):
        return list(self.sid)

    def _objs(self, sid):
        i = self.idx[sid]
        cl = np.array(self.d['class_labels'][i])
        tr, sz = np.array(self.d['translations'][i], float), np.array(self.d['sizes'][i], float) * (2 if self.half else 1)
        ang = np.array(self.d['angles'][i], float).reshape(-1)
        out, names = [], self.classes[self.src[i]]
        for j in range(len(tr)):
            c = int(cl[j].argmax())
            label = names[c] if names and c < len(names) else 'c%d' % c
            if label in self.skip:
                continue
            out.append((j + 1, label, sz[j], tr[j], ang[j]))
        return out

    def scene(self, sid):
        return dict(room=room_of(self.d['scene_ids'][self.idx[sid]]), floor=None,
                    objects=[dict(inst=n, label=l, foot=footprint(s, t, a).tolist(), y=[t[1] - s[1] / 2, t[1] + s[1] / 2],
                                  t=[t[0], t[2]]) for n, l, s, t, a in self._objs(sid)])

    def meshes(self, sid):
        out = []
        for n, l, s, t, a in self._objs(sid):
            m = trimesh.creation.box(extents=s)
            T = np.eye(4)
            T[:3, :3] = np.linalg.inv(rotation_3dfront(np.degrees(a)))
            T[:3, 3] = t
            m.apply_transform(T)
            out.append((n, l, m))
        return out


def make_run(r, stage):
    kind = r.get('kind', 'echoscene')
    if kind == 'echoscene':
        return EchoRun(r[stage], r['meshes'])
    if kind == 'placements':
        return PlacementRun(r[stage], r.get('floor', {}).get(stage), r.get('mesh_root'))
    if kind == 'boxes':
        return BoxRun(r[stage], r.get('classes'), r.get('half_sizes', False))
    raise ValueError('unknown run kind ' + kind)


# ---------------------------------------------------------------- index

def mesh_hits_job(args):
    rcfg, stage, sids = args
    run = make_run(rcfg, stage)
    res = {}
    for sid in sids:
        meshes = [m for _, _, m in run.meshes(sid)]
        hit = np.zeros(len(meshes), bool)
        for a in range(len(meshes)):
            for b in range(a + 1, len(meshes)):
                if hit[a] and hit[b]:
                    continue
                (l1, h1), (l2, h2) = meshes[a].bounds, meshes[b].bounds
                if np.any(h1 <= l2) or np.any(h2 <= l1):
                    continue
                if any_inside(meshes[b], meshes[a]) or any_inside(meshes[a], meshes[b]):
                    hit[a] = hit[b] = True
        res[sid] = hit.tolist()
    return res


def build_index(name, b, cache, workers):
    runs, panels = b['runs'], b['panels']
    stages = [(p['run'], s) for p in panels for s in ('raw', 'pp') if s in runs[p['run']]]
    stages = list(dict.fromkeys(stages))
    loaded = {rs: make_run(runs[rs[0]], rs[1]) for rs in stages}
    ref = loaded[(panels[0]['run'], panels[0]['stage'])]
    sids = ref.ids()
    for rs, run in loaded.items():
        if len(run.ids()) != len(sids):
            print('[index] %s: %s/%s has %d scenes, %s/%s has %d' % (name, rs[0], rs[1], len(run.ids()),
                  panels[0]['run'], panels[0]['stage'], len(sids)), flush=True)
    scenes, levels = {}, set()
    for sid in sids:
        s = ref.scene(sid)
        scenes[sid] = dict(id=sid, room=s['room'], floor=s['floor'],
                           objects=[dict(inst=o['inst'], label=o['label']) for o in s['objects']], stages={})
    for (r, st), run in loaded.items():
        rc = runs[r]
        given = rc.get('mesh_eval', {}).get(st)
        hits = None
        if given and os.path.exists(given):
            per = json.load(open(given))
            hits = {k: v['hit'] for k, v in per.items()}
            print('[index] %s %s/%s: mesh collisions from %s' % (name, r, st, given), flush=True)
        elif run.mesh_level and rc.get('compute_mesh_collisions', False):
            print('[index] %s %s/%s: computing mesh collisions...' % (name, r, st), flush=True)
            hits = {}
            chunks = [sids[k::workers * 4] for k in range(workers * 4)]
            with Pool(workers) as p:
                for part in p.imap_unordered(mesh_hits_job, [(rc, st, c) for c in chunks]):
                    hits.update(part)
        levels.add('mesh' if hits is not None else 'box')
        raw = loaded.get((r, 'raw')) if st == 'pp' else None
        ids = set(run.ids())
        for sid in sids:
            if sid not in ids:
                continue
            s = run.scene(sid)
            objs = s['objects']
            col = [o['inst'] for o, h in zip(objs, hits.get(sid, [])) if h] if hits is not None else box_collisions(objs)
            floor = s['floor'] or scenes[sid]['floor']
            oob = [o['inst'] for o in objs if floor and outside_floor(o['foot'], floor)]
            moved = []
            if raw is not None:
                rt = {o['inst']: o['t'] for o in raw.scene(sid)['objects']}
                for o in objs:
                    if o['inst'] in rt:
                        dist = float(np.hypot(o['t'][0] - rt[o['inst']][0], o['t'][1] - rt[o['inst']][1]))
                        if dist > b.get('moved_threshold', 0.1):
                            moved.append([o['inst'], round(dist, 3)])
            scenes[sid]['stages']['%s/%s' % (r, st)] = dict(col=col, oob=oob, moved=moved)
        print('[index] %s %s/%s done' % (name, r, st), flush=True)
    index = dict(name=name, title=b.get('title', name), panels=panels, compare=b.get('compare', []),
                 presets=b.get('presets', {}), room_groups=b.get('room_groups', {}),
                 collision_level=' / '.join(sorted(levels)),
                 runs={r: dict(label=runs[r].get('label', r), stages=[s for s in ('raw', 'pp') if s in runs[r]]) for r in runs},
                 scenes=[scenes[s] for s in sids])
    json.dump(index, open(os.path.join(cache, 'index_%s.json' % safe(name)), 'w'))
    return index


def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


# ---------------------------------------------------------------- GLB export

class SceneBuilder:
    def __init__(self, backbones, cache):
        self.bb, self.cache = backbones, cache
        self.runs, self.lock = {}, threading.Lock()

    def run(self, b, r, s):
        with self.lock:
            if (b, r, s) not in self.runs:
                self.runs[(b, r, s)] = make_run(self.bb[b]['runs'][r], s)
            return self.runs[(b, r, s)]

    def path(self, b, r, s, sid):
        return os.path.join(self.cache, 'glb', safe(b), r, s, sid + '.glb')

    def glb(self, b, r, s, sid):
        out = self.path(b, r, s, sid)
        if os.path.exists(out):
            return out
        sc = trimesh.Scene()
        for inst, label, m in self.run(b, r, s).meshes(sid):
            m.visual = trimesh.visual.ColorVisuals(m)  # geometry only; the viewer colours by class
            nm = 'obj_%d_%s' % (inst, label)
            sc.add_geometry(m, node_name=nm, geom_name=nm)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + '.%d.%d.tmp' % (os.getpid(), threading.get_ident())
        with open(tmp, 'wb') as f:
            f.write(sc.export(file_type='glb'))
        os.replace(tmp, out)
        return out


_WARM = None


def _warm_init(backbones, cache):
    global _WARM
    _WARM = SceneBuilder(backbones, cache)


def _warm_job(job):
    try:
        _WARM.glb(*job)
    except Exception as e:  # a broken scene must not stop the warm-up
        return '%s: %s' % ('/'.join(job), e)


def warm_cache(backbones, cache, indexes, workers):
    """Pre-build every panel's GLB in background processes, so browsing never waits on mesh fitting."""
    probe = SceneBuilder(backbones, cache)
    jobs = []
    for b, index in indexes.items():
        rs = list(dict.fromkeys((p['run'], s) for p in backbones[b]['panels'] for s in ('raw', 'pp') if s in backbones[b]['runs'][p['run']]))
        jobs += [(b, r, s, sc['id']) for sc in index['scenes'] for r, s in rs if not os.path.exists(probe.path(b, r, s, sc['id']))]
    if not jobs:
        return
    print('[warm] building %d scene files with %d processes' % (len(jobs), workers), flush=True)
    with Pool(workers, initializer=_warm_init, initargs=(backbones, cache)) as p:
        for k, err in enumerate(p.imap_unordered(_warm_job, jobs, chunksize=2), 1):
            if err:
                print('[warm] failed %s' % err, flush=True)
            if k % 200 == 0 or k == len(jobs):
                print('[warm] %d / %d' % (k, len(jobs)), flush=True)


# ---------------------------------------------------------------- HTTP

def make_handler(indexes, builder, cache):
    names = list(indexes)

    def picks_path(b):
        return os.path.join(cache, 'picks_%s.json' % safe(b))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=86400' if ctype == 'model/gltf-binary' else 'no-cache')
            self.end_headers()
            self.wfile.write(body)

        def backbone(self):
            b = parse_qs(urlparse(self.path).query).get('b', [names[0]])[0]
            return b if b in indexes else names[0]

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ('/', '/index.html'):
                return self.send(200, open(os.path.join(HERE, 'viewer.html'), 'rb').read(), 'text/html; charset=utf-8')
            if p == '/api/backbones':
                return self.send(200, json.dumps([dict(name=n, title=indexes[n]['title'], scenes=len(indexes[n]['scenes']))
                                                  for n in names]).encode(), 'application/json')
            if p == '/api/index':
                return self.send(200, json.dumps(indexes[self.backbone()]).encode(), 'application/json')
            if p == '/api/picks':
                f = picks_path(self.backbone())
                return self.send(200, open(f, 'rb').read() if os.path.exists(f) else b'[]', 'application/json')
            m = re.match(r'^/glb/([^/]+)/([^/]+)/(raw|pp)/([^/]+)\.glb$', p)
            if m:
                b, r, s, sid = (unquote(x) for x in m.groups())
                if b in indexes and r in builder.bb[b]['runs'] and s in builder.bb[b]['runs'][r]:
                    try:
                        f = builder.glb(b, r, s, sid)
                    except (ValueError, KeyError, IndexError) as e:
                        return self.send(404, str(e).encode(), 'text/plain')
                    return self.send(200, open(f, 'rb').read(), 'model/gltf-binary')
            self.send(404, b'not found', 'text/plain')

        def do_POST(self):
            if urlparse(self.path).path == '/api/picks':
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                json.loads(body)
                f = picks_path(self.backbone())
                with open(f + '.tmp', 'wb') as fh:
                    fh.write(body)
                os.replace(f + '.tmp', f)
                return self.send(200, b'{"ok":true}', 'application/json')
            self.send(404, b'not found', 'text/plain')

    return H


def load_config(path):
    """Config is {"backbones": {name: backbone}}; a single backbone ({"runs", "panels", ...}) is also accepted.
    Relative paths in mesh_eval are relative to the config file."""
    cfg = json.load(open(path))
    backbones = cfg['backbones'] if 'backbones' in cfg else {cfg.get('title', 'default'): cfg}
    cdir = os.path.dirname(os.path.abspath(path))
    for b in backbones.values():
        for r in b['runs'].values():
            for s, p in r.get('mesh_eval', {}).items():
                r['mesh_eval'][s] = os.path.join(cdir, p)
    return backbones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, 'config_lhanubu.json'))
    ap.add_argument('--cache', default=None, help='default: <config dir>/qual_cache')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to allow other machines on the LAN')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--rebuild', action='store_true', help='recompute the per-scene index')
    a = ap.parse_args()
    backbones = load_config(a.config)
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.config)), 'qual_cache')
    os.makedirs(cache, exist_ok=True)
    indexes = {}
    for name, b in backbones.items():
        ip = os.path.join(cache, 'index_%s.json' % safe(name))
        try:
            if a.rebuild or not os.path.exists(ip) or os.path.getmtime(ip) < os.path.getmtime(a.config):
                indexes[name] = build_index(name, b, cache, a.workers)
            else:
                indexes[name] = json.load(open(ip))
        except (OSError, KeyError, ValueError) as e:  # a backbone whose files are missing is skipped, not fatal
            print('[index] skipping %s: %s' % (name, e), flush=True)
    backbones = {n: backbones[n] for n in indexes}
    builder = SceneBuilder(backbones, cache)
    threading.Thread(target=warm_cache, args=(backbones, cache, indexes, max(1, min(4, a.workers - 1))), daemon=True).start()
    print('[serve] %s -> http://%s:%d' % (', '.join('%s (%d scenes)' % (n, len(i['scenes'])) for n, i in indexes.items()),
                                          'localhost' if a.host == '127.0.0.1' else a.host, a.port), flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(indexes, builder, cache)).serve_forever()


if __name__ == '__main__':
    sys.exit(main())
