"""Qualitative-figure browser for SteerScene: side-by-side 3D views of the same test scene under several methods.

Runs on the machine that holds the generated meshes (zips or extracted run folders). On first start it builds a
per-scene index of what goes wrong in each panel (mesh-level collisions, out-of-room objects, objects the collision
resolver shoved), which the viewer ranks so scenes where the other methods fail and ours is clean come first. Scene
meshes are fitted into their boxes exactly as in the FID renders / mesh collision eval and sent as GLB on demand
(cached on disk).

usage: python server.py --config config_lhanubu.json [--port 8765] [--host 127.0.0.1]
then open http://localhost:8765 (from another machine: ssh -L 8765:localhost:8765 <host>).
Only needs numpy + trimesh (python >= 3.8).
"""
import argparse, io, json, os, re, sys, threading, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from multiprocessing import Pool
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


# ---------------------------------------------------------------- data sources

def read_bytes(spec):
    """'path' or 'archive.zip!member' -> bytes."""
    if '!' in spec:
        z, member = spec.split('!', 1)
        with zipfile.ZipFile(z) as f:
            return f.read(member)
    with open(spec, 'rb') as f:
        return f.read()


class MeshSource:
    """object_meshes/<scene>/<label>_<cat>_<inst>.obj inside a zip or a folder."""

    def __init__(self, path):
        self.path, self.files = path, {}
        if path.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        else:  # a run folder (or its object_meshes folder): keep absolute paths
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


def scene_boxes(d, i):
    cats = np.argmax(np.array(d['class_labels'][i]), 1)
    obj = [o[0] for o in d['objectness'][i]]
    ang = np.array(d['angles'][i], dtype=float).reshape(-1)
    if np.abs(ang).max() > 6.29:
        ang = np.radians(ang)
    return cats, obj, np.array(d['sizes'][i], dtype=float), np.array(d['translations'][i], dtype=float), ang


def footprint(size, t, a):
    """Rotated XZ footprint corners, same convention as scripts/eval_ob.py."""
    dx = np.array([1, 1, -1, -1]) * size[0] / 2
    dz = np.array([1, -1, -1, 1]) * size[2] / 2
    c, s = np.cos(a), np.sin(a)
    return np.stack([t[0] + dx * c - dz * s, t[2] + dx * s + dz * c], 1)


def floor_index(sizes):
    flat = [j for j in range(len(sizes)) if abs(sizes[j][1]) < 0.1 and sizes[j][0] * sizes[j][2] > 0.5]
    return max(flat, key=lambda j: sizes[j][0] * sizes[j][2]) if flat else None


def objects_of(d, i, ms, sid):
    """Objects that get a mesh, in the order eval_mesh_collision.py uses: (box index j, inst, label, member)."""
    cats, obj, _, _, _ = scene_boxes(d, i)
    out, inst = [], 1
    for j, c in enumerate(cats):
        if obj[j] != 1.0:
            continue
        label, name = ms.find(sid, int(c), inst)
        inst += 1
        if name:
            out.append((j, inst - 1, label, name))
    return out


# ---------------------------------------------------------------- index

def mesh_hits_job(args):
    mesh_path, layout_spec, sids = args
    ms, d = MeshSource(mesh_path), json.loads(read_bytes(layout_spec))
    idx = {s: k for k, s in enumerate(d['scene_ids'])}
    res = {}
    for sid in sids:
        i = idx[sid]
        _, _, sizes, trans, ang = scene_boxes(d, i)
        meshes = []
        for j, inst, label, name in objects_of(d, i, ms, sid):
            meshes.append(fit_to_box(ms.load(name), *sizes[j], *trans[j], ang[j]))
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


def build_index(cfg, cache, workers):
    runs, panels = cfg['runs'], cfg['panels']
    stages = sorted({(r, s) for r in runs for s in ('raw', 'pp') if s in runs[r]})
    layouts = {(r, s): json.loads(read_bytes(runs[r][s])) for r, s in stages}
    sources = {r: MeshSource(runs[r]['meshes']) for r in runs}
    ref_r, ref_s = panels[0]['run'], panels[0]['stage']
    sids = layouts[(ref_r, ref_s)]['scene_ids']
    scenes = {}
    for sid in sids:
        d = layouts[(ref_r, ref_s)]
        i = d['scene_ids'].index(sid)
        _, _, sizes, trans, ang = scene_boxes(d, i)
        f = floor_index(sizes)
        objs = objects_of(d, i, sources[ref_r], sid)
        scenes[sid] = dict(id=sid, room=''.join(ch for ch in sid.split('-')[0] if ch.isalpha()),
                           floor=footprint(sizes[f], trans[f], ang[f]).round(4).tolist() if f is not None else None,
                           objects=[dict(inst=o[1], label=o[2]) for o in objs], stages={})
    for (r, s) in stages:
        d, ms = layouts[(r, s)], sources[r]
        # mesh collisions: reuse an eval_mesh_collision.py --out file when given, else compute
        hits = {}
        given = runs[r].get('mesh_eval', {}).get(s)
        if given and os.path.exists(given):
            per = json.load(open(given))
            hits = {k: v['hit'] for k, v in per.items()}
            print('[index] %s/%s: mesh collisions from %s' % (r, s, given), flush=True)
        else:
            print('[index] %s/%s: computing mesh collisions (%d scenes, %d workers)...' % (r, s, len(sids), workers), flush=True)
            chunks = [sids[k::workers * 4] for k in range(workers * 4)]
            with Pool(workers) as p:
                for part in p.imap_unordered(mesh_hits_job, [(runs[r]['meshes'], runs[r][s], c) for c in chunks]):
                    hits.update(part)
        raw = layouts.get((r, 'raw'))
        for i, sid in enumerate(d['scene_ids']):
            if sid not in scenes:
                continue
            _, _, sizes, trans, ang = scene_boxes(d, i)
            f = floor_index(sizes)
            objs = objects_of(d, i, ms, sid)
            h = hits.get(sid, [False] * len(objs))
            col = [o[1] for o, x in zip(objs, h) if x]
            oob, moved = [], []
            if f is not None:
                fp = footprint(sizes[f], trans[f], ang[f])
                c0, e1, e2 = fp[2], fp[1] - fp[2], fp[3] - fp[2]
                for j, inst, _, _ in objs:
                    q = footprint(sizes[j], trans[j], ang[j]) - c0
                    u, v = q @ e1 / (e1 @ e1), q @ e2 / (e2 @ e2)
                    tol1, tol2 = 1e-3 / np.sqrt(e1 @ e1), 1e-3 / np.sqrt(e2 @ e2)
                    if np.any(u < -tol1) or np.any(u > 1 + tol1) or np.any(v < -tol2) or np.any(v > 1 + tol2):
                        oob.append(inst)
            if s == 'pp' and raw is not None:
                ri = raw['scene_ids'].index(sid)
                rt = np.array(raw['translations'][ri], dtype=float)
                for j, inst, _, _ in objs:
                    dist = float(np.hypot(*(trans[j][[0, 2]] - rt[j][[0, 2]])))
                    if dist > cfg.get('moved_threshold', 0.1):
                        moved.append([inst, round(dist, 3)])
            scenes[sid]['stages']['%s/%s' % (r, s)] = dict(col=col, oob=oob, moved=moved)
    index = dict(title=cfg.get('title', 'SteerScene qualitative'), panels=panels, compare=cfg.get('compare', []),
                 room_groups=cfg.get('room_groups', {}),
                 runs={r: dict(label=runs[r].get('label', r), stages=[s for s in ('raw', 'pp') if s in runs[r]]) for r in runs},
                 scenes=[scenes[s] for s in sids])
    json.dump(index, open(os.path.join(cache, 'index.json'), 'w'))
    return index


# ---------------------------------------------------------------- GLB export

class SceneBuilder:
    def __init__(self, cfg, cache):
        self.cfg, self.cache = cfg, cache
        self.sources = {r: MeshSource(cfg['runs'][r]['meshes']) for r in cfg['runs']}
        self.layouts, self.lock = {}, threading.Lock()

    def layout(self, r, s):
        with self.lock:
            if (r, s) not in self.layouts:
                self.layouts[(r, s)] = json.loads(read_bytes(self.cfg['runs'][r][s]))
            return self.layouts[(r, s)]

    def glb(self, r, s, sid):
        out = os.path.join(self.cache, 'glb', r, s, sid + '.glb')
        if os.path.exists(out):
            return out
        d, ms = self.layout(r, s), self.sources[r]
        i = d['scene_ids'].index(sid)
        _, _, sizes, trans, ang = scene_boxes(d, i)
        sc = trimesh.Scene()
        for j, inst, label, name in objects_of(d, i, ms, sid):
            m = fit_to_box(ms.load(name), *sizes[j], *trans[j], ang[j])
            nm = 'obj_%d_%s' % (inst, label)
            sc.add_geometry(m, node_name=nm, geom_name=nm)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + '.%d.%d.tmp' % (os.getpid(), threading.get_ident())
        with open(tmp, 'wb') as f:
            f.write(sc.export(file_type='glb'))
        os.replace(tmp, out)
        return out


# ---------------------------------------------------------------- HTTP

_WARM = None


def _warm_init(cfg, cache):
    global _WARM
    _WARM = SceneBuilder(cfg, cache)


def _warm_job(job):
    try:
        _WARM.glb(*job)
    except Exception as e:  # a broken scene must not stop the warm-up
        return '%s: %s' % ('/'.join(job), e)


def warm_cache(cfg, cache, index, workers):
    """Pre-build every panel's GLB in background processes, so browsing never waits on mesh fitting."""
    jobs = [(p['run'], s, sc['id']) for sc in index['scenes'] for p in cfg['panels'] for s in cfg['runs'][p['run']]
            if s in ('raw', 'pp') and not os.path.exists(os.path.join(cache, 'glb', p['run'], s, sc['id'] + '.glb'))]
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        return
    print('[warm] building %d scene files with %d processes' % (len(jobs), workers), flush=True)
    with Pool(workers, initializer=_warm_init, initargs=(cfg, cache)) as p:
        for k, err in enumerate(p.imap_unordered(_warm_job, jobs, chunksize=2), 1):
            if err:
                print('[warm] failed %s' % err, flush=True)
            if k % 200 == 0 or k == len(jobs):
                print('[warm] %d / %d' % (k, len(jobs)), flush=True)


def make_handler(index, builder, cache):
    picks_path = os.path.join(cache, 'picks.json')

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

        def do_GET(self):
            p = self.path.split('?')[0]
            if p in ('/', '/index.html'):
                return self.send(200, open(os.path.join(HERE, 'viewer.html'), 'rb').read(), 'text/html; charset=utf-8')
            if p == '/api/index':
                return self.send(200, json.dumps(index).encode(), 'application/json')
            if p == '/api/picks':
                body = open(picks_path, 'rb').read() if os.path.exists(picks_path) else b'[]'
                return self.send(200, body, 'application/json')
            m = re.match(r'^/glb/([^/]+)/(raw|pp)/([^/]+)\.glb$', p)
            if m and m.group(1) in builder.cfg['runs'] and m.group(2) in builder.cfg['runs'][m.group(1)]:
                try:
                    f = builder.glb(m.group(1), m.group(2), m.group(3))
                except (ValueError, KeyError) as e:
                    return self.send(404, str(e).encode(), 'text/plain')
                return self.send(200, open(f, 'rb').read(), 'model/gltf-binary')
            self.send(404, b'not found', 'text/plain')

        def do_POST(self):
            if self.path == '/api/picks':
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                json.loads(body)
                with open(picks_path + '.tmp', 'wb') as f:
                    f.write(body)
                os.replace(picks_path + '.tmp', picks_path)
                return self.send(200, b'{"ok":true}', 'application/json')
            self.send(404, b'not found', 'text/plain')

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, 'config_lhanubu.json'))
    ap.add_argument('--cache', default=None, help='default: <config dir>/qual_cache')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to allow other machines on the LAN')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--rebuild', action='store_true', help='recompute the per-scene index')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cdir = os.path.dirname(os.path.abspath(a.config))
    for r in cfg['runs'].values():
        for s, p in r.get('mesh_eval', {}).items():
            r['mesh_eval'][s] = os.path.join(cdir, p)
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.config)), 'qual_cache')
    os.makedirs(cache, exist_ok=True)
    ip = os.path.join(cache, 'index.json')
    if a.rebuild or not os.path.exists(ip) or os.path.getmtime(ip) < os.path.getmtime(a.config):
        index = build_index(cfg, cache, a.workers)
    else:
        index = json.load(open(ip))
    builder = SceneBuilder(cfg, cache)
    threading.Thread(target=warm_cache, args=(cfg, cache, index, max(1, min(4, a.workers - 1))), daemon=True).start()
    print('[serve] %d scenes, %d panels -> http://%s:%d' % (len(index['scenes']), len(index['panels']),
                                                          'localhost' if a.host == '127.0.0.1' else a.host, a.port), flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(index, builder, cache)).serve_forever()


if __name__ == '__main__':
    sys.exit(main())
