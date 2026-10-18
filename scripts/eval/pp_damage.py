"""How much does the Collision Resolver (PP) have to change each run's layouts?

For every run in pp_controlled/{raw,pp}/<name>.json (PP re-run with identical settings),
measures per-object horizontal displacement caused by PP and left/right / front/behind
order reversals caused by PP, then compares each guided run with the unguided baseline
using paired bootstrap CIs over the 370 shared scenes.

usage: python pp_damage.py <pp_controlled dir>
"""
import json, sys, itertools, os
import numpy as np

D = sys.argv[1]
names = sorted(f[:-5] for f in os.listdir(os.path.join(D, 'raw')) if f.endswith('.json'))
names.remove('baseline'); names = ['baseline'] + names
rng = np.random.default_rng(0)

def load(path):
    d = json.load(open(path)); out = {}
    for i, s in enumerate(d['scene_ids']):
        o = np.array(d['objectness'][i]).ravel() > 0.5
        out[s] = np.array(d['translations'][i])[o][:, [0, 2]]
    return out

def scene_stats(a, b):
    d = np.linalg.norm(b - a, axis=1)
    flips = pairs = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        for ax in (0, 1):
            da, db = a[i, ax] - a[j, ax], b[i, ax] - b[j, ax]
            if abs(da) > 0.05:
                pairs += 1
                flips += (np.sign(da) != np.sign(db)) and abs(db) > 0.05
    return d, flips, pairs

runs = {}
for n in names:
    raw, pp = load(f'{D}/raw/{n}.json'), load(f'{D}/pp/{n}.json')
    runs[n] = {s: scene_stats(raw[s], pp[s]) for s in raw}
scenes = sorted(set.intersection(*(set(r) for r in runs.values())))

def summary(n):
    r = runs[n]; disp = np.concatenate([r[s][0] for s in scenes])
    smax = np.array([r[s][0].max() if len(r[s][0]) else 0 for s in scenes])
    fl = np.array([r[s][1] for s in scenes]); pr = np.array([r[s][2] for s in scenes])
    return {'objects': len(disp), 'mean_cm': disp.mean() * 100, 'median_cm': np.median(disp) * 100,
            'p95_cm': np.percentile(disp, 95) * 100, 'obj_gt30_pct': (disp > .3).mean() * 100,
            'scene_gt50_pct': (smax > .5).mean() * 100, 'flip_pairaxes_pct': fl.sum() / pr.sum() * 100,
            'scene_flip_pct': (fl > 0).mean() * 100}

def per_scene(n):
    r = runs[n]
    return (np.array([r[s][0].mean() if len(r[s][0]) else 0 for s in scenes]),
            np.array([float(r[s][1] > 0) for s in scenes]),
            np.array([float(r[s][0].max() > .5) if len(r[s][0]) else 0. for s in scenes]))

def boot(x, B=10000):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

res = {n: summary(n) for n in names}
base = per_scene('baseline')
print(f'{len(scenes)} shared scenes\n')
hdr = f"{'run':16s} {'mean':>6s} {'med':>5s} {'p95':>6s} {'>30cm%':>7s} {'sc>50%':>7s} {'flip%':>6s} {'scFlip%':>8s}"
print(hdr)
for n in names:
    s = res[n]
    print(f"{n:16s} {s['mean_cm']:6.1f} {s['median_cm']:5.1f} {s['p95_cm']:6.1f} {s['obj_gt30_pct']:7.1f} "
          f"{s['scene_gt50_pct']:7.1f} {s['flip_pairaxes_pct']:6.2f} {s['scene_flip_pct']:8.1f}")
print('\nPaired difference vs baseline (guided - baseline), mean [95% CI] over scenes:')
print(f"{'run':16s} {'scene-mean disp (cm)':>26s} {'scenes with flip (pts)':>26s} {'scenes with >50cm (pts)':>26s}")
for n in names[1:]:
    g = per_scene(n); cells = []
    for k, scale in ((0, 100), (1, 100), (2, 100)):
        m, lo, hi = boot((g[k] - base[k]) * scale); cells.append(f'{m:+6.1f} [{lo:+5.1f},{hi:+5.1f}]')
    print(f'{n:16s} ' + ' '.join(f'{c:>26s}' for c in cells))
json.dump(res, open(f'{D}/pp_damage_summary.json', 'w'), indent=1)
