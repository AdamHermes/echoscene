"""Per-scene means with 95% paired bootstrap CIs against a reference run.

Reads connectivity_results.json (eval_connectivity.py) or navigation_results*.json
(eval_navigation.py; metric `accessibility_rate`). Scenes are paired by name; for each metric
only scenes where both runs have a finite value are used (n is reported).

  python scripts/procthor_eval/bootstrap_nav.py --ref EchoScene+PP=a.json \
      --run work41+PP=b.json --run work37+PP=c.json --metrics reach_rate pair_success
"""
import json
import argparse
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)['scenes']


def paired(ref, run, metric):
    keys = sorted(set(ref) & set(run))
    a = np.array([ref[k].get(metric, np.nan) for k in keys], dtype=float)
    b = np.array([run[k].get(metric, np.nan) for k in keys], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def boot_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(ref, runs, metrics, n_boot=10000):
    rows = []
    for metric in metrics:
        for name, run in runs:
            a, b = paired(ref, run, metric)
            if not len(a):
                continue
            d = b - a
            lo, hi = boot_ci(d, n_boot)
            mlo, mhi = boot_ci(b, n_boot)
            rows.append({'metric': metric, 'run': name, 'n': int(len(a)), 'mean': float(b.mean()),
                         'ci': [mlo, mhi], 'ref_mean': float(a.mean()), 'diff': float(d.mean()),
                         'diff_ci': [lo, hi], 'win': int((d > 0).sum()), 'loss': int((d < 0).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='NAME=path')
    ap.add_argument('--run', action='append', default=[], help='NAME=path (repeatable)')
    ap.add_argument('--metrics', nargs='+', required=True)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rname, rpath = args.ref.split('=', 1)
    ref = load(rpath)
    runs = [(n, load(p)) for n, p in (r.split('=', 1) for r in [args.ref] + args.run)]
    rows = compare(ref, runs, args.metrics, args.n_boot)
    for r in rows:
        print(f"{r['metric']:<20} {r['run']:<18} n={r['n']:<4} {r['mean']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]"
              f"  Δ vs {rname} {r['diff']:+.4f} [{r['diff_ci'][0]:+.4f},{r['diff_ci'][1]:+.4f}]  W/L {r['win']}/{r['loss']}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
