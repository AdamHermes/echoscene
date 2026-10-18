"""FID / FID-CLIP / KID with CleanFID against the SG-FRONT SDF ground-truth renders.

protocol "original": mirrors echoscene/scripts/compute_fid_scores_3dfront.py (ATISS/EchoScene),
    the script behind main-paper Table 1 / Supp. Table II: CPU, FID + FID-CLIP + KID,
    splits bedroom / livingroom / diningroom / all (= 314 scenes, no Library).
protocol "new": mirrors fid_eval_work33.py (Supp. Table VII): FID + KID,
    adds library and all_370 splits.

usage: python fid_eval.py <original|new> <name> <fake_render_dir> <out_json>
"""
import json, os, shutil, sys, tempfile
import torch
from cleanfid import fid

protocol, name, fake_dir, out_json = sys.argv[1:5]
GT = os.environ.get("FID_GT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset_gt_renders/sdf_renders/sdf_fov90_h8_wo_lamp_no_stool/small/test"))
BED, LIV, DIN = ['Bedroom', 'MasterBedroom', 'SecondBedroom'], ['LivingDiningRoom', 'LivingRoom'], ['LivingDiningRoom', 'DiningRoom']
splits = {'bedroom': BED, 'livingroom': LIV, 'diningroom': DIN, 'all_314': BED + ['LivingRoom', 'DiningRoom', 'LivingDiningRoom']}
if protocol == 'new':
    splits.update({'library': ['Library'], 'all_370': splits['all_314'] + ['Library']})
dev = torch.device('cpu')

def stage(src, cats, dst):
    files = sorted(f for f in os.listdir(src) if f.endswith('.png') and not f.startswith('._') and f.split('-')[0] in cats)
    for k, f in enumerate(files):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f'{k:05d}.png'))
    return len(files)

res = {}
for split, cats in splits.items():
    with tempfile.TemporaryDirectory() as real, tempfile.TemporaryDirectory() as fake:
        nr, nf = stage(GT, cats, real), stage(fake_dir, cats, fake)
        r = {'n_real': nr, 'n_fake': nf,
             'fid': fid.compute_fid(real, fake, device=dev, num_workers=0, verbose=False)}
        if protocol == 'original':
            r['fid_clip'] = fid.compute_fid(real, fake, mode='clean', model_name='clip_vit_b_32', device=dev, num_workers=0, verbose=False)
        r['kid'] = fid.compute_kid(real, fake, device=dev, num_workers=0, verbose=False)
    res[split] = r
    print(f"{protocol:8s} {name:22s} {split:10s} n={nf:3d}/{nr:3d}  FID {r['fid']:7.2f}"
          + (f"  FID-CLIP {r['fid_clip']:6.2f}" if 'fid_clip' in r else '') + f"  KID x1e3 {r['kid']*1e3:6.2f}", flush=True)
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(res, open(out_json, 'w'), indent=1)
