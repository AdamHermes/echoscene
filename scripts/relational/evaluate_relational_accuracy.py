#!/usr/bin/env python3
"""
Official Relational Accuracy Evaluation Script for EchoScene Datasets
(Evaluates ALL 370 scenes using room_type='all' from 3D-FRONT relationships_all_test.json)

Usage:
    python scripts/relational/evaluate_relational_accuracy.py --json <path_to_json> [--limit N]
"""

import os
import sys
import json
import argparse
from unittest.mock import MagicMock

# Mock non-essential visual dependencies if missing
for mod in ['pytorch3d', 'pytorch3d.structures', 'pytorch3d.io', 'pytorch3d.renderer', 'pytorch3d.transforms', 'mcubes', 'clip', 'helpers.psutil']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.metrics_3dfront import validate_constrains

def evaluate_relational_accuracy(json_path, limit=None, front_root=None):
    if not front_root:
        front_root = os.path.join(REPO_ROOT, 'FRONT')

    # Load room_type='all' to cover ALL 370 scenes across all room categories!
    ds_all = ThreedFrontDatasetSceneGraph(
        root=front_root,
        split='val_scans',
        use_scene_rels=True,
        with_changes=False,
        eval=True,
        eval_type='none',
        with_CLIP=False,
        use_SDF=False,
        large=False,
        room_type='all'
    )
    scan_map_all = {scan_id: i for i, scan_id in enumerate(ds_all.scans)}

    with open(json_path, 'r') as f:
        data_export = json.load(f)

    all_scene_ids = data_export['scene_ids']
    scene2idx = {sid: i for i, sid in enumerate(all_scene_ids)}

    accuracy = {k: [] for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']}

    selected_ids = all_scene_ids[:limit] if limit else all_scene_ids
    eval_count = 0

    for sid in selected_ids:
        if sid not in scan_map_all:
            continue

        target_idx = scan_map_all[sid]
        eval_count += 1
        item = ds_all[target_idx]
        idx = scene2idx[sid]
        dec_triples = item['decoder']['triples']

        sizes = torch.tensor(data_export['sizes'][idx], dtype=torch.float32)
        trans = torch.tensor(data_export['translations'][idx], dtype=torch.float32)
        angles_rad = torch.tensor(data_export['angles'][idx], dtype=torch.float32)
        angles_deg = angles_rad * (180.0 / np.pi)

        boxes_pred_den = torch.cat([sizes, trans], dim=-1)

        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_deg, None, ds_all.vocab, accuracy)

    def _safe_m(k):
        return float(np.mean(accuracy[k])) if len(accuracy[k]) > 0 else float('nan')

    lr_mean = np.nanmean([_safe_m('left'), _safe_m('right')])
    fb_mean = np.nanmean([_safe_m('front'), _safe_m('behind')])
    bism_mean = np.nanmean([_safe_m('bigger'), _safe_m('smaller')])
    tash_mean = np.nanmean([_safe_m('taller'), _safe_m('shorter')])
    stand_mean = _safe_m('standing on')
    close_mean = _safe_m('close by')
    symm_mean = _safe_m('symmetrical to')
    total_mean = _safe_m('total')

    means_of_mean = float(np.nanmean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean]))

    return {
        'eval_count': eval_count,
        'total_scenes': len(all_scene_ids),
        'total_acc': total_mean * 100,
        'macro_avg': means_of_mean * 100,
        'lr_mean': lr_mean * 100,
        'fb_mean': fb_mean * 100,
        'bism_mean': bism_mean * 100,
        'tash_mean': tash_mean * 100,
        'stand_mean': stand_mean * 100,
        'close_mean': close_mean * 100,
        'symm_mean': symm_mean * 100,
        'left': _safe_m('left') * 100,
        'right': _safe_m('right') * 100,
        'front': _safe_m('front') * 100,
        'behind': _safe_m('behind') * 100,
        'bigger': _safe_m('bigger') * 100,
        'smaller': _safe_m('smaller') * 100,
        'taller': _safe_m('taller') * 100,
        'shorter': _safe_m('shorter') * 100,
    }

def write_report_file(json_path, res):
    out_txt_path = os.path.join(os.path.dirname(json_path), 'relational_accuracy_report.txt')
    with open(out_txt_path, 'w') as out_f:
        out_f.write('='*75 + '\n')
        out_f.write('  RELATIONAL ACCURACY EVALUATION REPORT\n')
        out_f.write(f'  Source Path: {json_path}\n')
        out_f.write(f'  Evaluated Scenes: {res["eval_count"]} / {res["total_scenes"]}\n')
        out_f.write('='*75 + '\n')
        out_f.write(f'  Total Accuracy  (Micro-Average): {res["total_acc"]:.2f}%\n')
        out_f.write(f'  Means of Means  (Macro-Average): {res["macro_avg"]:.2f}%\n')
        out_f.write('-'*75 + '\n')
        out_f.write('  Category Breakdown:\n')
        out_f.write(f'    - Left / Right (L/R)       : {res["lr_mean"]:.2f}%  (Left: {res["left"]:.2f}%, Right: {res["right"]:.2f}%)\n')
        out_f.write(f'    - Front / Behind (F/B)     : {res["fb_mean"]:.2f}%  (Front: {res["front"]:.2f}%, Behind: {res["behind"]:.2f}%)\n')
        out_f.write(f'    - Bigger / Smaller (Bi/Sm) : {res["bism_mean"]:.2f}%  (Bigger: {res["bigger"]:.2f}%, Smaller: {res["smaller"]:.2f}%)\n')
        out_f.write(f'    - Taller / Shorter (Ta/Sh) : {res["tash_mean"]:.2f}%  (Taller: {res["taller"]:.2f}%, Shorter: {res["shorter"]:.2f}%)\n')
        out_f.write(f'    - Standing On              : {res["stand_mean"]:.2f}%\n')
        out_f.write(f'    - Close By                 : {res["close_mean"]:.2f}%\n')
        out_f.write(f'    - Symmetrical To           : {res["symm_mean"]:.2f}%\n')
        out_f.write('='*75 + '\n')
    print(f'Wrote report to: {out_txt_path}')

def main():
    parser = argparse.ArgumentParser(description="Evaluate relational constraint accuracy across ALL 370 scenes.")
    parser.add_argument("--json", type=str, required=True, help="Path to prediction JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N scenes (e.g. 20)")
    args = parser.parse_args()

    res = evaluate_relational_accuracy(args.json, limit=args.limit)
    write_report_file(args.json, res)

    print('='*75)
    print(f'          RELATIONAL ACCURACY EVALUATION REPORT          ')
    print('='*75)
    print(f' Source File                            : {args.json}')
    print(f' Evaluated Scenes                       : {res["eval_count"]} / {res["total_scenes"]}')
    print(f' Micro-Average Accuracy (Total Acc)     : {res["total_acc"]:.2f}%')
    print(f' Macro-Average Accuracy (Means of Means): {res["macro_avg"]:.2f}%')
    print('-'*75)
    print(f' Left / Right (L/R)       : {res["lr_mean"]:.2f}%  (Left: {res["left"]:.2f}%, Right: {res["right"]:.2f}%)')
    print(f' Front / Behind (F/B)     : {res["fb_mean"]:.2f}%  (Front: {res["front"]:.2f}%, Behind: {res["behind"]:.2f}%)')
    print(f' Bigger / Smaller (Bi/Sm) : {res["bism_mean"]:.2f}%  (Bigger: {res["bigger"]:.2f}%, Smaller: {res["smaller"]:.2f}%)')
    print(f' Taller / Shorter (Ta/Sh) : {res["tash_mean"]:.2f}%  (Taller: {res["taller"]:.2f}%, Shorter: {res["shorter"]:.2f}%)')
    print(f' Standing On              : {res["stand_mean"]:.2f}%')
    print(f' Close By                 : {res["close_mean"]:.2f}%')
    print(f' Symmetrical To           : {res["symm_mean"]:.2f}%')
    print('='*75)

if __name__ == '__main__':
    main()
