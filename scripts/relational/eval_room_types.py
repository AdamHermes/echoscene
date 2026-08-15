import os
import sys
import json
import argparse
import numpy as np
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from unittest.mock import MagicMock

# Mock non-essential visual dependencies if missing
for mod in ['pytorch3d', 'pytorch3d.structures', 'pytorch3d.io', 'pytorch3d.renderer', 'pytorch3d.transforms', 'mcubes', 'clip', 'helpers.psutil']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.metrics_3dfront import validate_constrains

# Target experiment folders requested
DEFAULT_FOLDERS = [
    "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_relational_2",
    "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_relational_1",
    "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num27_attempt2",
    "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline",
    "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/to_be_merged/complete_released_full_model",
]


def load_3dfront_datasets(front_root):
    """Loads 3D-FRONT ground truth datasets for room type matching."""
    room_types = ['diningroom', 'bedroom', 'livingroom', 'library']
    ds_dict = {}
    scan_maps = {}

    for r in room_types:
        try:
            ds = ThreedFrontDatasetSceneGraph(
                root=front_root,
                split='val_scans',
                use_scene_rels=True,
                with_changes=False,
                eval=True,
                eval_type='none',
                with_CLIP=False,
                use_SDF=False,
                large=False,
                room_type=r
            )
            ds_dict[r] = ds
            scan_maps[r] = {scan_id: i for i, scan_id in enumerate(ds.scans)}
        except Exception:
            pass

    return ds_dict, scan_maps


def locate_physcene_input_json(folder_path):
    """Strictly locates physcene_collision_input.json for the target folder."""
    if os.path.isfile(folder_path) and folder_path.endswith('.json'):
        return folder_path

    candidate_paths = [
        os.path.join(folder_path, 'content', 'echoscene', 'released_full_model', 'vis', '2050', 'physcene_collision_input.json'),
        os.path.join(folder_path, 'vis', '2050', 'physcene_collision_input.json'),
        os.path.join(folder_path, '2050', 'physcene_collision_input.json'),
        os.path.join(folder_path, 'physcene_collision_input.json'),
    ]

    for cp in candidate_paths:
        if os.path.exists(cp):
            return cp

    for root_dir, _, files in os.walk(folder_path):
        if 'physcene_collision_input.json' in files:
            return os.path.join(root_dir, 'physcene_collision_input.json')

    return None


def evaluate_folder_for_room_type(
    folder_path, 
    room_type='diningroom', 
    start_idx=0, 
    end_idx=20,
    ds_dict=None,
    scan_maps=None,
    front_root=os.path.join(REPO_ROOT, 'FRONT')
):
    json_path = locate_physcene_input_json(folder_path)
    if not json_path or not os.path.exists(json_path):
        print(f"Error: Could not locate physcene_collision_input.json in '{folder_path}'.", flush=True)
        return None

    with open(json_path, 'r') as f:
        data_export = json.load(f)

    all_scene_ids = data_export['scene_ids']
    scene2idx = {sid: i for i, sid in enumerate(all_scene_ids)}

    # Filter scene IDs by room_type
    if room_type.lower() != 'all':
        matching_ids = [sid for sid in all_scene_ids if room_type.lower() in sid.lower()]
    else:
        matching_ids = list(all_scene_ids)

    total_matching = len(matching_ids)
    final_end_idx = min(end_idx, total_matching) if end_idx is not None else total_matching
    selected_ids = matching_ids[start_idx:final_end_idx]

    if not selected_ids:
        print(f"Warning: No scenes matching room_type='{room_type}' in range [{start_idx}:{end_idx}] for '{folder_path}'.", flush=True)
        return None

    if ds_dict is None or scan_maps is None:
        ds_dict, scan_maps = load_3dfront_datasets(front_root)

    accuracy = {k: [] for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']}

    eval_count = 0
    for sid in selected_ids:
        target_ds, target_idx = None, None
        for r, s_map in scan_maps.items():
            if sid in s_map:
                target_ds = ds_dict[r]
                target_idx = s_map[sid]
                break
        if target_ds is None:
            continue

        eval_count += 1
        item = target_ds[target_idx]
        idx = scene2idx[sid]
        dec_triples = item['decoder']['triples']

        sizes = torch.tensor(data_export['sizes'][idx], dtype=torch.float32)
        trans = torch.tensor(data_export['translations'][idx], dtype=torch.float32)
        angles_rad = torch.tensor(data_export['angles'][idx], dtype=torch.float32)
        angles_deg = angles_rad * (180.0 / np.pi)

        boxes_pred_den = torch.cat([sizes, trans], dim=-1)

        # Run direct 3D geometric constraint verification
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_deg, None, target_ds.vocab, accuracy)

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

    folder_name = os.path.basename(folder_path.rstrip('/'))
    print(f"\n=========================================================================", flush=True)
    print(f"  RELATIONAL ACCURACY REPORT: {folder_name}", flush=True)
    print(f"  Room Type Filter: {room_type.upper()}  |  Range: [{start_idx}:{final_end_idx}] ({eval_count} scenes)", flush=True)
    print(f"  Source JSON     : {json_path}", flush=True)
    print(f"=========================================================================", flush=True)
    print(f"  Total Accuracy  (Micro-Average): {total_mean * 100:.2f}%", flush=True)
    print(f"  Means of Means  (Macro-Average): {means_of_mean * 100:.2f}%", flush=True)
    print(f"-------------------------------------------------------------------------", flush=True)
    print(f"  Category Breakdown:", flush=True)
    print(f"    - Left / Right (L/R)       : {lr_mean * 100:.2f}%  (Left: {_safe_m('left')*100:.2f}%, Right: {_safe_m('right')*100:.2f}%)", flush=True)
    print(f"    - Front / Behind (F/B)     : {fb_mean * 100:.2f}%  (Front: {_safe_m('front')*100:.2f}%, Behind: {_safe_m('behind')*100:.2f}%)", flush=True)
    print(f"    - Bigger / Smaller (Bi/Sm) : {bism_mean * 100:.2f}%  (Bigger: {_safe_m('bigger')*100:.2f}%, Smaller: {_safe_m('smaller')*100:.2f}%)", flush=True)
    print(f"    - Taller / Shorter (Ta/Sh) : {tash_mean * 100:.2f}%  (Taller: {_safe_m('taller')*100:.2f}%, Shorter: {_safe_m('shorter')*100:.2f}%)", flush=True)
    print(f"    - Standing On              : {stand_mean * 100:.2f}%", flush=True)
    print(f"    - Close By                 : {close_mean * 100:.2f}%", flush=True)
    print(f"    - Symmetrical To           : {symm_mean * 100:.2f}%" if not np.isnan(symm_mean) else "    - Symmetrical To           : N/A", flush=True)
    print(f"=========================================================================\n", flush=True)

    return {
        'folder_name': folder_name,
        'folder_path': folder_path,
        'room_type': room_type,
        'total_acc': total_mean,
        'means_of_means': means_of_mean,
        'lr_mean': lr_mean,
        'fb_mean': fb_mean,
        'bism_mean': bism_mean,
        'tash_mean': tash_mean,
        'stand_mean': stand_mean,
        'close_mean': close_mean,
        'symm_mean': symm_mean,
        'eval_count': eval_count,
        'json_path': json_path
    }


def export_excel_report(results, room_type, start_idx, end_idx, output_excel):
    """Generates styled Excel report for room type relational accuracy benchmark."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{room_type.capitalize()} Relational Acc"
    ws.views.sheetView[0].showGridLines = True

    font_title = Font(name="Segoe UI", size=14, bold=True, color="FFFFFF")
    font_sub = Font(name="Segoe UI", size=10, italic=True, color="1F4E78")
    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Segoe UI", size=10, bold=True)
    
    fill_title = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_sub = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    fill_header = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_zebra = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    
    thin_border = Side(style="thin", color="D9D9D9")
    border_all = Border(left=thin_border, right=thin_border, top=thin_border, bottom=thin_border)

    ws.merge_cells("A1:K2")
    title_cell = ws["A1"]
    title_cell.value = f"  ROOM Model Relational Accuracy ({room_type.upper()} - First {end_idx - start_idx} Scenes)"
    title_cell.font = font_title
    title_cell.fill = fill_title
    title_cell.alignment = Alignment(vertical="center", horizontal="left")

    ws.merge_cells("A3:K3")
    sub_cell = ws["A3"]
    sub_cell.value = f"  Room Type: {room_type.upper()}  |  Range: [{start_idx}:{end_idx}]  |  Evaluated via physcene_collision_input.json"
    sub_cell.font = font_sub
    sub_cell.fill = fill_sub
    sub_cell.alignment = Alignment(vertical="center", horizontal="left")

    headers = [
        "Rank", "Experiment Model Folder", "Evaluated Scenes", "Total Accuracy (%)", 
        "Means of Means (%)", "L/R (%)", "F/B (%)", "Bi/Sm (%)", "Ta/Sh (%)", "Standing On (%)", "Close By (%)"
    ]

    ws.row_dimensions[5].height = 26
    for col_idx, h_text in enumerate(headers, 1):
        cell = ws.cell(row=5, column=col_idx, value=h_text)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border_all

    col_widths = {"A": 8, "B": 45, "C": 16, "D": 18, "E": 18, "F": 12, "G": 12, "H": 12, "I": 12, "J": 14, "K": 12}
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    start_row = 6
    for idx, r in enumerate(results, 1):
        row_num = start_row + idx - 1
        ws.row_dimensions[row_num].height = 24

        ws.cell(row=row_num, column=1, value=f"#{idx}").alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=2, value=r['folder_name']).alignment = Alignment(horizontal="left", vertical="center")
        ws.cell(row=row_num, column=3, value=r['eval_count']).alignment = Alignment(horizontal="center", vertical="center")

        c_tot = ws.cell(row=row_num, column=4, value=r['total_acc'])
        c_tot.number_format = '0.00%'
        c_tot.alignment = Alignment(horizontal="center", vertical="center")
        c_tot.font = font_bold

        c_mom = ws.cell(row=row_num, column=5, value=r['means_of_means'])
        c_mom.number_format = '0.00%'
        c_mom.alignment = Alignment(horizontal="center", vertical="center")
        c_mom.font = font_bold

        for c_i, key in enumerate(['lr_mean', 'fb_mean', 'bism_mean', 'tash_mean', 'stand_mean', 'close_mean'], 6):
            val = r.get(key, np.nan)
            cell = ws.cell(row=row_num, column=c_i, value=val if not np.isnan(val) else "N/A")
            if not np.isnan(val):
                cell.number_format = '0.0%'
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for col_idx in range(1, 12):
            cell = ws.cell(row=row_num, column=col_idx)
            cell.border = border_all
            if idx % 2 == 0:
                cell.fill = fill_zebra

    wb.save(output_excel)
    print(f"Excel Report saved to: {output_excel}", flush=True)


def run_room_type_benchmark(folders=None, room_type='diningroom', start_idx=0, end_idx=20, front_root=os.path.join(REPO_ROOT, 'FRONT')):
    if folders is None:
        folders = DEFAULT_FOLDERS

    print(f"=========================================================================", flush=True)
    print(f"  ROOM TYPE RELATIONAL ACCURACY BENCHMARK", flush=True)
    print(f"  Target Folders Count : {len(folders)}", flush=True)
    print(f"  Target Room Type    : '{room_type.upper()}'", flush=True)
    print(f"  Index Range         : [{start_idx} : {end_idx}] (First {end_idx - start_idx} scenes)", flush=True)
    print(f"=========================================================================\n", flush=True)

    print("Loading 3D-FRONT ground truth datasets...", flush=True)
    ds_dict, scan_maps = load_3dfront_datasets(front_root)

    results = []
    for idx, fpath in enumerate(folders, 1):
        print(f"[{idx}/{len(folders)}] Evaluating folder: {fpath} ...", flush=True)
        res = evaluate_folder_for_room_type(
            folder_path=fpath,
            room_type=room_type,
            start_idx=start_idx,
            end_idx=end_idx,
            ds_dict=ds_dict,
            scan_maps=scan_maps,
            front_root=front_root
        )
        if res:
            results.append(res)

    # Master Table
    print("\n" + "=" * 105, flush=True)
    print(f"  MASTER COMPARISON TABLE: RELATIONAL ACCURACY ({room_type.upper()} - [{start_idx}:{end_idx}])", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Folder / Model':<40} | {'Scenes':<7} | {'Total Acc':<10} | {'Means of Means':<14} | {'L/R':<7} | {'F/B':<7} | {'Bi/Sm':<7}"
    print(header, flush=True)
    print("-" * 105, flush=True)

    for r in results:
        tot_acc = f"{r['total_acc']*100:.2f}%" if not np.isnan(r['total_acc']) else "N/A"
        mom_acc = f"{r['means_of_means']*100:.2f}%" if not np.isnan(r['means_of_means']) else "N/A"
        lr = f"{r['lr_mean']*100:.1f}%" if not np.isnan(r['lr_mean']) else "N/A"
        fb = f"{r['fb_mean']*100:.1f}%" if not np.isnan(r['fb_mean']) else "N/A"
        bism = f"{r['bism_mean']*100:.1f}%" if not np.isnan(r['bism_mean']) else "N/A"

        print(f"{r['folder_name']:<40} | {r['eval_count']:<7} | {tot_acc:<10} | {mom_acc:<14} | {lr:<7} | {fb:<7} | {bism:<7}", flush=True)

    print("=" * 105 + "\n", flush=True)

    output_excel = os.path.join(REPO_ROOT, f"relational_{room_type.lower()}_{start_idx}_{end_idx}_report.xlsx")
    export_excel_report(results, room_type, start_idx, end_idx, output_excel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 3D-FRONT Relational Accuracy per Room Type (DiningRoom, LivingRoom, Bedroom) for first 20 scenes.")
    parser.add_argument("pos_room_type", type=str, nargs="?", default=None, help="Room type filter ('diningroom', 'livingroom', 'bedroom', 'library', 'all').")
    parser.add_argument("pos_start_idx", type=int, nargs="?", default=0, help="Start index (default: 0).")
    parser.add_argument("pos_end_idx", type=int, nargs="?", default=20, help="End index (default: 20).")
    parser.add_argument("--room_type", type=str, default="diningroom", help="Room type filter (default: diningroom).")
    parser.add_argument("--start_idx", type=int, default=0, help="Start scene index (default: 0).")
    parser.add_argument("--end_idx", type=int, default=20, help="End scene index (default: 20).")

    args = parser.parse_args()

    room_type = args.pos_room_type if args.pos_room_type else args.room_type
    start_idx = args.pos_start_idx if args.pos_start_idx is not None else args.start_idx
    end_idx = args.pos_end_idx if args.pos_end_idx is not None else args.end_idx

    run_room_type_benchmark(
        room_type=room_type,
        start_idx=start_idx,
        end_idx=end_idx
    )
