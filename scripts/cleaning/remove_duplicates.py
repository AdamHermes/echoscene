#!/usr/bin/env python3
"""
Single-Pass Log & Dataset Deduplication Tool for EchoScene

This script performs a 1-pass scan over log files (debug_bbox.txt, guidance_losses.txt)
and JSON dataset files to remove duplicate scene entries (e.g. from retried steps, warm-ups,
or duplicate evaluations) while preserving canonical appearance order.

Usage Examples:
    # Clean all log and JSON files in a directory (in-place with backup):
    python scripts/cleaning/remove_duplicates.py --dir current_works/work_num27_attempt3/2050 --inplace

    # Dry-run on a directory to preview duplicate removal statistics:
    python scripts/cleaning/remove_duplicates.py --dir current_works/work_num27_attempt3/2050 --dry-run

    # Deduplicate a specific debug_bbox.txt file and write to an output file:
    python scripts/cleaning/remove_duplicates.py --file current_works/work_num27_attempt3/2050/debug_bbox.txt --output debug_bbox_clean.txt
"""

import os
import sys
import re
import json
import shutil
import argparse


def clean_guidance_losses(file_path, keep='last'):
    """
    Deduplicates guidance_losses.txt in a single pass.
    
    Lines are formatted as:
    Scene <scene_id> - Final Step Collision Loss: ...
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.rstrip() for l in f]

    scene_entries = {}
    order = []
    
    for line in lines:
        if not line.strip():
            continue
        m = re.search(r'Scene\s+([A-Za-z0-9_-]+)\s+-', line)
        if m:
            scene_id = m.group(1)
            if scene_id not in scene_entries:
                order.append(scene_id)
            if keep == 'last' or scene_id not in scene_entries:
                scene_entries[scene_id] = line
        else:
            # Lines without header attached to previous entry or kept as-is
            pass

    cleaned_lines = [scene_entries[sid] for sid in order]
    return len(lines), len(cleaned_lines), '\n'.join(cleaned_lines) + '\n'


def clean_debug_bbox(file_path, keep='last'):
    """
    Deduplicates debug_bbox.txt in a single pass.
    
    Blocks start with '=' headers and 'SCENE: <scene_id>'.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Regex matching individual SCENE blocks
    pattern = r'(\s*=+\s*[\r\n]+\s*SCENE:\s*([A-Za-z0-9_-]+)\s*\|.*?(?=\s*=+\s*[\r\n]+\s*SCENE:|\Z))'
    blocks = re.findall(pattern, text, re.DOTALL)

    if not blocks:
        # Fallback: simple line scanning if block regex doesn't match custom format
        return 0, 0, text

    scene_blocks = {}
    order = []

    for full_block, scene_id in blocks:
        if scene_id not in scene_blocks:
            order.append(scene_id)
        if keep == 'last' or scene_id not in scene_blocks:
            scene_blocks[scene_id] = full_block.strip()

    cleaned_blocks = [scene_blocks[sid] for sid in order]
    cleaned_text = '\n\n'.join(cleaned_blocks) + '\n'
    return len(blocks), len(scene_blocks), cleaned_text


def clean_json_dataset(file_path, keep='last'):
    """
    Deduplicates parallel scene lists in physcene_collision_input*.json files.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'scene_ids' not in data:
        return 0, 0, data

    scene_ids = data['scene_ids']
    num_items = len(scene_ids)

    seen = {}
    order = []
    for idx, sid in enumerate(scene_ids):
        if sid not in seen:
            order.append(sid)
            seen[sid] = idx
        elif keep == 'last':
            seen[sid] = idx

    # If no duplicates, return
    if len(order) == num_items:
        return num_items, num_items, data

    indices = [seen[sid] for sid in order]

    cleaned_data = {}
    for key, val in data.items():
        if isinstance(val, list) and len(val) == num_items:
            cleaned_data[key] = [val[i] for i in indices]
        else:
            cleaned_data[key] = val

    return num_items, len(order), cleaned_data


def process_file(file_path, keep='last', inplace=False, backup=True, output_path=None, dry_run=False):
    """Processes a single file based on its filename / extension."""
    fname = os.path.basename(file_path)
    print(f"[*] Processing: {file_path}")

    if fname == 'guidance_losses.txt':
        orig_cnt, clean_cnt, content = clean_guidance_losses(file_path, keep=keep)
        unit = "entries"
    elif fname == 'debug_bbox.txt':
        orig_cnt, clean_cnt, content = clean_debug_bbox(file_path, keep=keep)
        unit = "SCENE blocks"
    elif file_path.endswith('.json'):
        orig_cnt, clean_cnt, content_json = clean_json_dataset(file_path, keep=keep)
        unit = "scenes in JSON"
        content = json.dumps(content_json, indent=2) + '\n' if isinstance(content_json, dict) else None
    else:
        print(f"    Skipping unsupported file type: {fname}")
        return

    removed = orig_cnt - clean_cnt
    print(f"    Original: {orig_cnt} {unit} -> Cleaned: {clean_cnt} {unit} (Removed {removed} duplicates)")

    if dry_run:
        print("    [DRY RUN] No files modified.")
        return

    if removed == 0 and not output_path:
        print("    No duplicates found. File unchanged.")
        return

    out_file = output_path if output_path else file_path

    if inplace and backup and out_file == file_path:
        bak_file = file_path + '.bak'
        shutil.copy2(file_path, bak_file)
        print(f"    Backup created: {bak_file}")

    if content is not None:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"    Saved cleaned output to: {out_file}")


def process_directory(dir_path, keep='last', inplace=False, backup=True, dry_run=False):
    """Scans and deduplicates all target log and JSON files in a directory."""
    print(f"=== Deduplicating Directory: {dir_path} ===")
    
    target_files = ['debug_bbox.txt', 'guidance_losses.txt']
    
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f in target_files or (f.startswith('physcene_collision') and f.endswith('.json')):
                fpath = os.path.join(root, f)
                process_file(fpath, keep=keep, inplace=inplace, backup=backup, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Single-Pass Log & Dataset Deduplication Tool for EchoScene"
    )
    parser.add_argument('--dir', type=str, help="Path to directory containing log/JSON files to deduplicate.")
    parser.add_argument('--file', type=str, help="Path to a single file to deduplicate.")
    parser.add_argument('--output', type=str, help="Path for output file (used with --file).")
    parser.add_argument('--keep', choices=['last', 'first'], default='last',
                        help="Which duplicate entry to retain: 'last' (default, final attempt) or 'first'.")
    parser.add_argument('--inplace', action='store_true', help="Overwrite original files in-place.")
    parser.add_argument('--no-backup', action='store_true', help="Do not create .bak backup files when modifying in-place.")
    parser.add_argument('--dry-run', action='store_true', help="Preview duplicate counts without modifying any files.")

    args = parser.parse_args()

    if not args.dir and not args.file:
        parser.print_help()
        sys.exit(1)

    backup = not args.no_backup

    if args.dir:
        process_directory(args.dir, keep=args.keep, inplace=args.inplace, backup=backup, dry_run=args.dry_run)
    elif args.file:
        process_file(args.file, keep=args.keep, inplace=args.inplace, backup=backup, output_path=args.output, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
