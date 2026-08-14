# Log & Dataset Deduplication Script (`scripts/cleaning/remove_duplicates.py`)

## Overview

The `scripts/cleaning/remove_duplicates.py` tool performs a single-pass scan over evaluation logs (`debug_bbox.txt`, `guidance_losses.txt`) and JSON dataset files (`physcene_collision_input*.json`) to remove duplicate scene entries while preserving original scene ordering.

Duplicate entries commonly occur when warm-up steps, re-evaluations, or retries write repeated blocks for the same `scene_id` during batch generation or evaluation runs.

---

## Features

1. **Single-Pass Deduplication**:
   - **`debug_bbox.txt`**: Parses and groups bounding box text blocks starting with `SCENE: <scene_id>`. Filters out duplicate blocks.
   - **`guidance_losses.txt`**: Scans lines formatted as `Scene <scene_id> - ...` and retains canonical entries.
   - **JSON Datasets**: Deduplicates parallel arrays (`scene_ids`, `translations`, `sizes`, `angles`, `objfeats_32`, `objectness`, `class_labels`) while maintaining element alignment.

2. **Retention Policy (`--keep`)**:
   - `last` *(default)*: Retains the final attempt/block logged for each scene ID (ideal for keeping the finalized output after retries).
   - `first`: Retains the initial occurrence of each scene ID.

3. **In-Place Modification with Backup**:
   - Supports `--inplace` editing with automatic creation of `.bak` backup files.
   - Dry-run mode (`--dry-run`) previews duplicate counts without altering any files on disk.

---

## Usage

### 1. Dry-Run Check on a Run Directory
Preview duplicate counts without modifying files:
```bash
python scripts/cleaning/remove_duplicates.py --dir current_works/work_num27_attempt3/2050 --dry-run
```

### 2. Deduplicate a Directory In-Place
Clean `debug_bbox.txt`, `guidance_losses.txt`, and JSON files in a directory while creating backups (`.bak` files):
```bash
python scripts/cleaning/remove_duplicates.py --dir current_works/work_num27_attempt3/2050 --inplace
```

### 3. Clean a Single Log File
Deduplicate a specific file and save the result to a new file:
```bash
python scripts/cleaning/remove_duplicates.py --file current_works/work_num27_attempt3/2050/debug_bbox.txt --output debug_bbox_clean.txt
```

---

## CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dir <path>` | Directory path containing log/JSON files to deduplicate. | None |
| `--file <path>` | Path to a single log or JSON file to deduplicate. | None |
| `--output <path>` | Output path for a single file (used with `--file`). | Overwrites / stdout |
| `--keep {last,first}` | Retain the `last` (final attempt) or `first` occurrence per scene ID. | `last` |
| `--inplace` | Modify target files in-place. | `False` |
| `--no-backup` | Skip creation of `.bak` backup files when modifying in-place. | `False` |
| `--dry-run` | Display duplicate statistics without making changes. | `False` |
