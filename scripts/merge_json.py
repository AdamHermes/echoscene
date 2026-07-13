"""
Merge multiple JSON files (each produced by a separate run covering a different slice of samples) into a single combined file.

Each input file has this shape (every key is a list indexed by scene):
{
  "class_labels": [scene0_objs, scene1_objs, ...],
  "translations": [...],
  "sizes":        [...],
  "angles":       [...],
  "objfeats_32":  [...],
  "objectness":   [...],
  "scene_ids":    [scan_id0, scan_id1, ...]
}

Merging = concatenating each key's list across files, in the same file order.

Usage:
    python merge_json.py final.json in1.json in2.json in3.json ...
"""
import json
import sys
from pathlib import Path

KEYS = ["class_labels", "translations", "sizes", "angles",
        "objfeats_32", "objectness", "scene_ids"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def merge_physcene_jsons(paths, dedupe_by_scene_id=True):
    merged = {k: [] for k in KEYS}
    seen_scene_ids = set()
    n_skipped_dupes = 0

    for path in paths:
        data = load_json(path)

        # sanity check: same keys, same number of scenes across all keys in this file
        n_scenes = len(data["scene_ids"])
        for k in KEYS:
            if k not in data:
                raise ValueError(f"{path} is missing key '{k}'")
            if len(data[k]) != n_scenes:
                raise ValueError(
                    f"{path}: key '{k}' has {len(data[k])} entries, "
                    f"but scene_ids has {n_scenes}. File is inconsistent."
                )

        print(f"Loaded {path}: {n_scenes} scenes")

        for i in range(n_scenes):
            scan_id = data["scene_ids"][i]

            if dedupe_by_scene_id and scan_id in seen_scene_ids:
                n_skipped_dupes += 1
                continue

            seen_scene_ids.add(scan_id)
            for k in KEYS:
                merged[k].append(data[k][i])

    print(f"\nMerged total: {len(merged['scene_ids'])} scenes")
    if n_skipped_dupes:
        print(f"Skipped {n_skipped_dupes} duplicate scene_id(s) "
              f"(same scene present in multiple input files)")

    return merged


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    for p in in_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    merged = merge_physcene_jsons(in_paths)

    with open(out_path, "w") as f:
        json.dump(merged, f)

    print(f"\nWrote merged file to: {out_path}")


if __name__ == "__main__":
    main()