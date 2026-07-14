# How to Merge Output JSON files

This guide explains how to merge multiple output JSON files (generated from different runs) into a single combined `final.json` file. 

## Merging the JSON files
We use the script `scripts/merge_json.py` located inside the `echoscene` directory to combine multiple JSON files. The script simply concatenates the lists from each input file into a single output file.

**Command syntax (from inside `echoscene` directory):**
```bash
python scripts/merge_json.py final.json <input_file_1> <input_file_2> ...
```

**Example:**
If you have generated two separate output files and want to merge them into a single `final.json` inside your local output folder, you just run:

```bash
python scripts/merge_json.py \
    ./output/released_full_model/vis/2050/final.json \
    ./output/released_full_model/vis/2050/part1.json \
    ./output/released_full_model/vis/2050/part2.json
```

This will produce a single combined `final.json` file containing all scenes from the provided input files! You can then directly run the ProcTHOR Walkability/Navigation scripts against it!
