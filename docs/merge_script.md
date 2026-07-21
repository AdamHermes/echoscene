# Merging Dataset Chunks (`scripts/merge/merge.py`)

## Overview
The `scripts/merge/merge.py` script is designed to consolidate multiple chunked output folders (e.g., `released_full_model_0_50`, `released_full_model_50_100`) located in the `to_be_merged/` directory into a single, cohesive dataset structure named `complete_released_full_model`.

## Key Features

1. **Deterministic Numerical Sorting**  
   It extracts the numerical range from the folder names (e.g., `50_100`) to sort and process the chunks in correct sequential order. This guarantees that the final merged data strictly aligns with the expected indices in the canonical room lists, bypassing alphabetical sorting errors where `50` would sort after `300`.

2. **JSON Data Merging**  
   Instead of just blindly copying files, the script loads and carefully concatenates `physcene_collision_input.json`. Array elements across `scene_ids`, `sizes`, `translations`, `angles`, `objfeats_32`, `objectness`, and `class_labels` are appended seamlessly.

3. **Redundancy Cleanup**  
   The script filters out redundant intermediate files that are no longer needed for the final unified dataset, such as `final.json` and `echoscene_collision_summary.json`, keeping the target directory clean.

4. **Automated Sanity Checks**  
   After completing the merge, the script converts `test_rooms_list.txt` to UTF-8 (if it isn't already) and aggressively cross-references the expected list of rooms against the actual `.glb` files found in the newly merged `vis/2050/echoscene` directory to ensure zero data loss.

## Usage

1. Ensure all your dataset chunk folders (`released_full_model_*`) and the `test_rooms_list.txt` reference file are placed inside the `to_be_merged/` directory.
2. From the root directory of the project, execute:
   ```bash
   python scripts/merge/merge.py
   ```
3. The script will print the processing order, merge the files, and output a missing-rooms report at the end. The final unified dataset will be ready in `to_be_merged/complete_released_full_model`.
