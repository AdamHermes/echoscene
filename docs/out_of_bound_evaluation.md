# Out-of-Bound (OB) Evaluation

The `eval_ob.py` script calculates the number of out-of-bound (OB) furniture objects in generated 3D room scenes. 

An object is considered out-of-bound if its 2D footprint (bounding box) is not entirely contained within the room's floor boundary. This metric helps evaluate the physical realism of the generated scene layouts by checking if the generative model places items outside the physical constraints of the room.

## Usage

Run the evaluation script from the `echoscene` root directory:

```bash
python scripts/eval_ob.py --json <path_to_prediction_json>
```

### Arguments

- `--json`: The path to the JSON file containing the predicted room layouts (e.g., translations, sizes, angles, and class labels).
  - **Default**: `../physcene_collision_input_merged.json` (relative to the `scripts` directory).

### Example Command

```bash
python scripts/eval_ob.py --json current_works/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_resolved.json
```

## How It Works

1. **Loads Layout Data**: Parses the input JSON file containing scene translations, sizes, angles, and one-hot classes.
2. **Identifies Floor**: Dynamically identifies the room floor polygon based on flat bounding box geometry ($\text{height} < 0.1\text{m}$ and maximum footprint area).
3. **Identifies Furniture**: Iterates over all valid furniture items in the scene and creates 2D oriented bounding box (OBB) polygons based on their translations, sizes, and rotation angles.
4. **Containment Check**: Verifies whether each furniture polygon is fully contained within the floor polygon using a shapely polygon intersection test.
5. **Generates Statistics**: Computes and outputs:
   - Total number of scenes evaluated.
   - Total out-of-bound (OB) objects.
   - Total valid furniture objects.
   - **Object Out-of-Bound Rate (`tot_ob / tot_obj`)**: The primary metric, bounded in $[0, 1]$, matching `ColObj`.
   - **Average #OB count per scene (`tot_ob / processed_scenes`)**.
