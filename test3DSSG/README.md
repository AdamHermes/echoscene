# 3DSSG generalization test (paper Table 4)

Ported from branch `test3DSSG` (archived as tag `archive/test3DSSG`).

1. Put the 3DSSG annotations at `<repo>/3DSSG/objects.json` and `<repo>/3DSSG/relationships.json` (git-ignored).
2. Convert them to EchoScene scene graphs:
   ```bash
   python test3DSSG/adapter.py   # writes test3DSSG/adapted_scenes/adapted_<scan_id>.json
   ```
   `adapter.py` maps 3DSSG labels and predicates to the SG-FRONT vocabulary (`CLASS_MAPPING`, `REL_MAPPING`) and drops scenes that are empty after filtering. Room size is fixed to `[6.0, 3.0, 5.0]`.
3. Generate and evaluate:
   ```bash
   cd scripts
   python eval_3dssg.py --json_dir ../test3DSSG/adapted_scenes --start_idx 0 --max_samples 2 \
     --exp ../released_full_model --dataset ../FRONT --epoch 2050 --room_type all \
     --render_type echoscene --gen_shape True --save_3d True
   ```
   Use `--json_path <file>` to run a single scene, for example the included `adapted_0988ea72-eb32-2e61-8344-99e2283c2728.json`.

The roughly 1,370 generated `adapted_scenes/*.json` files are not committed here. Regenerate them with step 2, or take them from tag `archive/test3DSSG`.
