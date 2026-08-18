# Full Benchmark Evaluation Workflow (Start to End)

This guide documents the complete, official workflow to evaluate a new work directory
(e.g. `current_works/work_XX`) from the raw generation zips to the final numbers in
`SceneGraph Experiement Log.xlsx`. It reflects the exact procedure used for
work_31 through work_34, baseline, work27, and work28 re-evaluations.

All commands are run from the `echoscene/` repo root unless stated otherwise.

```bash
cd /Users/lehoangan/Documents/GitHub/ROOM/echoscene
```

## Where the Works Live (SSD is the ACTIVE working folder)

- **ACTIVE working folder**: `/Volumes/ExternalSSD/current_works/` — all new works,
  extractions, eval artifacts, and result files are created here. Point every
  `--json` / `--scenes_dir` / `--out_dir` at this path. In the commands below,
  substitute `current_works/…` with `/Volumes/ExternalSSD/current_works/…`.
- **Legacy internal copy**: `echoscene/current_works/` (inside the repo) — kept for
  reference only; do **not** create new works there. A few historical dirs
  (`baseline_52`, `baseline_pp_52`, `current_best_SIGG`) exist only internally.
- The scripts themselves (`scripts/…`), `docs/`, and the `FRONT/` dataset still live
  in the repo — run scripts from the repo root, but read/write work data on the SSD.
- **Directory mapping** between the two locations is tracked in `docs/mapping.txt`.
  When a work moves or gets a duplicate, update its entry there.
- Rules: one work lives in exactly one place at a time (SSD authoritative if both
  exist); diff before deleting leftovers; never modify original zips on either volume.

## Overview of the Pipeline

### Two dataset scales — two Excel sheets

There are **two evaluation scales**, logged in **different sheets** of
`SceneGraph Experiement Log.xlsx`:

| Scale | Scenes | Excel sheet | Where the data comes from |
|---|---|---|---|
| **Full run** | 370 (all test rooms) | `ColObj Score` | `current_works/work_XX/…` zips |
| **Small test** | 52 (fixed subset) | `SmallTests` | dedicated 52-scene runs, or the **first 52 scenes of a full run** |

**The 52-scene small test** (`--smalltest` flag in `eval_3dfront.py`: start_idx=0,
max_samples=52 → 20 bedrooms + 12 livingrooms + 20 diningrooms per the canonical
`test_rooms_list` order) is a **fixed, always-identical subset** — the *same* 52
scenes every time. Two situations feed the `SmallTests` sheet:

1. **Dedicated 52-scene run** — generated with `--smalltest` (e.g.
   `current_works/baseline_52/`, `current_works/SIG_walkv3_52/`). The JSON directly
   contains those 52 scenes.
2. **Subset taken from a full 370 run** — some larger runs have their first 52
   scenes (in canonical `test_rooms_list` order) carved out and evaluated on the
   same 52 scenes, so they are comparable with the dedicated runs.

> **Comparability note:** because the full-run JSON is already sorted in canonical
> order, "first 52 scenes of a full run" = "the 52 smalltest scenes" — the subsets
> align scene-for-scene. Never compare a `SmallTests` row against a `ColObj Score`
> row as if they were the same benchmark: 52-scene numbers (especially ColScene,
> walkability, navigability) are not comparable to 370-scene numbers.

To evaluate a small test, run the exact same Steps 1–6 below but on the 52-scene JSON
(either the dedicated run's JSON, or slice the first 52 entries of a full-run JSON —
do **not** modify the original file, write the slice to a new file), and write the
results to the `SmallTests` sheet instead of `ColObj Score`. The `SmallTests` sheet
has the same column layout minus the Trial Number column (Trial Name is C1); mark
the scale in Method Note (e.g. `52 Rooms`).

### Variants

For every new work you produce **two variants** and evaluate both:

| Variant | Meaning | Source JSON |
|---|---|---|
| **RAW** | Raw model output (inference guidance only, no post-processing) | `physcene_collision_input.json` |
| **+ PP** | Post-processed via collision resolution | `physcene_collision_resolved.json` |

Metrics computed for each variant:

1. **ColObj / ColScene** — object & scene collision rates (`scripts/eval_collision.py`)
2. **Relational accuracy** — Total, Means of Means, L/R, F/B, Bi/Sm, Ta/Sh, Standing On, Close By, Symmetrical To (`scripts/relational/evaluate_relational_accuracy.py`)
3. **Walkability** — NavMesh walkable area ratio (`scripts/procthor_eval/eval_walkability.py`)
4. **Navigability** — object accessibility rate (`scripts/procthor_eval/eval_navigation.py`)

---

## Golden Rule: Never Touch the Original Data

**The zips and everything inside `current_works/work_XX/` as downloaded are
read-only source material.** Never re-zip, edit, delete, or "clean up" anything in
place. One accidental overwrite of a RAW directory or a modified zip means
re-downloading/re-generating data that cannot be reproduced.

Rules:

1. **Never extract into the same folder as the zip** and never let output land next to
   the originals. Always extract into a *newly created* folder
   (e.g. `current_works/work_XX/extracted/`).
2. **All derived artifacts go into a separate working folder**
   (e.g. `current_works/work_XX/2050/`) — merged JSONs, resolved JSONs, ProcTHOR
   conversions, result JSONs. The original `physcene_collision_input.json` stays
   untouched; scripts write to *new* files.
3. **Never overwrite an existing eval output.** Use fresh, clearly named output dirs
   (`procthor_scenes_RAW_fresh` vs old `procthor_scenes`) so stale and fresh results
   can be told apart. (This matters: old `procthor_scenes` dirs from a stale converter
   have silently different content — see Pitfalls.)
4. If a run needs modifying, **copy first, then modify the copy.**

### Original data structure (do not modify these)

A work directory as delivered looks like this:

```
current_works/work_XX/
├── vis_2050.zip                  # single-zip works: everything inside
│   (or vis_2050_0_124.zip + vis_2050_124_324.zip + vis_2050_324_end.zip  # multi-part)
└── extracted/                    # <- YOU create this (safe to touch)
    └── 2050/
        ├── physcene_collision_input.json   # RAW layout JSON (the key input)
        ├── debug_bbox.txt                  # human-readable bbox dump (all scenes)
        ├── guidance_losses.txt             # per-scene final guidance loss values
        ├── none_accuracy_analysis.txt      # quick relational accuracy summary
        ├── echoscene/                      # one *_echoscene.glb per scene
        │   ├── object_meshes/              # per-scene OBJ meshes
        │   └── *_echoscene.glb
        └── render_imgs/                    # rendered snapshots
```

Zip naming convention: `vis_2050_<start>_<end>.zip` covers scenes `[start, end)` of the
370-scene test set; `vis_2050.zip` (or `vis_2050_end.zip`) means the full run.
Everything under `extracted/` is fair game; everything above it is not.

---


### 0.1 Extract

Single zip (recent works):

```bash
unzip -o current_works/work_XX/vis_2050.zip -d current_works/work_XX/extracted
```

Multiple parts (e.g. `vis_2050_0_124.zip`, `vis_2050_124_324.zip`, `vis_2050_324_end.zip`):
extract each part into its own directory, then **merge the part JSONs into one
370-scene JSON** — see Step 0.4 below. This merged JSON is required by *every*
downstream metric (collision, relational accuracy, ProcTHOR conversion); the
relational eval (Total Accuracy / Means of Means) has no part-wise mode — it looks up
ground-truth triples by scene ID from a single JSON, so an unmerged run silently
covers only the scenes in whichever partial JSON you feed it.

### 0.2 Verify contents

```bash
# GLB count (expect 370)
ls current_works/work_XX/extracted/2050/echoscene/*.glb | wc -l

# JSON scene count (expect 370) and required keys
python3 -c "
import json
d = json.load(open('current_works/work_XX/extracted/2050/physcene_collision_input.json'))
print('scenes:', len(d['scene_ids']))
print('keys:', list(d.keys()))
"
```

Required JSON keys: `class_labels, translations, sizes, angles, objfeats_32,
objectness, scene_ids`.

### 0.3 Merge part JSONs (multi-part works only)

If the work was generated in multiple runs/zips, each part has its own partial
`physcene_collision_input.json`. Merge them with `scripts/merge/merge_json.py`
(merging = concatenating every key's list across files, in the given order):

```bash
mkdir -p current_works/work_XX/2050
python3 scripts/merge/merge_json.py \
    current_works/work_XX/2050/physcene_collision_input.json \
    current_works/work_XX/part1/2050/physcene_collision_input.json \
    current_works/work_XX/part2/2050/physcene_collision_input.json \
    current_works/work_XX/part3/2050/physcene_collision_input.json
```

**Verify before trusting the merge:**

```bash
python3 -c "
import json
merged = json.load(open('current_works/work_XX/2050/physcene_collision_input.json'))
ids = merged['scene_ids']
print('merged scenes:', len(ids))
assert len(ids) == len(set(ids)), 'DUPLICATE scene ids across parts!'
assert len(ids) == 370, 'expected 370 scenes'
print('no duplicates, count OK')
"
```

- Part counts must sum to 370 (e.g. work_33: 124 + 200 + 46 = 370; work_32: 183 + 187).
- Parts must **not overlap** (the assert above catches this).
- Also cross-check against the GLBs: every scene ID in the merged JSON should have a
  `*_echoscene.glb` in the combined `echoscene/` folder (copy GLBs from all parts into
  one folder first).

If a part zip is **missing its JSON** (happened with work_32 part 1), options are:
re-download/re-export the zip, or as a last resort reconstruct the missing scenes from
`debug_bbox.txt` (which covers all scenes) — but note the reconstruction lacks proper
`class_labels` indices needed by collision eval, so a real JSON is strongly preferred.
See `merge_json_instructions.md` and `merge_script.md` for the older chunk-folder
merging tooling.

### How merging relates to Total Accuracy / Means of Means

The parts are **not** evaluated separately and then combined — there is no such thing
as per-part accuracies being averaged. The flow is strictly:

1. **Merge first**: `merge_json.py` concatenates the lists, so scene 371 of the merged
   JSON is literally scene 0 of part 2's JSON. After merging there is zero difference
   between "generated in one run" and "generated in three runs".
2. **Evaluate once**: `evaluate_relational_accuracy.py` makes a single pass over the
   merged JSON. For each scene ID it fetches that scene's ground-truth relation
   triples from 3D-FRONT, checks them against that scene's predicted boxes, and
   appends 1/0 results to **shared global lists** (`accuracy['left']`,
   `accuracy['total']`, ...).
3. **Then**:
   - **Total Accuracy (micro)** = one mean over the pooled `total` list, containing
     checks from *all* scenes in *all* parts. Frequent relations (close by,
     symmetrical) dominate this number.
   - **Means of Means (macro)** = per-relation means computed over checks from *all*
     scenes, then averaged across the 7 categories (L/R, F/B, Bi/Sm, Ta/Sh,
     Standing On, Close By, Symmetrical To) — every relation counts equally.

Consequence: running the eval on the parts separately and averaging the two numbers
does **not** reproduce the merged result (the micro-average especially, since parts
contain different numbers of triple checks). Always merge, then run once.

### 0.4 Set up the working directory

Mirror the structure used by previous works:

```bash
mkdir -p current_works/work_XX/2050
cp current_works/work_XX/extracted/2050/physcene_collision_input.json current_works/work_XX/2050/
```

(Single-zip works: this is just a copy. Multi-part works: the file already created by
the merge in Step 0.3.)

---

## Step 1: Collision Resolution (produces the PP variant)

The solver reads the RAW JSON and writes the resolved JSON. Angles are radians in the
JSON; the script converts internally.

```bash
python3 scripts/collision/resolve_collision_json.py \
    --in_file  current_works/work_XX/2050/physcene_collision_input.json \
    --out_file current_works/work_XX/2050/physcene_collision_resolved.json
```

(Takes ~1–2 min for 370 scenes. For the full sorting / visualization / GLB
post-processing pipeline see `collision_resolution.md` — only the resolve step is
required for the metrics below.)

---

## Step 2: Collision Metrics (RAW and PP)

> **IMPORTANT:** `--max_rooms` defaults to `190`. Always pass `--max_rooms 0`
> to evaluate all 370 scenes, otherwise the numbers are not comparable with
> previously logged rows.

```bash
# RAW
python3 scripts/eval_collision.py \
    --json current_works/work_XX/2050/physcene_collision_input.json --max_rooms 0

# PP
python3 scripts/eval_collision.py \
    --json current_works/work_XX/2050/physcene_collision_resolved.json --max_rooms 0
```

Read from output:
- `Overall ColObj (Object Collision Rate)` → Excel **ColObj Score**
- `Overall ColScene (Scene Collision Rate)` → Excel **ColScene**

---

## Step 3: Relational Accuracy (RAW and PP)

```bash
# RAW
python3 scripts/relational/evaluate_relational_accuracy.py \
    --json current_works/work_XX/2050/physcene_collision_input.json

# PP
python3 scripts/relational/evaluate_relational_accuracy.py \
    --json current_works/work_XX/2050/physcene_collision_resolved.json
```

A `relational_accuracy_report.txt` is written next to each JSON. Read:
- `Micro-Average Accuracy (Total Acc)` → **Total Accuracy**
- `Macro-Average Accuracy (Means of Means)` → **Means of Means**
- `L/R`, `F/B`, `Bi/Sm`, `Ta/Sh`, `Standing On`, `Close By`, `Symmetrical To`

Requires the `FRONT/` dataset to be present at the repo root (it looks up ground-truth
triples by scene ID).

---

## Step 4: ProcTHOR Conversion (RAW and PP)

> **IMPORTANT:** Always regenerate the ProcTHOR scenes with the **current**
> converter. Older converted directories (e.g. from previous months) used a stale
> converter version and produce different object counts in navigation
> (e.g. 2126/2142 objects instead of ~1574–1780), making numbers incomparable.
>
> Also verify the RAW directory actually contains RAW content — in one case
> `baseline/procthor_scenes` had been silently overwritten with PP scenes.
> A quick `md5` of one scene file across RAW/PP dirs will reveal this.

```bash
# RAW
python3 scripts/procthor_eval/convert_echoscene_to_procthor.py --full \
    --bbox_path current_works/work_XX/2050/physcene_collision_input.json \
    --out_dir   current_works/work_XX/2050/procthor_scenes_RAW

# PP
python3 scripts/procthor_eval/convert_echoscene_to_procthor.py --full \
    --bbox_path current_works/work_XX/2050/physcene_collision_resolved.json \
    --out_dir   current_works/work_XX/2050/procthor_scenes_PP
```

`--full` converts all scenes (without it only the first 50 are converted).
Wait for conversion to finish before launching the Unity evals.

---

## Step 5: Walkability (RAW and PP)

Launches a local AI2-THOR Unity instance per run (~4–5 min per variant):

```bash
python3 scripts/procthor_eval/eval_walkability.py \
    --scenes_dir current_works/work_XX/2050/procthor_scenes_RAW

python3 scripts/procthor_eval/eval_walkability.py \
    --scenes_dir current_works/work_XX/2050/procthor_scenes_PP
```

Read `walkability_results.json` → `summary.average_walkability` → **Walkability**.

**If AI2-THOR times out** (`TimeoutError: Reading from AI2-THOR backend timed out`),
just rerun with `--resume` — it continues from the checkpoint:

```bash
python3 scripts/procthor_eval/eval_walkability.py \
    --scenes_dir current_works/work_XX/2050/procthor_scenes_RAW --resume
```

---

## Step 6: Navigation Accessibility (RAW and PP)

```bash
python3 scripts/procthor_eval/eval_navigation.py \
    --scenes_dir current_works/work_XX/2050/procthor_scenes_RAW

python3 scripts/procthor_eval/eval_navigation.py \
    --scenes_dir current_works/work_XX/2050/procthor_scenes_PP
```

Read `navigation_results.json` → `summary.average_accessibility_rate` →
**Navigability** (also log `accessible/evaluated` object counts for reference).

---

## Step 7: Log the Results in the Excel Sheet

File: `/Users/lehoangan/Documents/GitHub/ROOM/SceneGraph Experiement Log.xlsx`,
sheet **`ColObj Score`**. One work occupies **two rows**: `<name>` (RAW) directly
followed by `<name> + PP`.

| Column | Field | Source |
|---|---|---|
| C1 | Trial Number | work number |
| C2 | Trial Name | `workXX` / `workXX + PP` |
| C3 | Method | loss config (filled manually) |
| C4 | Method Note | guidance config (filled manually) |
| C5 | ColObj Score | Step 2 |
| C6 | ColScene | Step 2 |
| C7 | Log | `physcene_collision_input.json` / `physcene_collision_resolved.json` |
| C9 | Walkability | Step 5 |
| C10 | Navigability | Step 6 |
| C11 | Total Accuracy | Step 3 |
| C12 | Means of Means | Step 3 |
| C13–C19 | L/R, F/B, Bi/Sm, Ta/Sh, Standing On, Close By, Symmetrical To | Step 3 |

Conventions:
- All metric cells are **fractions in [0, 1]** rounded to 4 decimals (e.g. `0.2562`).
- Font: **Arial 11** for every written cell.
- RAW row uses the input JSON as `Log`; PP row uses the resolved JSON.

Python snippet used for writing:

```python
import openpyxl
from openpyxl.styles import Font

path = '/Users/lehoangan/Documents/GitHub/ROOM/SceneGraph Experiement Log.xlsx'
wb = openpyxl.load_workbook(path)
ws = wb['ColObj Score']
arial = Font(name='Arial', size=11)

rows = {
    R_RAW: {1: XX, 2: 'workXX',  5: 0.2562, 6: 0.4730, 7: 'physcene_collision_input.json',
            9: 0.2496, 10: 0.6237, 11: 0.9622, 12: 0.8803,
            13: 0.9870, 14: 0.9923, 15: 0.9628, 16: 0.9600,
            17: 0.9976, 18: 0.7150, 19: 0.5474},
    R_PP:  {1: XX, 2: 'workXX + PP', 5: 0.0158, 6: 0.0514, 7: 'physcene_collision_resolved.json',
            9: 0.2353, 10: 0.6192, 11: 0.9628, 12: 0.8689,
            13: 0.9896, 14: 0.9978, 15: 0.9628, 16: 0.9600,
            17: 0.9976, 18: 0.7002, 19: 0.4741},
}
for r, cells in rows.items():
    for c, v in cells.items():
        ws.cell(row=r, column=c, value=v).font = arial
wb.save(path)
```

---

## Parallelization Tips

Steps 2 and 3 (both variants) and Step 4 conversions are independent of each other and
can all run in background simultaneously right after Step 1:

```bash
nohup python3 scripts/eval_collision.py --json ..._input.json    --max_rooms 0 > /tmp/col_raw.log 2>&1 &
nohup python3 scripts/eval_collision.py --json ..._resolved.json --max_rooms 0 > /tmp/col_pp.log 2>&1 &
nohup python3 scripts/relational/evaluate_relational_accuracy.py --json ..._input.json    > /tmp/rel_raw.log 2>&1 &
nohup python3 scripts/relational/evaluate_relational_accuracy.py --json ..._resolved.json > /tmp/rel_pp.log 2>&1 &
nohup python3 scripts/procthor_eval/convert_echoscene_to_procthor.py --full --bbox_path ..._input.json    --out_dir .../procthor_scenes_RAW > /tmp/conv_raw.log 2>&1 &
nohup python3 scripts/procthor_eval/convert_echoscene_to_procthor.py --full --bbox_path ..._resolved.json --out_dir .../procthor_scenes_PP  > /tmp/conv_pp.log 2>&1 &
```

Then, once conversions finish (verify scene file counts = 370), launch the four Unity
evals (walkability + navigation, RAW + PP) in parallel. Running multiple AI2-THOR
instances concurrently on one machine is fine.

## Common Pitfalls Checklist

- [ ] Never modify/delete/re-zip original data — extract & work only in created folders
- [ ] New works go in `/Volumes/ExternalSSD/current_works/` (the active working
      folder); `echoscene/current_works/` is legacy — no new works there
- [ ] `eval_collision.py` without `--max_rooms 0` silently evaluates only 190 scenes
- [ ] Old `procthor_scenes` dirs from a stale converter → always regenerate fresh
- [ ] RAW dir containing PP content (overwrite accident) → `md5` one scene file to check
- [ ] Launching Unity evals before conversion completes → verify file count first
- [ ] AI2-THOR timeout mid-run → rerun with `--resume`
- [ ] Multi-part zips → confirm part JSONs sum to 370 scenes with no overlap
- [ ] Multi-part works → all metrics (incl. Total Accuracy / Means of Means) run on
      the **merged** JSON, never on a part JSON
- [ ] 52-scene small tests → results go to the `SmallTests` sheet, never mix with
      370-scene numbers from `ColObj Score`; the 52 scenes are always the same fixed
      subset (first 52 in canonical order)
- [ ] Excel cells written as fractions (0–1), Arial 11, RAW row above PP row
