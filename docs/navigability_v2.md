# Navigability v2: global connectivity, path success, and a converter bug

*2026-10-18. Answers the WACV reviewer's point: the old metric "only checks whether a NavMesh point exists near each object, not global path connectivity or navigation success".*

## TL;DR

1. **The old ProcTHOR numbers are invalid for three separate reasons**:
   - (a) **Skipped scenes.** Scenes where `GetReachablePositions` failed were dropped. Only 293–346 of 370 scenes were scored, and work 37 lost the most (293/370).
   - (b) **Wrong spawn.** The spawn point was the first grid cell clear of the axis-aligned boxes, not the EDT maximum that `README_PROCTHOR.md` describes.
   - (c) **Converter bug.** `convert_echoscene_to_procthor.py` decoded the 15-class one-hot labels with the 14-word `vocab_14`:
     - class 14 (the real floor) was dropped, so the room became the furniture's bounding extent;
     - class 13 (**wardrobe**) was dropped as `_scene_`;
     - class 12 (**tv_stand**) was used as the "floor" whenever it appeared.

     So 61–86 scenes per model had a "room" under 1 m² (e.g. 1.62 × 0.07 m), and 48–90 scenes had no free floor at all.
2. Once all three are fixed, every model is 86–96% accessible. **work 37 (navigability guidance) is the only model with a clear, significant gain**: about +9 pts on AI2-THOR accessibility and +8 to +11 pts on every connectivity metric. The gain is larger at an agent radius of 0.3 m.
3. **work 41's navigability advantage does not survive.** On the corrected scenes it is tied with EchoScene+PP on AI2-THOR accessibility (Δ −0.2 [−2.0, +1.6]), on reachability from the spawn, and on largest-component ratio. It is slightly better on the PhyScene-style R_walkable/R_reach (+1.8 / +2.4 pts) and on object-to-object path success (+1.6 [+0.1, +3.2]). Its routes are longer (detour +0.09). The old "64.92 vs 62.19" came from skipped scenes. Gaussian-v1 is slightly better than EchoScene on most metrics (+1.5 to +3.5).
4. **Recommendation for the ICME paper:** report the corrected-converter numbers below. Present work 41 as roughly *on par* with EchoScene+PP on navigability, and work 37 as the navigability trade-off. Do not reuse 62.19/63.85/64.92/83.20 or the WACV 74.93/80.03.

## What changed in the code

- `scripts/procthor_eval/eval_navigation.py`
  - The spawn is now the EDT maximum of the free floor: exact rotated footprints, snapped to the 0.25 m grid.
  - Load failures, failed `GetReachablePositions` calls and empty results now score **0** (with a `status`) instead of being skipped.
  - The default output is `navigation_results_v2.json`. `--legacy` restores the old behaviour and output file, and reproduces 62.19 / 63.85 / 64.92 / 83.20 exactly.
  - The summary now has both the pooled rate and the per-scene mean.
- `scripts/procthor_eval/convert_echoscene_to_procthor.py`
  - 15-dim labels are now decoded with `class_map_13` plus {14: floor}, and the floor is the largest class-14 box. The other class-14 entry is the tiny scene node.
  - `--legacy_vocab` reproduces the old conversions byte for byte (checked with `diff -r` on work 41).
- `scripts/procthor_eval/eval_connectivity.py` (new). Works offline from the scene JSONs; 370 scenes take about 5 s. Geometry is the exact rotated footprint of each object whose bottom is below 1.0 m (the same rule as `furniture_boxes`). A 0.05 m cell is free if its exact distance to every footprint and wall is at least the agent radius. Metrics:
  - **Reach from spawn:** objects with a free cell within 0.75 m of their footprint, in the EDT-spawn component.
  - **Reach, THOR lattice:** the same rule, but the reachable set is an AI2-THOR-like 4-neighbour flood fill on the 0.25 m lattice with a swept-segment check.
  - **Obj-obj path success:** the fraction of object pairs whose 0.75 m approach regions share a component.
  - **Detour ratio:** shortest path (8-connected Dijkstra, i.e. A* with a zero heuristic, then line-of-sight string pulling) divided by the straight-line distance between approach points. Pairs closer than 0.5 m are excluded.
  - **Largest component / free area** and **# free components** (components of at least 0.0625 m²).
  - **PhyScene-compatible R_walkable / R_reach** (see below).
  - `--thor_check N` compares N random scenes against AI2-THOR `GetReachablePositions` and `GetShortestPathToPoint`.
- `scripts/procthor_eval/bootstrap_nav.py` (new): per-scene means, 95% percentile bootstrap CIs (10k resamples), and paired-difference CIs plus win/loss counts against a reference run.

### Agent radius calibration (why 0.2 m, not 0.3 m)

- **Wall clearance.** Across 40 scenes, AI2-THOR reachable points never come closer than **0.207 m** to a wall, so the effective radius is about 0.2 m. The "0.6 m agent width" in older docs is wrong.
- **Fine-raster vs THOR match.** With r = 0.2, the offline reachable region matches THOR's at mean IoU **0.80–0.82**. With r = 0.3 the IoU is 0.40.
- **Lattice vs THOR match.** On corrected scenes, the lattice flood fill reproduces THOR's reachable-point count exactly on 22/30 scenes and within 6 points on 29/30.
- **Sensitivity check.** r = 0.3 is reported below as a stricter setting.

### AI2-THOR artefacts found on the way

- **Leaky boxes.** On the old conversions, THOR "reachable" points sometimes lie *inside* large furniture boxes, e.g. 50 points inside a 2.4 × 2.3 m bookshelf. The converter's one-sided wall meshes let the capsule sweep through.
- **Unreliable path call on old scenes.** `GetShortestPathToPoint` to a THOR-reachable point succeeded only 41–48% of the time on old conversions; on corrected scenes it succeeds 90% of the time (mean detour 1.08).
- **Consequence.** The offline metrics are the cleaner basis for path or connectivity claims; AI2-THOR is kept as the familiar headline number.

## PhyScene metric definitions (paper + released code)

Sources: `PhyScene/scripts/eval/walkable_metric.py` and `PhyScene/utils/overlap.py::calc_wall_overlap`. Both use a 256-px top-down image with `scale = max|vertex| + 0.2`. The robot width is 0.3 m: the floor is eroded with a 0.3 m square kernel and each box is drawn with a 0.3 m contour, so the effective radius is about 0.15 m. Connected components use 8-connectivity.

- **R_walkable** = largest walkable component / total walkable area. It only uses boxes whose centre y is below 1.5 m.
  - **Code bug:** the released code `return`s inside the component loop, so it actually reports *the first-labelled component* (raster order), not the largest.
  - We report both the intended version (`phy_walkable`) and the code-literal one (`phy_walkable_code`).
- **R_reach** = Σ over components of (objects whose dilated box touches the component / all objects) × (component area / walkable area). This is an area-weighted reachability over *all* boxes, with no start point. It is **not** the same as our 0.75 m-from-spawn rule, so the two are reported separately.

Both are reimplemented in `eval_connectivity.py::physcene_metrics` on our rooms (full-extent sizes are halved to match PhyScene's half-size convention).

## Results: corrected converter (use these)

Controlled PP layouts (`ROOM/revision/pp_controlled/pp/*.json`) were converted with the fixed converter into `ROOM/revision/pp_controlled/procthor_fixed/<run>/`. All 370 scenes are scored for every model and no scene is dropped. Each cell shows the per-scene mean with its 95% bootstrap CI; Δ is the paired difference to EchoScene+PP, and * means the CI excludes 0. Raw JSON: `ROOM/revision/pp_controlled/connectivity_fixed/` and `procthor_fixed/<run>/navigation_results_v2.json`.

### AI2-THOR object accessibility (v2: EDT spawn, failures = 0; 0 failed scenes for every model)

Pooled (object-weighted) rates: EchoScene 85.7, Gaussian-v1 87.5, work 41 85.6, work 37 95.2.

| Model | AI2-THOR accessibility v2 (%) |
|---|---|
| EchoScene + PP | 86.6 [84.8, 88.4] |
| Gaussian-v1 + PP | 88.1 [86.4, 89.7]<br>Δ +1.5 [-0.3, +3.2] |
| work 41 + PP | 86.4 [84.6, 88.2]<br>Δ -0.2 [-2.0, +1.6] |
| work 37 + PP | 95.6 [94.3, 96.8]<br>Δ +9.0 [+7.0, +11.0] * |

### Offline connectivity and paths (agent radius 0.2 m, matched to AI2-THOR)

Scene-level counts:

| Model | Single connected free region | All objects reachable |
|---|---|---|
| EchoScene + PP | 215/370 | 251/370 |
| Gaussian-v1 + PP | 206/370 | 268/370 |
| work 41 + PP | 217/370 | 256/370 |
| work 37 + PP | 318/370 | 354/370 |

| Model | Reach from spawn (%) | Reach, THOR lattice (%) | Obj-obj path success (%) | Largest comp. / free (%) | # free components | Detour ratio ↓ |
|---|---|---|---|---|---|---|
| EchoScene + PP | 90.7 [89.1, 92.3] | 87.2 [85.3, 89.0] | 90.9 [89.2, 92.5] | 89.9 [88.3, 91.5] | 1.53 [1.46, 1.61] | 1.254 [1.231, 1.280] |
| Gaussian-v1 + PP | 93.1 [91.7, 94.3]<br>Δ +2.4 [+0.8, +3.9] * | 89.7 [88.1, 91.2]<br>Δ +2.5 [+0.8, +4.3] * | 93.3 [92.0, 94.7]<br>Δ +2.4 [+0.7, +4.2] * | 91.9 [90.4, 93.3]<br>Δ +2.0 [+0.4, +3.6] * | 1.60 [1.52, 1.68]<br>Δ +0.07 [-0.01, +0.16] | 1.271 [1.244, 1.300]<br>Δ +0.017 [-0.013, +0.048] |
| work 41 + PP | 91.1 [89.5, 92.7]<br>Δ +0.4 [-1.1, +2.0] | 86.9 [85.1, 88.7]<br>Δ -0.2 [-2.0, +1.5] | 92.5 [91.1, 93.8]<br>Δ +1.6 [+0.1, +3.2] * | 89.6 [87.8, 91.2]<br>Δ -0.4 [-1.9, +1.2] | 1.53 [1.46, 1.61]<br>Δ +0.00 [-0.07, +0.08] | 1.343 [1.310, 1.378]<br>Δ +0.087 [+0.053, +0.122] * |
| work 37 + PP | 98.9 [98.3, 99.5]<br>Δ +8.2 [+6.6, +9.8] * | 97.7 [96.8, 98.5]<br>Δ +10.6 [+8.7, +12.4] * | 99.3 [98.9, 99.7]<br>Δ +8.5 [+6.9, +10.1] * | 98.1 [97.3, 98.8]<br>Δ +8.2 [+6.5, +9.9] * | 1.15 [1.12, 1.20]<br>Δ -0.38 [-0.45, -0.30] * | 1.324 [1.293, 1.357]<br>Δ +0.069 [+0.030, +0.108] * |

### PhyScene-compatible metrics (robot width 0.3 m) and free space

| Model | R_walkable (%) | R_walkable, code-literal (%) | R_reach (%) | Free C-space / room (%) |
|---|---|---|---|---|
| EchoScene + PP | 91.2 [89.7, 92.8] | 77.9 [74.4, 81.3] | 81.6 [80.0, 83.3] | 28.0 [26.5, 29.4] |
| Gaussian-v1 + PP | 93.6 [92.3, 94.9]<br>Δ +2.4 [+0.9, +3.9] * | 80.8 [77.1, 84.3]<br>Δ +2.9 [-1.4, +7.2] | 85.1 [83.5, 86.8]<br>Δ +3.5 [+2.0, +5.0] * | 32.9 [31.6, 34.1]<br>Δ +4.9 [+4.3, +5.5] * |
| work 41 + PP | 93.1 [91.7, 94.4]<br>Δ +1.8 [+0.5, +3.2] * | 83.2 [80.0, 86.3]<br>Δ +5.4 [+1.6, +9.2] * | 84.1 [82.5, 85.6]<br>Δ +2.4 [+1.1, +3.8] * | 30.6 [29.3, 32.0]<br>Δ +2.7 [+2.1, +3.2] * |
| work 37 + PP | 98.8 [98.1, 99.4]<br>Δ +7.6 [+6.0, +9.2] * | 95.8 [93.9, 97.6]<br>Δ +18.0 [+14.2, +21.8] * | 95.4 [94.4, 96.4]<br>Δ +13.8 [+12.1, +15.5] * | 51.1 [50.1, 52.0]<br>Δ +23.1 [+21.9, +24.4] * |

### Sensitivity: agent radius 0.3 m

| Model | Reach from spawn (%) | Reach, THOR lattice (%) | Obj-obj path success (%) | Largest comp. / free (%) | # free components | Detour ratio ↓ |
|---|---|---|---|---|---|---|
| EchoScene + PP | 80.2 [77.9, 82.4] | 72.0 [69.5, 74.3] | 75.6 [72.7, 78.4] | 85.6 [83.5, 87.7] | 1.55 [1.47, 1.63] | 1.183 [1.161, 1.207] |
| Gaussian-v1 + PP | 83.8 [82.1, 85.5]<br>Δ +3.6 [+1.7, +5.6] * | 78.3 [76.4, 80.1]<br>Δ +6.3 [+4.3, +8.4] * | 79.2 [76.9, 81.5]<br>Δ +3.6 [+1.2, +6.0] * | 89.7 [88.0, 91.3]<br>Δ +4.0 [+2.0, +6.2] * | 1.64 [1.55, 1.73]<br>Δ +0.09 [+0.01, +0.18] * | 1.156 [1.136, 1.176]<br>Δ -0.029 [-0.054, -0.006] * |
| work 41 + PP | 79.9 [77.7, 82.1]<br>Δ -0.3 [-2.3, +1.7] | 71.6 [69.3, 73.9]<br>Δ -0.4 [-2.5, +1.8] | 77.8 [75.1, 80.3]<br>Δ +2.2 [-0.0, +4.4] | 83.6 [81.6, 85.7]<br>Δ -2.0 [-4.1, +0.2] | 1.70 [1.62, 1.78]<br>Δ +0.15 [+0.07, +0.23] * | 1.225 [1.197, 1.256]<br>Δ +0.036 [+0.010, +0.064] * |
| work 37 + PP | 96.1 [95.1, 97.1]<br>Δ +15.9 [+13.7, +18.3] * | 91.1 [89.5, 92.6]<br>Δ +19.1 [+16.6, +21.7] * | 96.7 [95.7, 97.6]<br>Δ +21.1 [+18.3, +24.0] * | 94.8 [93.5, 96.0]<br>Δ +9.2 [+6.9, +11.5] * | 1.32 [1.26, 1.38]<br>Δ -0.23 [-0.32, -0.14] * | 1.296 [1.262, 1.330]<br>Δ +0.111 [+0.075, +0.148] * |

## Results: old (buggy) converter, for reference only

These are the same PP layouts converted with `--legacy_vocab`, i.e. what the Aug-18 numbers used. The controlled-PP and Aug-18 PP folders give identical results. **Do not report these.** They show how much of the old picture came from the evaluator.

| Model | Legacy eval (as reported; scenes scored) | v2 eval, pooled | v2 eval, per-scene mean | v2 failed scenes (scored 0) |
|---|---|---|---|---|
| EchoScene + PP | 62.19 (346/370) | 50.0 | 56.5 | 57 |
| Gaussian-v1 + PP | 63.85 (342/370) | 56.5 | 62.2 | 57 |
| work 41 + PP | 64.92 (339/370) | 50.0 | 56.4 | 69 |
| work 37 + PP | 83.20 (293/370) | 60.3 | 65.6 | 88 |

| Model | AI2-THOR accessibility v2 (%) |
|---|---|
| EchoScene + PP | 56.5 [52.9, 60.0] |
| Gaussian-v1 + PP | 62.2 [58.4, 65.9]<br>Δ +5.7 [+2.9, +8.5] * |
| work 41 + PP | 56.4 [52.7, 60.1]<br>Δ -0.1 [-2.7, +2.6] |
| work 37 + PP | 65.6 [61.5, 69.7]<br>Δ +9.1 [+6.1, +12.2] * |

| Model | Reach from spawn (%) | Reach, THOR lattice (%) | Obj-obj path success (%) | Largest comp. / free (%) | # free components | Detour ratio ↓ |
|---|---|---|---|---|---|---|
| EchoScene + PP | 58.1 [54.4, 61.7] | 55.9 [52.2, 59.4] | 57.4 [53.3, 61.5] | 72.8 [69.3, 76.2] | 1.35 [1.26, 1.45] | 1.106 [1.086, 1.128] |
| Gaussian-v1 + PP | 65.1 [61.2, 68.9]<br>Δ +7.0 [+4.7, +9.4] * | 62.6 [58.8, 66.3]<br>Δ +6.7 [+4.5, +9.1] * | 65.0 [60.9, 69.0]<br>Δ +7.5 [+4.9, +10.2] * | 78.2 [74.7, 81.5]<br>Δ +5.4 [+2.0, +8.7] * | 1.28 [1.20, 1.37]<br>Δ -0.07 [-0.15, +0.01] | 1.104 [1.086, 1.122]<br>Δ -0.007 [-0.028, +0.013] |
| work 41 + PP | 56.2 [52.4, 60.1]<br>Δ -1.9 [-4.4, +0.6] | 52.8 [49.0, 56.5]<br>Δ -3.1 [-5.6, -0.7] * | 55.0 [50.8, 59.1]<br>Δ -2.4 [-5.1, +0.0] | 68.6 [64.6, 72.3]<br>Δ -4.3 [-8.1, -0.5] * | 1.16 [1.06, 1.26]<br>Δ -0.19 [-0.27, -0.12] * | 1.126 [1.103, 1.151]<br>Δ +0.005 [-0.018, +0.028] |
| work 37 + PP | 68.4 [64.0, 72.7]<br>Δ +10.3 [+7.4, +13.1] * | 66.2 [61.9, 70.3]<br>Δ +10.3 [+7.4, +13.1] * | 69.5 [65.0, 73.8]<br>Δ +12.0 [+9.2, +14.9] * | 70.1 [65.8, 74.3]<br>Δ -2.7 [-7.1, +1.6] | 0.96 [0.89, 1.04]<br>Δ -0.39 [-0.46, -0.31] * | 1.147 [1.123, 1.174]<br>Δ +0.027 [-0.005, +0.060] |


On the old conversions, 48–90 scenes per model have no free floor at radius 0.2 m. That is a symptom of the converter bug (rooms shrunk to the furniture hull), not of the layouts.

## Does work 41's navigability advantage survive?

**No.** Corrected scenes, paired against EchoScene+PP:

| Metric | work 41 Δ (95% CI) | Significant? |
|---|---|---|
| AI2-THOR accessibility | −0.2 [−2.0, +1.6] | no |
| Offline reach from spawn | +0.4 [−1.1, +2.0] | no |
| Reach, THOR lattice | −0.2 [−2.0, +1.5] | no |
| Largest-component ratio | −0.4 | no |
| Number of free components | identical | no |
| Object-to-object path success | +1.6 [+0.1, +3.2] | barely |
| PhyScene R_walkable | +1.8 | yes |
| PhyScene R_reach | +2.4 | yes |
| Free C-space | +2.7 | yes |
| Detour ratio | +0.09 (longer routes) | yes, worse |

- **At radius 0.3 m:** reach −0.3 and largest-component ratio −2.0 (both not significant), and work 41 has *more* free components (+0.15, significant).
- **Honest framing:** work 41 keeps EchoScene's navigability while improving collisions and relations. It does not improve navigability.
- **Where the guidance helps:** work 37's explicit navigability guidance gives the only large, robust gain: +9.0 [+7.0, +11.0] AI2-THOR accessibility; 318/370 single-region scenes vs 215; 354/370 scenes with every object reachable vs 251.

## Reproduce

```bash
cd echoscene
python scripts/procthor_eval/convert_echoscene_to_procthor.py --bbox_path ../revision/pp_controlled/pp/work41.json --out_dir ../revision/pp_controlled/procthor_fixed/work41 --full
python scripts/procthor_eval/eval_navigation.py --scenes_dir ../revision/pp_controlled/procthor_fixed/work41          # AI2-THOR v2
python scripts/procthor_eval/eval_connectivity.py --scenes_dir ../revision/pp_controlled/procthor_fixed/work41 --out ../revision/pp_controlled/connectivity_fixed/work41_r0.2.json
python scripts/procthor_eval/eval_connectivity.py --scenes_dir ../revision/pp_controlled/procthor_fixed/work41 --thor_check 30   # cross-check
python scripts/procthor_eval/bootstrap_nav.py --ref EchoScene+PP=../revision/pp_controlled/connectivity_fixed/baseline_r0.2.json --run work41+PP=../revision/pp_controlled/connectivity_fixed/work41_r0.2.json --metrics reach_rate pair_success
```

Old numbers: add `--legacy_vocab` to the converter and `--legacy` to `eval_navigation.py`.


## Round 2: function-aware accessibility metrics (2026-10-18)

The question was whether the 0.75 m rule is too crude, so that a stricter, more functional definition would show work 41 ahead. The metrics below were **fixed before any results were seen**, and all of them are reported. The spawn-independent `*_aw` variants were added after one look at example plots: those plots showed the EDT-spawn rules depend on which free region the spawn lands in. Script: `scripts/procthor_eval/eval_functional.py`; raw JSON: `ROOM/revision/pp_controlled/functional/`.

### Metric definitions

- **Functional access**
  - An object counts only if the agent (r = 0.2 m) can stand in its *usable zone*: a 0.6 m band along the side you use, connected to the spawn.
  - Which side is usable: the front for wardrobes, cabinets, shelves, nightstands, desks, sofas and TV stands; either long side for beds; any side for tables, chairs and lamps.
  - The *any start* variant averages over a uniformly random start point in the free floor, weighting each free region by its area. It is therefore independent of where the EDT spawn falls.
- **Usable-zone clear:** the share of the usable band that is inside the room and not covered by other furniture. **>50% blocked** is the share of objects whose band is mostly covered.
- **Reach via ≥0.8 m corridor:** reachable from the spawn with agent radius 0.4 m, i.e. through an accessible-route-width corridor. **Bottleneck width** is the widest-path corridor width to each object (0 if unreachable).

### Orientation fix

These metrics need each object's front, and that showed **another convention problem**. With the converter's yaw sign, object fronts have no consistent direction. With the yaw **negated**, the back of wall-hugging furniture faces the nearest wall for 629/695 objects. So the converter (and `scripts/eval_collision.py`, which uses the same rotation) mirrors the footprints of the 17% of objects that are not axis-aligned (542/3275 objects with more than 5° off-axis). `eval_functional.py` rebuilds footprints with the corrected sign (`--yaw_sign -1`). Fixing the converter and ColObj is a separate follow-up.

### PP layouts (corrected converter, corrected yaw)

| Model | Functional access | Functional access, any start | Reach, any start | Usable-zone clear | Usable zone >50% blocked ↓ | Reach via ≥0.8 m corridor | Bottleneck width (m) |
|---|---|---|---|---|---|---|---|
| EchoScene | 82.9 | 82.1 | 89.5 | 87.1 | 8.5 | 61.2 | 0.81 |
| Gaussian-v1 | 85.1<br>Δ +2.2 [-0.2, +4.6] | 83.3<br>Δ +1.2 [-0.9, +3.4] | 91.5<br>Δ +2.0 [+0.6, +3.5] * | 89.8<br>Δ +2.8 [+1.7, +3.9] * | 5.9<br>Δ -2.6 [-4.2, -1.0] * | 74.3<br>Δ +13.1 [+10.1, +16.2] * | 0.92<br>Δ +0.10 [+0.09, +0.12] * |
| work 41 | 82.8<br>Δ -0.1 [-2.4, +2.3] | 81.1<br>Δ -1.0 [-3.1, +1.0] | 90.0<br>Δ +0.5 [-0.8, +1.8] | 87.4<br>Δ +0.3 [-0.8, +1.5] | 7.6<br>Δ -0.9 [-2.5, +0.7] | 62.9<br>Δ +1.7 [-0.9, +4.4] | 0.83<br>Δ +0.01 [-0.00, +0.03] |
| work 37 | 96.1<br>Δ +13.2 [+10.8, +15.5] * | 95.6<br>Δ +13.5 [+11.3, +15.8] * | 99.1<br>Δ +9.6 [+8.0, +11.1] * | 96.6<br>Δ +9.5 [+8.2, +10.9] * | 1.1<br>Δ -7.4 [-9.2, -5.7] * | 89.7<br>Δ +28.4 [+25.1, +31.9] * | 1.08<br>Δ +0.26 [+0.24, +0.29] * |

### RAW layouts (before collision post-processing)

| Model | Functional access | Functional access, any start | Reach, any start | Usable-zone clear | Usable zone >50% blocked ↓ | Reach via ≥0.8 m corridor | Bottleneck width (m) |
|---|---|---|---|---|---|---|---|
| EchoScene | 86.8 | 85.9 | 93.2 | 84.7 | 9.3 | 65.2 | 0.85 |
| Gaussian-v1 | 86.5<br>Δ -0.4 [-2.6, +1.8] | 84.8<br>Δ -1.1 [-3.2, +1.0] | 92.8<br>Δ -0.4 [-1.7, +0.9] | 88.5<br>Δ +3.7 [+2.6, +4.9] * | 6.2<br>Δ -3.1 [-4.8, -1.4] * | 77.4<br>Δ +12.2 [+9.3, +15.4] * | 0.94<br>Δ +0.09 [+0.08, +0.11] * |
| work 41 | 84.0<br>Δ -2.8 [-5.0, -0.6] * | 82.7<br>Δ -3.2 [-5.2, -1.1] * | 92.0<br>Δ -1.1 [-2.3, +0.1] | 86.3<br>Δ +1.6 [+0.5, +2.7] * | 8.0<br>Δ -1.2 [-2.9, +0.5] | 66.0<br>Δ +0.8 [-1.8, +3.5] | 0.85<br>Δ +0.00 [-0.01, +0.01] |
| work 37 | 96.1<br>Δ +9.3 [+7.1, +11.5] * | 95.8<br>Δ +9.9 [+7.7, +12.2] * | 99.2<br>Δ +6.1 [+4.7, +7.4] * | 95.3<br>Δ +10.6 [+9.2, +12.0] * | 1.4<br>Δ -7.9 [-9.6, -6.2] * | 90.4<br>Δ +25.2 [+21.9, +28.7] * | 1.09<br>Δ +0.24 [+0.21, +0.26] * |

### Outcome

- **work 41:** stricter, function-aware definitions do not change the conclusion. On PP every metric is a tie with EchoScene (all CIs contain 0). On RAW it is slightly *worse* on functional access (−2.8 [−5.0, −0.6]) and slightly better on usable-zone clearance (+1.6 [+0.5, +2.7]).
- **work 37:** the gains get *larger* under the stricter metrics: functional access +13 pts, corridor reach +28 pts, bottleneck width +0.26 m.
- **Gaussian-v1:** clearly better on corridor width (+0.10 m bottleneck, +13 pts ≥0.8 m reach) and usable-zone clearance.
- **Takeaway:** any visible advantage of work 41 is not one these geometric navigability or accessibility measures pick up. It is more likely in relational plausibility (its relation accuracy is the best of all runs), which is the claim the paper should make for work 41.

`docs/navigability_v2_examples.png`: three scenes, EchoScene+PP (left) vs work 41+PP (right). Each panel shows the free C-space connected to the spawn (green; dark green = ≥0.8 m corridor), C-space not connected to the spawn (pink), the usable zones (orange) and front arrows (blue). The top row is the scene where work 41 loses most, shown to illustrate the spawn-region sensitivity that motivated the `*_aw` variants.
