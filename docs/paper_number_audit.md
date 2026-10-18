# Audit: every number in the WACV 2027 #3201 submission vs. its source

Audited 2026-10-18 against the logs in this repo, the SSD work folders (`/Volumes/ExternalSSD/current_works`), and fresh re-runs (`scripts/eval/`, results in `docs/results_2026-10-18/`).

Status key: ✅ reproduced from source · 🔎 traced to a log but not re-run · ⚠️ wrong model or inconsistent · ❌ wrong number · ❓ unreproducible

## Main paper

| Item | Paper value | Source | Status | Correct / recomputed value |
|---|---|---|---|---|
| **Table 1** EchoScene (raw) FID / FID-CLIP / KID | Bed 52.99 / 4.73 / 3.47, Liv 84.96 / 7.73 / 9.72, Din 72.15 / 8.08 / 10.93 | `baseline_result.txt` (Aug 13 log) | ❓ No surviving render folder reproduces it: `baseline_without_lamp` gives 51.66, `gt_renders` as reference gives 61.20, the with-lamp renders give 69.79 | Bed **51.66 / 6.15 / 8.56**, Liv **78.63 / 10.02 / 13.61**, Din **62.50 / 8.71 / 9.18** (original protocol, `baseline_without_lamp` renders) |
| Table 1 EchoScene + Resolver | Bed 50.69 / 6.15 / 8.46, Liv 79.45 / 10.51 / 14.52, Din 65.99 / 9.45 / 10.91 | `baseline_post_processed_result.txt`, `baseline_post_processed_without_lamp/` | ✅ 50.69 / 6.18 / 8.46, 79.45 / 10.56 / 14.56, 65.99 / 9.48 / 10.91. FID-CLIP and KID differ only in the 2nd decimal (KID subset sampling) | — |
| Table 1 SteerScene (Gaussian-v1, raw) | Bed 58.50 / 7.29 / 14.84, Liv 80.92 / 9.03 / 15.01, Din 68.06 / 9.11 / 12.38 | `guidance_result.txt`, `to_be_merged/complete_released_full_model_without_lamp` | ✅ 58.50 / 7.32 / 14.84, 80.92 / 9.06 / 15.01, 68.06 / 9.14 / 12.38 | — |
| Table 1 SteerScene + Resolver | Bed 57.28 / 7.21 / 14.06, Liv 82.24 / 9.65 / 16.40, Din 70.31 / 9.56 / 14.25 | `guidance_postprocess.txt`, `…_post_processed_without_lamp` | ✅ 57.27 / 7.25 / 14.06, 82.24 / 9.69 / 16.40, 70.31 / 9.60 / 14.25 | — |
| Table 1 Graph-to-3D … CommonScenes rows | — | Copied from the EchoScene / CommonScenes papers | 🔎 Not re-run | — |
| **Table 2** collisions (mesh-level), all 4 rows | e.g. EchoScene Bed 0.4084 / 0.7284 | `baseline_result.txt`, `baseline_post_processed_result.txt`, `guidance_result.txt`, `guidance_postprocess.txt`: lines `[bedroom] overlap object … cnt 341/835` (PhyScene mesh evaluator) | 🔎 Traced exactly; not re-run (needs the PhyScene mesh-collision evaluator) | — |
| Table 2 OOB column (1.2135 → 0.0405) | 1.2135 / 0.2541 / 0.7135 / 0.0405 | `docs/results_tables.md` (2D top-down proxy, `scripts/eval_ob.py`) | ⚠️ Mixes the 2D proxy with the mesh-level ColObj columns in the same table, without saying so | Re-run with one resolver (`scripts/eval/rerun_pp.py`): EchoScene + PP 0.254, Gaussian-v1 + PP **0.051** (paper's 0.0405 came from an older resolver run) |
| **Table 3** scene-graph consistency, SteerScene rows | 96.42 / 0.9904 / … / 0.5431 and +Resolver 96.46 / … / 0.5517 | `results_tables.md` rows **work33** / **work33 + PP** | ⚠️ **Wrong model.** These are work 33 (Relation-v2 weight 0.5), not the Gaussian-v1 model used in Tables 1, 2 and 5. The label "(Ours / Work 33)" is printed in the paper | Gaussian-v1: Total 95.60 → 95.61 (+PP), Close By 68.06 → 66.67, Symmetry 52.16 → 50.00 |
| Table 3 EchoScene rows | 96.36 / … and +PP 96.33 / … | `results_tables.md` baseline rows | ✅ Re-run gives 96.36 and 96.33 | — |
| **Table 4** 3DSSG collisions | 0.2621 / 0.2750 … | `test3DSSG` branch outputs (archived as tag `archive/test3DSSG`) | 🔎 Not located in the repo logs | — |
| **Table 5** navigability | EchoScene + PP **74.93**, SteerScene **80.03** | `benchmark_evaluation_report.md`, `docs/procthor_eval_results.md` (Jul 22 run) | ❌ **Stale run:** only 313 / 314 of 369 / 370 scenes evaluated (old converter), baseline missing `LivingRoom-4789`, SteerScene JSON had `Bedroom-13858` twice | **62.19 → 63.85** (Aug 18 full-370 rerun, `results_tables.md`; same as Supp. Tables IV and VI) |
| Abstract: "reduces … collision rates by over 99%" | — | Table 2 | ⚠️ Holds only for SteerScene **+ Resolver**. SteerScene alone reduces bedroom ColObj by about 56% (0.4084 → 0.1810) | Reword |
| Eq. 9 room-center Gaussian | exp(−d²/σ²), objects with o_i = 1 | `compute_walkable_loss` (current_best era) | ⚠️ Code uses exp(−d²/σ) over **all** nodes | Fix the equation or the code |

## Supplementary

| Item | Paper value | Status | Correct value |
|---|---|---|---|
| §B.C text: "improves Navigability from 74.93% to 80.03%" | 74.93 → 80.03 | ❌ Same stale run as main Table 5; contradicts the supplementary's own Tables IV and VI | 62.19 → 63.85 |
| Table I | = main Table 2 | 🔎 Same as above | — |
| Table II | = main Table 1 | Same as above (raw EchoScene row ❓) | See main Table 1 |
| Table III SteerScene rows | work 33 | ⚠️ Same as main Table 3 | — |
| Tables IV and VI (370-scene 2D proxy ablation) | e.g. EchoScene 0.2550 / 0.4676, +PP 0.0179 / 0.0541 | ✅ Re-run reproduces baseline, work 37 (0.0025 / 0.0081 / 0 OOB) and others. Gaussian-v1 + PP differs slightly (0.0179 → 0.0163) because its PP file came from an older resolver | `docs/results_2026-10-18/` |
| **Table VII** "Relation-v2 weight 0.5" (work 33) FID / KID | All-370 145.84, Bed **272.10**, Library **257.37** | ❌ **Rendering bug:** `fid_eval_work33.py` loaded meshes from `work_33/part2` only (200 of 370 scenes), so **170 renders are blank** (both RAW and PP folders) | Recomputed with all 3 mesh parts (0 blank): **All-370 41.16 / 6.80, Bed 51.32 / 9.06, Library 89.34 / 5.31** (raw); +PP All-370 42.07 / 6.91, Bed 51.87 / 9.83, Library 89.09 / 4.50. Living (79.52) and Dining (61.51) were unaffected, since all those scenes were in part2 |
| Table VII Light Gaussian (work 27), High-weight Relation-v2 (work 37) | e.g. 41.76 / 0.0073, 79.26 / 0.0420 | ✅ Render folders have 0 blank images, so no rendering bug | — |
| Table VII EchoScene / Selected rows, Library and All-370 shown as "–" | – | Now computed | EchoScene raw: Library 87.03 / 3.08, All-370 40.73 / 6.07. Gaussian-v1 + PP: Library 97.18 / 8.78, All-370 45.66 / 9.66 |
| Table IX (52-room screening) | — | 🔎 From `results_tables.md` SmallTests, not re-run. `SIG Best obb` covers only 32 of 52 scenes (already noted there) | — |

## Root causes (to avoid next time)

1. **Mixed models across tables:** Gaussian-v1 (Tables 1, 2, 5), work 33 (Table 3). → Pick one model and generate every table from it.
2. **Stale navigability run** copied into the paper after the full rerun existed.
3. **Partial mesh folders** when a run is split into parts (`part1/2/3`). → `scripts/eval/fid_eval.py` users should check for blank renders. In this audit, blank renders were detected by image std < 1.
4. **Two collision pipelines** (mesh-level PhyScene evaluator vs 2D proxy) mixed in one table. → Label the pipeline in every table caption.
5. **PP files from different resolver versions.** → Re-run PP for all compared runs with `scripts/eval/rerun_pp.py`.
