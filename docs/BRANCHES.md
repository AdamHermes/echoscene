# Branches, tags and cleanup log

Last cleanup: **2026-10-16**. No commit was deleted. Every removed or renamed branch tip is preserved as an `archive/*` tag on origin.

## Live branches

| Branch | Status | What it is |
|---|---|---|
| `physcene_fusion` | **Trunk** | The full SteerScene method (collision, room-boundary, Walkability-v3, Relation-v2 guidance, SAT Collision Resolver, and all evaluators). Tag `paper-wacv27-submission` marks its state at submission. |
| `integrate/fixes-and-3dssg` | Review, then merge into trunk | Trunk plus the robustness fixes from `real_current_best` and the 3DSSG adapter from `test3DSSG`. See below. |
| `test3DSSG` | Keep until the integration branch is merged | Holds the roughly 1,370 generated 3DSSG scene JSONs. The code is already ported. |
| `main` | Stale (teammate's line, last activity Jul 22) | Left untouched because it is the GitHub default branch. Plan: point `main` at the trunk after agreeing with the teammate. Old tip is in `archive/old-main`. |
| `bad_faster_physcene_fusion` | Bad | 18 performance commits (AMP autocast, CUDA-graph replay, DDIM layout sampler, batched inference). Considered erroneous, and its outputs are not comparable to the paper. Do not merge. |
| `bad_physics` | Bad (abandoned) | LoRA fine-tuning, an optimization-based collision fixer and A*/RRT navigation smoke tests. The paper method is training-free, so none of this is used. |
| `bad_test` | Bad (superseded) | Pairwise OBB-intersection collision loss. The trunk covers this with `collision.metric: obb`. |
| `bad_testing` | Bad (superseded) | Convex-hull floor polygon in `structured_scene_export.py` plus run JSONs. |
| `bad_test_ablation` | Bad | Early post-processing scripts and draft `.tex` files, with a 32 MB PDF committed. Do not merge. |
| `upstream/*` | Upstream | The original EchoScene repository (ymxlzgy). Not ours. |

## Deleted branches (all were config variants or already merged)

| Former branch | Tag | Experiment it held (Supp. = WACV'27 #3201 supplementary) |
|---|---|---|
| `current_best` | `archive/current_best` | Code era of the **selected Gaussian-v1 model** (`current_best_SIGG`, trial 3): interval 1, start 0.9, `10·L_col + 10·L_room + L_walk-gausv1`. Merged into the trunk on Aug 10. |
| `real_current_best` | `archive/real_current_best` | Same era. Its fixes are ported to `integrate/fixes-and-3dssg`. Its early OBB resolver scripts are superseded by `scripts/collision/`. |
| `real_current_best95` | `archive/real_current_best95` | 52-room screening "Start ratio 0.95" (SmallTests "SIG Best 0.95"). |
| `real_current_best5` | `archive/real_current_best5` | Screening "Interval-5 guidance" (SmallTests "SIG Best 5"). |
| `real_current_bestobb` | `archive/real_current_bestobb` | Screening "OBB guidance" (SmallTests "SIG Best obb", only 32 of 52 scenes exist). |
| `SIG_walkv3` | `archive/SIG_walkv3` | Work 29 `SIGG_walkv3` = Supp. "Without relational guidance": interval 1, start 0.9, Walkability-v3 1:1. |
| `relational` | `archive/relational` | Development of Relation-v2 (`compute_relational_guidance_loss`). |
| `fast_relational` | `archive/fast_relational` | Relational guidance with its own start_ratio and interval (the work 31/33 config). |

Stashes are snapshotted as `archive/stash-0` … `archive/stash-4`; the local stashes themselves were not dropped.

To recover any of these: `git checkout -b <name> archive/<name>`.

## `integrate/fixes-and-3dssg` contents

1. `53f40aad`: `cudnn.benchmark = False` in both `diff_utils/util.py` copies, for deterministic runs. VQVAE `decode_no_quant` decodes in chunks of 4 to avoid OOM. Output is unchanged.
2. `7a80a7af`: `pytorch3d_to_trimesh(mesh, batch_idx)` indexes the batch directly and returns an empty `Trimesh` for empty meshes instead of crashing. Output is the same as before for non-empty meshes.
3. `a0e48663`: `test3DSSG/adapter.py`, `scripts/eval_3dssg.py`, `test3DSSG/README.md` and one sample scene (paper Table 4).

Once this is merged into the trunk, delete `test3DSSG` (the tag keeps the data).

## Where each experiment config lives

Configs differ only in `config/full_mp.yaml` → `layout_branch.inference_guidance`.

| Paper row | Where to get the config |
|---|---|
| Selected Gaussian-v1 (Tables 1–2, Supp. I) | `archive/current_best` |
| Balanced / Light Gaussian (work 27, Supp. VII) | current trunk HEAD config (interval 5, start 0.95, center Gaussian only) |
| Without relational guidance (work 29) | `archive/SIG_walkv3` |
| Relational weight 0.5 (work 33, also used in **Table 3**) | `archive/fast_relational` |
| High-weight Relation-v2 (work 37) | trunk commit `40c86a12` |
| Center-centric (work 42) | trunk commit `8699cae5` |
| Work 38–41 ablations | trunk history Aug 19–20; see `docs/mapping.txt` and `docs/results_tables.md` |

## Known number-provenance issues (for the camera-ready)

* **Main-paper Table 5** (74.93 → 80.03) comes from the Jul 22 run in `benchmark_evaluation_report.md`, which covered only about 313 of 370 scenes and had a duplicated `Bedroom-13858`. The full 370-scene rerun (`docs/results_tables.md`, Aug 18) gives **62.19 → 63.85**, which matches Supp. Tables IV and VI. The Supp. text in §B.C still quotes the stale numbers.
* **Main-paper Table 3 and Supp. Table III**: the SteerScene rows are **work 33**, not the selected Gaussian-v1 model, and the paper prints the label "(Ours / Work 33)".
* The Room-Center Gaussian in the selected model's code computes `exp(-d²/σ)` over all objects. Paper Eq. 9 uses σ² and only objects with o_i = 1.
