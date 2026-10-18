# FID and KID Evaluation Benchmark Results

This document compiles the **FID (Fréchet Inception Distance)** and **KID (Kernel Inception Distance)** evaluations across generated 3D scene datasets on the 3D-FRONT benchmark.

---

## 1. Evaluation Methodology & Protocol

### Metric Definitions
- **FID (Fréchet Inception Distance)**: Evaluates the distance between Gaussian distributions fitted to Inception-v3 feature embeddings of real vs synthetic top-down renderings. Lower is better ($\downarrow$).
- **KID (Kernel Inception Distance)**: Unbiased squared Maximum Mean Discrepancy (MMD) using a polynomial kernel ($k(x, y) = (\frac{1}{d} x^T y + 1)^3$). Computed with 100 subsets ($N=100$) per standard protocol. Lower is better ($\downarrow$).
- **Engine**: Implemented using [`cleanfid`](https://github.com/GaParmar/clean-fid) (`num_workers=0` for macOS multiprocessing stability, `device="mps"` or `"cpu"`).

### Reference Ground Truth Renders
- **Path**: `echoscene/dataset_gt_renders/sdf_renders/sdf_fov90_h8_wo_lamp_no_stool/small/test`
- **Render Specifications**: 256×256 top-down orthogonal/perspective orthant view at camera height $h = 8.0\,\text{m}$, FOV $90^\circ$, `without_lamp=True`, `without_stool=True`.
- **Dataset Size**: Canonical 370 test scenes defined in `FRONT/relationships_all_test.json`.

### Room Category Splits
| Room Split | Categories Included | Total Scenes |
| :--- | :--- | :---: |
| **`all_370`** | All 4 room categories (Bedroom, Living, Dining, Library) | 370 |
| **`all_314`** | Standard 3-room benchmark (Bedroom, Living, Dining) | 314 |
| **`bedroom`** | `["Bedroom", "MasterBedroom", "SecondBedroom"]` | 162 |
| **`livingroom`** | `["LivingDiningRoom", "LivingRoom"]` | 83 |
| **`diningroom`** | `["LivingDiningRoom", "DiningRoom"]` | 100 |
| **`library`** | `["Library"]` | 56 |

*(Note: `LivingDiningRoom` belongs to both `livingroom` and `diningroom` splits per 3D-FRONT standard)*.

> [!NOTE]
> **Why Baseline and SteerScene rows omit Library / All 370**: See detailed explanation in [library_and_all370_fid_kid_faq.md](library_and_all370_fid_kid_faq.md).

---

## 2. Cross-Model Benchmark Comparison (Standard: Without Lamps)

| Model / Run | Variant | `all_370` (FID / KID) | `all_314` (FID / KID) | Bedroom (FID / KID) | Living (FID / KID) | Dining (FID / KID) | Library (FID / KID) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`work_27`** | RAW | **41.76 / 0.0073** | **43.88 / 0.0080** | **52.19 / 0.0104** | **81.44 / 0.0163** | **63.79 / 0.0098** | **89.51 / 0.0035** |
| **`work_27`** | PP | **42.22 / 0.0072** | **44.51 / 0.0080** | **51.71 / 0.0097** | **83.24 / 0.0181** | **66.87 / 0.0115** | **89.12 / 0.0030** |
| **`work_37`** | RAW | 79.26 / 0.0420 | 83.85 / 0.0460 | 114.54 / 0.0802 | 110.23 / 0.0516 | 89.38 / 0.0329 | 115.74 / 0.0310 |
| **`work_37`** | PP | 78.02 / 0.0411 | 83.09 / 0.0455 | 113.17 / 0.0793 | 109.87 / 0.0506 | 88.58 / 0.0323 | 113.90 / 0.0276 |
| **`work_33`** | RAW | 145.84 / 0.1093 | 135.57 / 0.0946 | 272.10 / 0.2904 | 79.52 / 0.0150 | 61.51 / 0.0079 | 257.37 / 0.2393 |
| **`work_33`** | PP | 146.50 / 0.1087 | 136.42 / 0.0941 | 273.14 / 0.2915 | 82.51 / 0.0159 | 65.19 / 0.0088 | 257.52 / 0.2397 |

---

## 3. Detailed Results by Experiment

### Experiment: `work_27` (`real_num27`)
- **Source Directory**: `/Volumes/ExternalSSD/current_works/real_num27/2050`
- **Model Config**: `echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1` (interval: 5, start_ratio: 0.95, strength: 20.0)

#### A. `work27_RAW` (Without Lamps)
- **all_370** (370 scenes): **FID = 41.7629** | **KID = 0.007315**
- **all_314** (314 scenes): **FID = 43.8827** | **KID = 0.007972**
- **bedroom** (162 scenes): **FID = 52.1850** | **KID = 0.010371**
- **livingroom** (83 scenes): **FID = 81.4397** | **KID = 0.016291**
- **diningroom** (100 scenes): **FID = 63.7894** | **KID = 0.009811**
- **library** (56 scenes): **FID = 89.5126** | **KID = 0.003472**

#### B. `work27_PP` (Post-Processed / Collision-Resolved, Without Lamps)
- **all_370** (370 scenes): **FID = 42.2222** | **KID = 0.007181**
- **all_314** (314 scenes): **FID = 44.5079** | **KID = 0.007962**
- **bedroom** (162 scenes): **FID = 51.7065** | **KID = 0.009720**
- **livingroom** (83 scenes): **FID = 83.2357** | **KID = 0.018105**
- **diningroom** (100 scenes): **FID = 66.8673** | **KID = 0.011458**
- **library** (56 scenes): **FID = 89.1171** | **KID = 0.003042**

#### C. `work27_RAW` (Default Renders: With Lamps)
- **all_370** (370 scenes): **FID = 49.1164** | **KID = 0.009023**
- **all_314** (314 scenes): **FID = 51.6524** | **KID = 0.010050**
- **bedroom** (162 scenes): **FID = 66.0107** | **KID = 0.020046**
- **livingroom** (83 scenes): **FID = 87.4595** | **KID = 0.012256**
- **diningroom** (100 scenes): **FID = 72.6815** | **KID = 0.009269**
- **library** (56 scenes): **FID = 102.8962** | **KID = 0.006249**

---

### Experiment: `work_37`
- **Source Directory**: `/Volumes/ExternalSSD/current_works/work_37/2050`

#### A. `work37_RAW` (Without Lamps)
- **all_370** (370 scenes): **FID = 79.2553** | **KID = 0.042001**
- **all_314** (314 scenes): **FID = 83.8537** | **KID = 0.045963**
- **bedroom** (162 scenes): **FID = 114.5387** | **KID = 0.080215**
- **livingroom** (83 scenes): **FID = 110.2343** | **KID = 0.051578**
- **diningroom** (100 scenes): **FID = 89.3753** | **KID = 0.032942**
- **library** (56 scenes): **FID = 115.7355** | **KID = 0.030989**

#### B. `work37_PP` (Post-Processed / Collision-Resolved, Without Lamps)
- **all_370** (370 scenes): **FID = 78.0175** | **KID = 0.041080**
- **all_314** (314 scenes): **FID = 83.0937** | **KID = 0.045544**
- **bedroom** (162 scenes): **FID = 113.1703** | **KID = 0.079321**
- **livingroom** (83 scenes): **FID = 109.8740** | **KID = 0.050641**
- **diningroom** (100 scenes): **FID = 88.5778** | **KID = 0.032321**
- **library** (56 scenes): **FID = 113.9038** | **KID = 0.027646**

#### C. `work37_RAW` (Default Renders: With Lamps)
- **all_370** (370 scenes): **FID = 77.5251** | **KID = 0.034741**
- **all_314** (314 scenes): **FID = 81.8788** | **KID = 0.038532**
- **bedroom** (162 scenes): **FID = 114.7506** | **KID = 0.072909**
- **livingroom** (83 scenes): **FID = 104.8869** | **KID = 0.034048**
- **diningroom** (100 scenes): **FID = 92.8358** | **KID = 0.028984**
- **library** (56 scenes): **FID = 126.8715** | **KID = 0.025059**

---

### Experiment: `work_33`
- **Source Directory**: `/Volumes/ExternalSSD/current_works/work_33/2050`
- **Model Config**: `echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 + relational_loss * 0.5`

#### A. `work33_RAW` (Without Lamps)
- **all_370** (370 scenes): **FID = 145.8419** | **KID = 0.109263**
- **all_314** (314 scenes): **FID = 135.5721** | **KID = 0.094621**
- **bedroom** (162 scenes): **FID = 272.0963** | **KID = 0.290357**
- **livingroom** (83 scenes): **FID = 79.5235** | **KID = 0.014995**
- **diningroom** (100 scenes): **FID = 61.5112** | **KID = 0.007891**
- **library** (56 scenes): **FID = 257.3678** | **KID = 0.239300**

#### B. `work33_PP` (Post-Processed / Collision-Resolved, Without Lamps)
- **all_370** (370 scenes): **FID = 146.4978** | **KID = 0.108705**
- **all_314** (314 scenes): **FID = 136.4172** | **KID = 0.094054**
- **bedroom** (162 scenes): **FID = 273.1442** | **KID = 0.291500**
- **livingroom** (83 scenes): **FID = 82.5076** | **KID = 0.015854**
- **diningroom** (100 scenes): **FID = 65.1904** | **KID = 0.008806**
- **library** (56 scenes): **FID = 257.5236** | **KID = 0.239677**

#### C. `work33_RAW` (Default Renders: Pre-rendered `part2`)
- **all_370** (200 scenes): **FID = 70.1479** | **KID = 0.021570**
- **all_314** (190 scenes): **FID = 71.1860** | **KID = 0.022471**
- **bedroom** (38 scenes): **FID = 103.8540** | **KID = 0.021922**
- **livingroom** (83 scenes): **FID = 86.2603** | **KID = 0.010443**
- **diningroom** (100 scenes): **FID = 72.6529** | **KID = 0.010551**
- **library** (10 scenes): **FID = 153.3613** | **KID = 0.001461**

---

---

## 4. Recomputation (2026-10-18): both protocols, all runs on the same renders

All images are re-rendered lamp-free from layout JSON (`scripts/rendering/render_images_only.py --without_lamp`) and scored with `scripts/eval/fid_eval.py` against `sdf_fov90_h8_wo_lamp_no_stool/small/test`. KID ×10⁻³. +PP rows use the controlled resolver re-run (`scripts/eval/rerun_pp.py`).

**Validation:** on the paper's own render folders, the original protocol reproduces EchoScene + PP (bedroom 50.69 / 8.46) and Gaussian-v1 raw / +PP (58.50 / 14.84, 57.27 / 14.06) exactly.

> [!CAUTION]
> **The `work_33` numbers in sections 2–3 above are invalid.** `fid_eval_work33.py` used meshes from `work_33/part2` only (200 of 370 scenes), so 170 renders were blank. Corrected values are below. The paper's raw EchoScene row (bedroom 52.99 / 4.73 / 3.47, from `baseline_result.txt`) cannot be reproduced from any surviving render folder; use the recomputed EchoScene row.

### Original protocol (paper Table 1): FID / FID-CLIP / KID

| Model | bedroom | livingroom | diningroom | all_314 |
|---|---|---|---|---|
| EchoScene | 51.66 / 6.15 / 8.56 | 78.63 / 10.02 / 13.61 | 62.50 / 8.71 / 9.18 | 42.83 / 5.41 / 6.71 |
| EchoScene + PP | 50.72 / 6.19 / 8.52 | 79.36 / 10.60 / 14.40 | 65.92 / 9.46 / 10.87 | 43.37 / 5.69 / 7.15 |
| Gaussian-v1 (WACV model) | 58.50 / 7.32 / 14.84 | 80.92 / 9.06 / 15.01 | 68.06 / 9.14 / 12.38 | 47.40 / 5.96 / 10.21 |
| Gaussian-v1 + PP | 57.33 / 7.25 / 14.11 | 82.22 / 9.68 / 16.40 | 70.29 / 9.59 / 14.25 | 47.66 / 6.15 / 10.42 |
| Work 33 (Relation-v2 w=0.5) | 51.32 / 6.44 / 9.06 | 79.52 / 9.07 / 14.99 | 61.51 / 8.24 / 7.89 | 43.08 / 5.32 / 7.12 |
| Work 33 + PP | 51.87 / 6.41 / 9.83 | 82.51 / 9.82 / 15.85 | 65.19 / 9.15 / 8.81 | 44.35 / 5.66 / 7.54 |
| **Work 41** | 54.62 / 6.53 / 12.61 | 82.19 / 9.14 / 19.32 | 62.86 / 8.03 / 8.78 | 44.71 / 5.29 / 9.06 |
| **Work 41 + PP** | 54.37 / 6.62 / 12.69 | 82.44 / 9.53 / 17.30 | 63.06 / 8.33 / 7.33 | 44.77 / 5.46 / 8.80 |

### Newer protocol (Supp. Table VII): FID / KID

| Model | bedroom | livingroom | diningroom | library | all_314 | all_370 |
|---|---|---|---|---|---|---|
| EchoScene | 51.66 / 8.56 | 78.63 / 13.61 | 62.50 / 9.18 | 87.03 / 3.08 | 42.83 / 6.71 | 40.73 / 6.07 |
| EchoScene + PP | 50.72 / 8.52 | 79.36 / 14.40 | 65.92 / 10.87 | 87.60 / 1.31 | 43.37 / 7.15 | 41.02 / 6.17 |
| Gaussian-v1 (WACV model) | 58.50 / 14.84 | 80.92 / 15.01 | 68.06 / 12.38 | 99.13 / 9.89 | 47.40 / 10.21 | 45.86 / 9.61 |
| Gaussian-v1 + PP | 57.33 / 14.11 | 82.22 / 16.40 | 70.29 / 14.25 | 97.18 / 8.78 | 47.66 / 10.42 | 45.66 / 9.66 |
| Work 33 (Relation-v2 w=0.5) | 51.32 / 9.06 | 79.52 / 14.99 | 61.51 / 7.89 | 89.34 / 5.31 | 43.08 / 7.12 | 41.16 / 6.80 |
| Work 33 + PP | 51.87 / 9.83 | 82.51 / 15.85 | 65.19 / 8.81 | 89.09 / 4.50 | 44.35 / 7.54 | 42.07 / 6.91 |
| **Work 41** | 54.62 / 12.61 | 82.19 / 19.32 | 62.86 / 8.78 | 97.01 / 10.44 | 44.71 / 9.06 | 43.81 / 9.06 |
| **Work 41 + PP** | 54.37 / 12.69 | 82.44 / 17.30 | 63.06 / 7.33 | 96.84 / 9.53 | 44.77 / 8.80 | 43.67 / 8.69 |

Raw JSON: `docs/results_2026-10-18/fid_original/`, `fid_new/`. Full audit: `docs/paper_number_audit.md`.

## 4. How to Calculate FID & KID

To reproduce or compute FID / KID for any new generated folder:

```bash
# Example command using CleanFID script
python3 echoscene/scripts/compute_fid_scores_3dfront.py \
    --path_to_real_renderings echoscene/dataset_gt_renders/sdf_renders/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings /path/to/generated_renders_wo_lamp
```
