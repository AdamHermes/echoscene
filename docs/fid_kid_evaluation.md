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

## 4. How to Calculate FID & KID

To reproduce or compute FID / KID for any new generated folder:

```bash
# Example command using CleanFID script
python3 echoscene/scripts/compute_fid_scores_3dfront.py \
    --path_to_real_renderings echoscene/dataset_gt_renders/sdf_renders/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings /path/to/generated_renders_wo_lamp
```
