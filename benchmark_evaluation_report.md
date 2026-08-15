# 📊 EchoScene / ROOM Benchmark Evaluation Report

This report presents the complete physical layout, navigation simulation, and 3D spatial relational accuracy evaluation metrics across all **8 model versions**:

1. **Current Best Sig (Raw)** — `current_works/to_be_merged/complete_released_full_model`
2. **Current Best Sig (PP)** — `current_works/to_be_merged/complete_released_full_model_post_processed`
3. **Baseline (Raw)** — `baseline`
4. **Baseline (PP)** — `baseline_post_processed`
5. **Work 28 (Raw)** — `current_works/work_num28_attempt3`
6. **Work 28 (PP)** — `current_works/work_num28_attempt3_pp`
7. **Work 27 (Raw)** — `current_works/real_num27`
8. **Work 27 (PP)** — `current_works/real_num27_pp`

---

## 🏆 1. Master Evaluation Summary (All Columns)

| # | Model Version | JSON File Path | Total Scenes | Evaluated GT Scenes | Col Obj (↓) | Col Scene (↓) | Walkability (↑) | Navigability (↑) | Total Accuracy (↑) | Means of Means (↑) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Current Best Sig (Raw)** | [`to_be_merged/complete_released_full_model`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/to_be_merged/complete_released_full_model/vis/2050/physcene_collision_input.json) | 370 | 314 | 0.1489 | 0.2947 | 23.18% | 81.15% | 0.9586 | 0.8703 |
| 2 | **Current Best Sig (PP)** | [`to_be_merged/complete_released_full_model_post_processed`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/to_be_merged/complete_released_full_model_post_processed/vis/2050/physcene_collision_input.json) | 370 | 314 | 0.0154 | 0.0421 | 22.04% | 80.03% | 0.9589 | 0.8669 |
| 3 | **Baseline (Raw)** | [`baseline`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline/vis/2050/physcene_collision_input.json) | 369 | 313 | 0.2892 | 0.5105 | 23.52% | 79.26% | 0.9662 | 0.8883 |
| 4 | **Baseline (PP)** | [`baseline_post_processed`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/baseline_post_processed/vis/2050/physcene_collision_input.json) | 369 | 313 | 0.0270 | 0.0632 | 20.12% | 74.93% | 0.9657 | 0.8730 |
| 5 | **Work 28 (Raw)** | [`work_num28_attempt3`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3/2050/physcene_collision_input.json) | 370 | 314 | **0.1294** | **0.2158** | **24.40%** | **81.18%** | 0.9501 | 0.8545 |
| 6 | **Work 28 (PP)** | [`work_num28_attempt3_pp`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/work_num28_attempt3_pp/vis/2050/physcene_collision_input.json) | 370 | 314 | 🏆 **0.0142** | 🏆 **0.0211** | 🏆 **23.49%** | 🏆 **79.29%** | 0.9506 | 0.8535 |
| 7 | **Work 27 (Raw)** | [`real_num27`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/real_num27/2050/physcene_collision_input.json) | 370 | 314 | 0.1879 | 0.3316 | 22.91% | 79.74% | 0.9646 | 0.8796 |
| 8 | **Work 27 (PP)** | [`real_num27_pp`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/current_works/real_num27_pp/vis/2050/physcene_collision_input.json) | 370 | 314 | 0.0222 | 0.0579 | 21.79% | 76.84% | 0.9655 | 0.8709 |

---

## 📐 2. Detailed Relational Accuracy Category Breakdown (`0.abcd` Format)

Evaluated over all matching scenes against 3D-FRONT ground truth relational graphs:

| Model Version | Scenes | Total Accuracy | Means of Means | Left/Right (L/R) | Front/Behind (F/B) | Bigger/Smaller (Bi/Sm) | Taller/Shorter (Ta/Sh) | Standing On | Close By | Symmetrical To |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Current Best Sig (Raw)** | 314 | **0.9586** | **0.8703** | 0.9897 | 0.9923 | 0.9498 | 0.9629 | 0.9952 | 0.6875 | 0.5147 |
| **Current Best Sig (PP)** | 314 | **0.9589** | **0.8669** | 0.9925 | 0.9935 | 0.9498 | 0.9629 | 0.9952 | 0.6743 | 0.5000 |
| **Baseline (Raw)** | 313 | **0.9662** | **0.8883** | 0.9876 | 0.9930 | 0.9655 | 0.9631 | 0.9979 | 0.7644 | 0.5468 |
| **Baseline (PP)** | 313 | **0.9657** | **0.8730** | 0.9905 | 0.9987 | 0.9655 | 0.9631 | 0.9979 | 0.7227 | 0.4729 |
| **Work 28 (Raw)** | 314 | **0.9501** | **0.8545** | 0.9946 | 0.9967 | 0.9272 | 0.9570 | 0.9936 | 0.6127 | 0.5000 |
| **Work 28 (PP)** | 314 | **0.9506** | **0.8535** | 0.9953 | 0.9967 | 0.9272 | 0.9570 | 0.9936 | 0.6241 | 0.4804 |
| **Work 27 (Raw)** | 314 | **0.9646** | **0.8796** | 0.9907 | 0.9944 | 0.9619 | 0.9603 | 0.9979 | 0.7377 | 0.5147 |
| **Work 27 (PP)** | 314 | **0.9655** | **0.8709** | 0.9932 | 0.9993 | 0.9619 | 0.9603 | 0.9979 | 0.7324 | 0.4510 |

---

## 💥 3. Physical Scene Collision Breakdown by Room Category

### Work 28 (Raw vs PP)
- **Raw Object Collision:** `0.1294` (137/1059 objects collided)
- **Raw Scene Collision:** `0.2158` (41/190 scenes collided)
- **PP Object Collision:** `0.0142` (15/1059 objects collided) — 🟢 **-89.0% reduction**
- **PP Scene Collision:** `0.0211` (4/190 scenes collided) — 🟢 **-90.2% reduction**

**PP Category Details:**
- `bedroom`: ColObj `0.0000` \| ColScene `0.0000` (0/41 collided)
- `livingdiningroom`: ColObj `0.0000` \| ColScene `0.0000` (0/13 collided)
- `livingroom`: ColObj `0.0127` \| ColScene `0.0667` (1/15 collided)
- `masterbedroom`: ColObj `0.0325` \| ColScene `0.0290` (2/69 collided)
- `secondbedroom`: ColObj `0.0091` \| ColScene `0.0192` (1/52 collided)

---

### Work 27 (Raw vs PP)
- **Raw Object Collision:** `0.1879` (195/1038 objects collided)
- **Raw Scene Collision:** `0.3316` (63/190 scenes collided)
- **PP Object Collision:** `0.0222` (23/1038 objects collided) — 🟢 **-88.2% reduction**
- **PP Scene Collision:** `0.0579` (11/190 scenes collided) — 🟢 **-82.5% reduction**

---

### Baseline (Raw vs PP)
- **Raw Object Collision:** `0.2892` (328/1134 objects collided)
- **Raw Scene Collision:** `0.5105` (97/190 scenes collided)
- **PP Object Collision:** `0.0270` (28/1038 objects collided) — 🟢 **-90.7% reduction**
- **PP Scene Collision:** `0.0632` (12/190 scenes collided) — 🟢 **-87.6% reduction**

---

## 🛠️ 4. Code & Evaluation Script Reference

- **Collision Evaluation Script:** [`scripts/eval_collision.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/eval_collision.py)
- **ProcTHOR Walkability Evaluation Script:** [`eval_walkability.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/eval_walkability.py)
- **ProcTHOR Navigability Evaluation Script:** [`eval_navigation.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/eval_navigation.py)
- **3D Spatial Relational Accuracy Script:** [`scripts/relational/eval_room_types.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/scripts/relational/eval_room_types.py)
- **3D Spatial Constraint Function:** [`helpers/metrics_3dfront.py`](file:///Users/lehoangan/Documents/GitHub/ROOM/echoscene/helpers/metrics_3dfront.py#L57)
