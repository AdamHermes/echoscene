# FAQ: Why Baseline and SteerScene Lack Library / All 370 FID & KID Metrics

> [!NOTE]
> **Update 2026-10-18:** Library and All-370 scores now exist for the EchoScene and Gaussian-v1 rows (EchoScene raw: Library 87.03 / 3.08, All-370 40.73 / 6.07; Gaussian-v1 + PP: Library 97.18 / 8.78, All-370 45.66 / 9.66; KID ×10⁻³). The work 33 ("Relation-v2 weight 0.5") Library 257.37 in the table below came from a rendering bug (170 blank images); the corrected value is 89.34. See `fid_kid_evaluation.md` §4.


This document clarifies why the **EchoScene baseline** and **Selected SteerScene** rows contain dashes (`– –`) for the **Library** and **All 370** columns in the benchmark comparison tables (such as Table VII in the supplementary material and Table 1 in the main paper), while newer ablation configurations report full numbers.

---

## 1. Quick Summary (Simple Terms)

1. **The prior papers never tested Library**: Earlier works (ATISS, DiffuScene, CommonScenes, and original EchoScene) only evaluated **3 room types**: Bedroom, Living Room, and Dining Room (314 scenes). Library was omitted because it only has 56 test rooms in 3D-FRONT.
2. **Table 1 followed that convention**: When the authors first evaluated the **Baseline** and **Selected SteerScene** for the main paper's Table 1, they only tested those 3 standard room types to enable a direct, fair comparison against published baselines.
3. **A newer script evaluated all 370 rooms**: Later ablation runs (*Light Gaussian guidance*, *High-weight Relation-v2*, *Relation-v2 weight 0.5*) used a newly written script (`fid_eval_work33.py`) that explicitly looped over every room category, including the 56 Library scenes and the combined 370-room set.
4. **Why the dashes (`– –`) exist**: When constructing the supplementary comparison table (Table VII), the authors copied the baseline and SteerScene numbers directly from Table 1. Because the new FID script was never rerun on the legacy baseline archives, those Library scores simply did not exist. Rather than fabricating or approximating numbers, they were marked with dashes (`– –`).

---

## 2. Table Context: Table VII in Supplementary Material

```text
Configuration                 Variant   All 370          Bedroom          Living Room      Dining Room      Library
                                        FID    KID       FID    KID       FID    KID       FID    KID       FID    KID
EchoScene baseline            Raw       –      –         52.99  0.00347   84.96  0.00972   72.15  0.01093   –      –
EchoScene baseline            +PP       –      –         50.69  0.00846   79.45  0.01452   65.99  0.01091   –      –
Selected SteerScene           Guided    –      –         58.50  0.01484   80.92  0.01501   68.06  0.01238   –      –
Selected SteerScene           +PP       –      –         57.28  0.01406   82.24  0.01640   70.31  0.01425   –      –
Light Gaussian guidance       Guided    41.76  0.0073    52.19  0.0104    81.44  0.0163    63.79  0.0098    89.51  0.0035
Light Gaussian guidance       +PP       42.22  0.0072    51.71  0.0097    83.24  0.0181    66.87  0.0115    89.12  0.0030
High-weight Relation-v2       Guided    79.26  0.0420    114.54 0.0802    110.23 0.0516    89.38  0.0329    115.74 0.0310
High-weight Relation-v2       +PP       78.02  0.0411    113.17 0.0793    109.87 0.0506    88.58  0.0323    113.90 0.0276
Relation-v2 weight 0.5        Guided    145.84 0.1093    272.10 0.2904    79.52  0.0150    61.51  0.0079    257.37 0.2393
Relation-v2 weight 0.5        +PP       146.50 0.1087    273.14 0.2915    82.51  0.0159    65.19  0.0088    257.52 0.2397
```

As stated in `supp_text.txt`:
> *"For context, the table also includes the EchoScene baseline and the selected SteerScene model, before and after PP, using the category-level results from Table II. KID is converted from the ×10⁻³ presentation in Table II to its original scale. Aggregate all-370 and library scores were not reported for those four benchmark rows and are therefore marked with dashes rather than estimated."*

---

## 3. Technical & Statistical Reasons Why Earlier Papers Omitted Library

### A. Inception Covariance Estimation / Sample Size Bias ($N \ll 2048$)
- Fréchet Inception Distance (FID) evaluates the distance between two Gaussians in Inception-v3 feature space ($d = 2048$ dimensions):
  $$d^2 = \|\mu_r - \mu_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$
- The 3D-FRONT test split contains only **56 Library scenes**.
- Estimating a $2048 \times 2048$ covariance matrix $\Sigma$ from $N = 56$ samples is **severely rank-deficient** ($\text{rank} \le 55 \ll 2048$).
- FID is known to exhibit significant finite-sample bias ($\mathcal{O}(1/N)$) when $N$ is small, causing values to blow up and exhibit extreme variance. For example, on `work_33`, Library FID spiked to **257.37**.
- Even Kernel Inception Distance (KID), which is theoretically unbiased, typically samples subsets of size $N = 100$ per standard CleanFID protocol; with only 56 images, even a single full subset cannot be drawn without replacement.

### B. Extreme Class Imbalance in 3D-FRONT / SG-FRONT
In the SG-FRONT dataset splits:
- **Bedroom**: 3,879 train scenes / 162 test scenes
- **Living Room**: 730 train scenes / 83 test scenes
- **Dining Room**: 800 train scenes / 100 test scenes
- **Library**: only **230 train scenes** / **56 test scenes**

Because Library represents a small tail distribution, prior works focused on the three dominant room types.

### C. Missing Category-Specific Files in `FRONT/`
In `echoscene/FRONT/`, category-specific preprocessed assets exist for Bedroom, Living Room, and Dining Room, but **not** for Library:
- `boxes_centered_stats_{bedroom,livingroom,diningroom}_*.txt` exist, but no `boxes_centered_stats_library_*.txt`.
- `centered_bounds_{bedroom,livingroom,diningroom}_trainval.txt` exist, but no `centered_bounds_library_trainval.txt`.
- `relationships_{bedroom,livingroom,diningroom}_*.json` exist, but no `relationships_library_{trainval,test}.json`.

Consequently, single-category training or evaluation on `room_type='library'` was unsupported out-of-the-box in the original EchoScene / CommonScenes codebase.

### D. Hardcoded Legacy Evaluation Script
The official evaluation script inherited from ATISS and InstructScene (`echoscene/scripts/compute_fid_scores_3dfront.py`) hardcoded the categories:
```python
room_dict = {
    'bedroom': ["Bedroom", "MasterBedroom", "SecondBedroom"],
    'livingroom': ['LivingDiningRoom', 'LivingRoom'],
    'diningroom': ['LivingDiningRoom', 'DiningRoom'],
    'all': ["Bedroom", "MasterBedroom", "SecondBedroom", 'LivingDiningRoom', 'LivingRoom', 'DiningRoom']
}
```
Notice that `'all'` was explicitly defined as the 314-scene union of Bedroom, Living, and Dining, completely omitting Library.

---

## 4. How the Bottom Ablation Rows Computed Library FID/KID

1. **Joint Training Covered Library**: The joint model was trained on `relationships_all_trainval.json` (`--room_type all`), which included the 230 Library scenes.
2. **Ground Truth Renders Existed**: The GT directory (`echoscene/dataset_gt_renders/sdf_renders/sdf_fov90_h8_wo_lamp_no_stool/small/test`) includes all 370 canonical scenes, including the 56 `Library-*.png` files.
3. **Execution in `cleanfid`**: The newer script (`fid_eval_work33.py`) passed the folder of 56 synthetic Library renders and 56 real Library renders to `cleanfid.compute_fid()`. The library computes NumPy covariance and matrix square roots without throwing an error, returning a score regardless of sample-size limitations.

---

## 5. Why Physical Metrics (Table 2 / ColObj) Included Library

Physical metrics—Collision Rate (`ColObj`, `ColScene`) and Out-of-Bounds (`OOB`)—are evaluated using Oriented Bounding Box Separating Axis Theorem (OBB-SAT) tests on individual 3D scenes. Because geometric overlap checks do not depend on statistical distribution fitting or sample-size minimums, evaluating collision on 56 Library rooms is mathematically sound and was reported across all 370 scenes in Table 2.
