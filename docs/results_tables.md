# SceneGraph Experiment Results

Mirror of the `ColObj Score` and `SmallTests` sheets from `SceneGraph Experiement Log.xlsx` (fresh rerun, Aug 18 2026, current converter).

- **ColObj Score** = 370-scene benchmark
- **SmallTests** = fixed 52-scene subset (20 bedrooms + 12 living rooms + 20 dining rooms)
- All values are fractions 0–1 (higher = better)
- "PP" rows = post-processed (collision-resolved) scenes
- SIG_obb evaluated on only 32/52 available scenes (20 bedrooms missing from source zips)

---

## Table 1: ColObj Score (370 scenes)

| Trial # | Trial Name | Method | Method Note | ColObj | ColScene | OB Score | Walkability | Navigability | Total Acc | Means of Means | L/R | F/B | Bi/Sm | Ta/Sh | Standing | Close By | Symm |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | current_best_SIGG | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_gausv1 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 | 0.1762 | 0.3784 | 0.7135 | 0.2421 | 0.6374 | 0.9560 | 0.8687 | 0.9891 | 0.9901 | 0.9470 | 0.9618 | 0.9909 | 0.6806 | 0.5216 |
| 3 | current_best_SIGG + pp | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_gausv1 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 | 0.0179 | 0.0568 | 0.0405 | 0.2355 | 0.6385 | 0.9560 | 0.8641 | 0.9918 | 0.9913 | 0.9470 | 0.9618 | 0.9909 | 0.6658 | 0.5000 |
| 1 | baseline | echoscene | — | 0.2550 | 0.4676 | 1.2162 | 0.2506 | 0.6430 | 0.9636 | 0.8820 | 0.9851 | 0.9935 | 0.9629 | 0.9623 | 0.9947 | 0.7494 | 0.5259 |
| 2 | baseline + pp | echoscene | — | 0.0179 | 0.0541 | 0.2541 | 0.2358 | 0.6219 | 0.9633 | 0.8693 | 0.9881 | 0.9985 | 0.9629 | 0.9623 | 0.9947 | 0.7133 | 0.4655 |
| — | GROUND TRUTH | GROUND TRUTH | GROUND TRUTH | 0.4189 | 0.7081 | N/A | 0.3192 | 0.8573 | — | — | — | — | — | — | — | — | — |
| — | GROUND TRUTH + PP | GROUND TRUTH | GROUND TRUTH | — | — | N/A | 0.3042 | 0.8297 | — | — | — | — | — | — | — | — | — |
| 28 | work28 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 | interval: 1 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 | 0.1294 | 0.2158 | 0.5419 | 0.2185 | 0.7090 | 0.9501 | 0.8545 | 0.9946 | 0.9967 | 0.9272 | 0.9570 | 0.9936 | 0.6127 | 0.5000 |
| 28 | work28 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 | interval: 1 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 | 0.0142 | 0.2158 | 0.0541 | 0.2113 | 0.6892 | 0.9506 | 0.8535 | 0.9953 | 0.9967 | 0.9272 | 0.9570 | 0.9936 | 0.6241 | 0.4804 |
| 27 | work27 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 | 0.2150 | 0.4135 | 0.8378 | 0.2503 | 0.6365 | 0.9625 | 0.8746 | 0.9889 | 0.9945 | 0.9613 | 0.9589 | 0.9947 | 0.7281 | 0.4957 |
| 27 | work27 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 | 0.0154 | 0.0486 | 0.1622 | 0.2390 | 0.6124 | 0.9632 | 0.8667 | 0.9908 | 0.9992 | 0.9613 | 0.9589 | 0.9947 | 0.7224 | 0.4397 |
| 29 | SIGG_walkv3 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 | 0.0246 | 0.0784 | 1.9757 | 0.3092 | 0.8338 | 0.8687 | 0.7610 | 0.9603 | 0.9633 | 0.7688 | 0.9108 | 0.9229 | 0.4087 | 0.3922 |
| 29 | SIGG_walkv3 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 | 0.0042 | 0.0135 | 1.3514 | 0.3762 | 0.8138 | 0.8678 | 0.7580 | 0.9596 | 0.9623 | 0.7688 | 0.9108 | 0.9229 | 0.3980 | 0.3836 |
| 30 | work30 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 + relational_loss * 0.5 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.0367 | 0.1081 | 0.9757 | 0.2426 | 0.8179 | 0.8941 | 0.7728 | 0.9921 | 0.9884 | 0.7883 | 0.9511 | 0.9368 | 0.4169 | 0.3362 |
| 30 | work30 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 + relational_loss * 0.5 | interval: 1 start_ratio: 0.9 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.0033 | 0.0108 | 0.0000 | 0.2360 | 0.8046 | 0.8940 | 0.7739 | 0.9915 | 0.9884 | 0.7883 | 0.9511 | 0.9368 | 0.4161 | 0.3448 |
| 31 | work31 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.2304 | 0.4432 | 0.8216 | 0.2382 | 0.6369 | 0.9637 | 0.8776 | 0.9876 | 0.9936 | 0.9619 | 0.9627 | 0.9971 | 0.7445 | 0.4957 |
| 31 | work31 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.0183 | 0.0568 | 0.2189 | 0.2326 | 0.6111 | 0.9636 | 0.8693 | 0.9901 | 0.9982 | 0.9619 | 0.9627 | 0.9971 | 0.7142 | 0.4612 |
| 32 | work32 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.2283 | 0.4216 | 0.8571 | 0.2420 | 0.6165 | 0.9633 | 0.8803 | 0.9883 | 0.9943 | 0.9615 | 0.9578 | 0.9976 | 0.7453 | 0.5172 |
| 32 | work32 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix0.5-0.5 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.0175 | 0.0541 | 0.8571 | 0.2327 | 0.6091 | 0.9635 | 0.8774 | 0.9897 | 0.9992 | 0.9615 | 0.9578 | 0.9976 | 0.7191 | 0.5172 |
| 33 | work33 | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.2246 | 0.4324 | 0.8913 | 0.2373 | 0.6373 | 0.9642 | 0.8825 | 0.9904 | 0.9941 | 0.9627 | 0.9620 | 0.9976 | 0.7273 | 0.5431 |
| 33 | work33 + PP | echoscene + collision_loss * 10 + room_outer_loss * 10 + walkable_loss_v3_mix1-1 + relational_loss * 0.5 | interval: 5 start_ratio: 0.95 grad_clip: 10.1 strength: 20.0 relational: weight 0.5 start_ratio 0.5 | 0.0096 | 0.0297 | 0.8913 | 0.2276 | 0.6252 | 0.9646 | 0.8818 | 0.9915 | 0.9985 | 0.9627 | 0.9620 | 0.9976 | 0.7084 | 0.5517 |
| 34 | work34 | echoscene (method not filled in sheet) | — | 0.2562 | 0.4730 | 0.9162 | 0.2496 | 0.6237 | 0.9622 | 0.8803 | 0.9870 | 0.9923 | 0.9628 | 0.9600 | 0.9976 | 0.7150 | 0.5474 |
| 34 | work34 + PP | echoscene (method not filled in sheet) | — | 0.0158 | 0.0514 | 0.9162 | 0.2353 | 0.6192 | 0.9628 | 0.8689 | 0.9896 | 0.9978 | 0.9628 | 0.9600 | 0.9976 | 0.7002 | 0.4741 |
| 35 | work35 | echoscene (method not filled in sheet) | — | 0.2117 | 0.4054 | 1.3676 | 0.2498 | 0.6550 | 0.9620 | 0.8758 | 0.9892 | 0.9943 | 0.9609 | 0.9605 | 0.9971 | 0.7027 | 0.5259 |
| 35 | work35 + PP | echoscene (method not filled in sheet) | — | 0.0125 | 0.0405 | 0.1973 | 0.2358 | 0.6247 | 0.9626 | 0.8707 | 0.9904 | 0.9987 | 0.9609 | 0.9605 | 0.9971 | 0.6962 | 0.4914 |
| 36 | work36 | echoscene (method not filled in sheet) | — | 0.2442 | 0.4378 | 0.7838 | 0.2428 | 0.6111 | 0.9636 | 0.8848 | 0.9863 | 0.9941 | 0.9587 | 0.9607 | 0.9976 | 0.7658 | 0.5302 |
| 36 | work36 + PP | echoscene (method not filled in sheet) | — | 0.0179 | 0.0568 | 0.7838 | 0.2318 | 0.5983 | 0.9628 | 0.8741 | 0.9879 | 0.9983 | 0.9587 | 0.9607 | 0.9976 | 0.7281 | 0.4871 |
| 37 | work37 | echoscene (method not filled in sheet) | — | 0.0367 | 0.1054 | 0.6957 | 0.2342 | 0.8529 | 0.9216 | 0.8289 | 0.9961 | 0.9968 | 0.8358 | 0.9660 | 0.9650 | 0.5553 | 0.4871 |
| 37 | work37 + PP | echoscene (method not filled in sheet) | — | 0.0025 | 0.0081 | 0.6957 | 0.2279 | 0.8320 | 0.9215 | 0.8231 | 0.9958 | 0.9977 | 0.8358 | 0.9660 | 0.9650 | 0.5577 | 0.4440 |
| 38 | work38 | echoscene (method not filled in sheet) | — | 0.3321 | 0.5703 | 0.9459 | 0.2326 | 0.6576 | 0.9529 | 0.8650 | 0.9689 | 0.9703 | 0.9611 | 0.9675 | 0.9765 | 0.7322 | 0.4784 |
| 38 | work38 + PP | echoscene (method not filled in sheet) | — | 0.0208 | 0.0649 | 0.9459 | 0.2294 | 0.6310 | 0.9586 | 0.8434 | 0.9849 | 0.9908 | 0.9611 | 0.9675 | 0.9765 | 0.7125 | 0.3103 |
| 39 | work39 | echoscene (method not filled in sheet) | — | 0.0825 | 0.2405 | 0.8676 | 0.2199 | 0.8205 | 0.9229 | 0.8287 | 0.9924 | 0.9928 | 0.8499 | 0.9635 | 0.9516 | 0.5676 | 0.4828 |
| 39 | work39 + PP | echoscene (method not filled in sheet) | — | 0.0100 | 0.0324 | 0.8676 | 0.2126 | 0.8177 | 0.9237 | 0.8251 | 0.9945 | 0.9972 | 0.8499 | 0.9635 | 0.9516 | 0.5577 | 0.4612 |
| 40 | work40 | echoscene (method not filled in sheet) | — | 0.0283 | 0.0811 | 2.1946 | 0.2558 | 0.7951 | 0.9313 | 0.8403 | 0.9969 | 0.9995 | 0.8738 | 0.9640 | 0.9693 | 0.5053 | 0.5733 |
| 40 | work40 + PP | echoscene (method not filled in sheet) | — | 0.0050 | 0.0135 | 2.1946 | 0.2236 | 0.7747 | 0.9306 | 0.8307 | 0.9928 | 0.9963 | 0.8738 | 0.9640 | 0.9693 | 0.5446 | 0.4741 |
| 41 | work41 | echoscene (method not filled in sheet) | — | 0.1913 | 0.4000 | 0.4784 | 0.2221 | 0.6608 | 0.9692 | 0.8998 | 0.9871 | 0.9968 | 0.9662 | 0.9678 | 0.9952 | 0.7952 | 0.5905 |
| 41 | work41 + PP | echoscene (method not filled in sheet) | — | 0.0129 | 0.0378 | 0.4784 | 0.2178 | 0.6492 | 0.9688 | 0.8940 | 0.9896 | 0.9978 | 0.9662 | 0.9678 | 0.9952 | 0.7723 | 0.5690 |
| 42 | work42 | echoscene (method not filled in sheet) | — | 0.2029 | 0.4081 | 0.9351 | 0.2355 | 0.6486 | 0.9599 | 0.8699 | 0.9863 | 0.9919 | 0.9561 | 0.9649 | 0.9823 | 0.7297 | 0.4784 |
| 42 | work42 + PP | echoscene (method not filled in sheet) | — | 0.0167 | 0.0459 | 0.9351 | 0.2288 | 0.6336 | 0.9603 | 0.8613 | 0.9889 | 0.9953 | 0.9561 | 0.9649 | 0.9823 | 0.7191 | 0.4224 |
| 42.2 | work42_2 | echoscene (method not filled in sheet) | — | 0.2029 | 0.4027 | 0.9324 | 0.2181 | 0.6754 | 0.9593 | 0.8676 | 0.9861 | 0.9915 | 0.9522 | 0.9691 | 0.9832 | 0.7297 | 0.4612 |
| 42.2 | work42_2 + PP | echoscene (method not filled in sheet) | — | 0.0204 | 0.0595 | 0.9324 | 0.2131 | 0.6632 | 0.9589 | 0.8591 | 0.9863 | 0.9930 | 0.9522 | 0.9691 | 0.9832 | 0.7207 | 0.4095 |

---

## Table 2: SmallTests (52 scenes)

| Trial Name | Method | Method Note | ColObj | ColScene | OB Score | Walkability | Navigability | Total Acc | Means of Means | L/R | F/B | Bi/Sm | Ta/Sh | Standing | Close By | Symm |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | PhysScene Baseline | 52 Rooms | 0.2791 | 0.5000 | 1.0577 | 0.2388 | 0.6109 | 0.9645 | 0.8855 | 0.9815 | 0.9892 | 0.9689 | 0.9666 | 0.9937 | 0.7529 | 0.5455 |
| baseline + pp | PhysScene Baseline + OBB PP | 52 Rooms | 0.0163 | 0.0577 | 0.2885 | 0.2392 | 0.5798 | 0.9637 | 0.8589 | 0.9826 | 1.0000 | 0.9689 | 0.9666 | 0.9937 | 0.7069 | 0.3939 |
| SIG | echoscene (released full model) | 52 Rooms | 0.1816 | 0.4038 | 0.7885 | 0.1983 | 0.5859 | 0.9576 | 0.8600 | 0.9862 | 0.9930 | 0.9517 | 0.9621 | 0.9842 | 0.7184 | 0.4242 |
| SIG + pp | echoscene + OBB PP | 52 Rooms | 0.0000 | 0.0000 | 0.0577 | 0.1964 | 0.6133 | 0.9586 | 0.8551 | 0.9930 | 0.9941 | 0.9517 | 0.9621 | 0.9842 | 0.7069 | 0.3939 |
| SIG Best 0.95 | echoscene + SIG 0.95 guidance | 52 Rooms | 0.1766 | 0.4231 | 3.3462 | 0.2432 | 0.7872 | 0.9540 | 0.8839 | 0.9827 | 0.9893 | 0.9364 | 0.9651 | 0.9779 | 0.7299 | 0.6061 |
| SIG Best 0.95 + pp | echoscene + SIG 0.95 guidance + OBB PP | 52 Rooms | 0.0163 | 0.0577 | 2.5385 | 0.3433 | 0.5993 | 0.9502 | 0.8681 | 0.9791 | 0.9929 | 0.9364 | 0.9651 | 0.9779 | 0.6494 | 0.5758 |
| SIG Best obb | echoscene + OBB guidance | 52 Rooms¹ | 0.2194 | 0.5000 | 1.0000 | 0.2023 | 0.5585 | 0.9509 | 0.8563 | 0.9836 | 0.9849 | 0.9413 | 0.9541 | 0.9870 | 0.6986 | 0.4444 |
| SIG Best obb + pp | echoscene + OBB guidance + OBB PP | 52 Rooms¹ | 0.0144 | 0.0625 | 0.0962 | 0.2039 | 0.5638 | 0.9505 | 0.8369 | 0.9905 | 0.9876 | 0.9413 | 0.9541 | 0.9870 | 0.6644 | 0.3333 |
| SIG Best 5 | echoscene + SIG int5 guidance | 52 Rooms | 0.1902 | 0.4231 | 3.3269 | 0.2652 | 0.7964 | 0.9545 | 0.8853 | 0.9838 | 0.9869 | 0.9392 | 0.9619 | 0.9779 | 0.7414 | 0.6061 |
| SIG Best 5 + pp | echoscene + SIG int5 guidance + OBB PP | 52 Rooms | 0.0217 | 0.0769 | 2.6538 | 0.3476 | 0.6000 | 0.9509 | 0.8725 | 0.9802 | 0.9929 | 0.9392 | 0.9619 | 0.9779 | 0.6494 | 0.6061 |
| SIG Best walkability v3 | echoscene + SIG walkability v3 guidance | 52 Rooms | 0.0108 | 0.0385 | 2.5962 | 0.3043 | 0.7991 | 0.8631 | 0.7596 | 0.9511 | 0.9715 | 0.7575 | 0.8854 | 0.9325 | 0.4253 | 0.3939 |
| SIG Best walkability v3 + pp | echoscene + SIG walkability v3 guidance + OBB PP | 52 Rooms | 0.0054 | 0.0192 | 2.1346 | 0.4154 | 0.7729 | 0.8636 | 0.7612 | 0.9546 | 0.9680 | 0.7575 | 0.8854 | 0.9325 | 0.4368 | 0.3939 |
| SIG Best walkability v3_no_gausv1 | echoscene + SIG walkability v3_no_gausv1 guidance | 52 Rooms | 0.0108 | 0.0385 | 0.4038 | 0.1926 | 0.7123 | 0.9124 | 0.8058 | 0.9988 | 0.9988 | 0.8208 | 0.9444 | 0.9495 | 0.5345 | 0.3939 |
| SIG Best walkability v3_no_gausv1 + pp | echoscene + SIG walkability v3_no_gausv1 guidance + OBB PP | 52 Rooms | 0.0000 | 0.0000 | 0.0000 | 0.1903 | 0.7075 | 0.9124 | 0.8058 | 0.9988 | 0.9988 | 0.8208 | 0.9444 | 0.9495 | 0.5345 | 0.3939 |
| SIG Best no_walk | echoscene + SIG no_walk guidance | 52 Rooms | 0.1707 | 0.3654 | 0.4615 | 0.1981 | 0.5885 | 0.9653 | 0.8887 | 0.9919 | 0.9928 | 0.9574 | 0.9668 | 0.9905 | 0.7759 | 0.5455 |
| SIG Best no_walk + pp | echoscene + SIG no_walk guidance + OBB PP | 52 Rooms | 0.0054 | 0.0192 | N/A | 0.1963 | 0.5720 | 0.9632 | 0.8703 | 0.9930 | 0.9940 | 0.9574 | 0.9668 | 0.9905 | 0.7356 | 0.4545 |
| SIG rerun | echoscene + SIG rerun guidance | 52 Rooms | 0.1870 | 0.3846 | 1.1538 | 0.1889 | 0.6154 | 0.9448 | 0.8511 | 0.9756 | 0.9798 | 0.9328 | 0.9525 | 0.9685 | 0.7241 | 0.4242 |
| SIG rerun + pp | echoscene + SIG rerun guidance + OBB PP | 52 Rooms | 0.0163 | 0.0577 | N/A | 0.1857 | 0.6218 | 0.9456 | 0.8487 | 0.9717 | 0.9858 | 0.9328 | 0.9525 | 0.9685 | 0.7356 | 0.3939 |
| SIG relationv2_no_walk | echoscene + SIG relationv2_no_walk guidance | 52 Rooms | 0.1870 | 0.4231 | 0.4615 | 0.1937 | 0.6189 | 0.9709 | 0.8966 | 0.9918 | 0.9988 | 0.9668 | 0.9664 | 0.9905 | 0.8161 | 0.5455 |
| SIG relationv2_no_walk + pp | echoscene + SIG relationv2_no_walk guidance + OBB PP | 52 Rooms | 0.0000 | 0.0000 | N/A | 0.1876 | 0.6066 | 0.9678 | 0.8692 | 0.9918 | 0.9988 | 0.9668 | 0.9664 | 0.9905 | 0.7759 | 0.3939 |
| SIG relationv2 | echoscene + SIG relationv2 guidance | 52 Rooms | 0.1951 | 0.4231 | 0.9808 | 0.1780 | 0.6522 | 0.9440 | 0.8458 | 0.9838 | 0.9822 | 0.9346 | 0.9524 | 0.9306 | 0.7126 | 0.4242 |
| SIG relationv2 + pp | echoscene + SIG relationv2 guidance + OBB PP | 52 Rooms | 0.0108 | 0.0385 | 0.0192 | 0.1892 | 0.6217 | 0.9451 | 0.8368 | 0.9873 | 0.9882 | 0.9346 | 0.9524 | 0.9306 | 0.7011 | 0.3636 |
| SIG walkv2 | echoscene + SIG walkv2 guidance | 52 Rooms | 0.1789 | 0.3462 | 0.5000 | 0.2146 | 0.5785 | 0.9676 | 0.8904 | 0.9895 | 0.9988 | 0.9612 | 0.9683 | 0.9937 | 0.7759 | 0.5455 |
| SIG walkv2 + pp | echoscene + SIG walkv2 guidance + OBB PP | 52 Rooms | 0.0081 | 0.0192 | 0.0577 | 0.2127 | 0.5862 | 0.9678 | 0.8915 | 0.9940 | 1.0000 | 0.9612 | 0.9683 | 0.9937 | 0.7471 | 0.5758 |

¹ Only 32/52 scenes available (20 dining + 12 living; 20 bedrooms missing from source zips).

---

## Notes

- Ground truth rows only have collision + walkability/navigability filled in the sheet.
- "ColObj" = object collision rate, "ColScene" = scene collision rate (lower = better).
- L/R = Left/Right, F/B = Front/Behind, Bi/Sm = Bigger/Smaller, Ta/Sh = Taller/Shorter,
  Standing = Standing On, Close By, Symm = Symmetrical To.
- SIG (SmallTests) = plain echoscene released full model, reconstructed from SSD
  `to_be_merged` chunk JSONs (canonical 52 verified identical to baseline_52).