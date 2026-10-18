# Where work 41 is better (paired, per-scene, 95% bootstrap CIs)

*2026-10-18. All 370 scenes, controlled re-run layouts (`ROOM/revision/pp_controlled/{pp,raw}`).*

**Method.** Per-scene values come from `scripts/eval/per_scene_metrics.py` and the CIs from `scripts/procthor_eval/bootstrap_nav.py` (10k resamples). Δ is the paired per-scene difference; * means the CI excludes 0. Raw JSON: `ROOM/revision/pp_controlled/per_scene_v3/`.

**Relation metrics** are per-scene means. The published table pools triples (micro average), so absolute values differ slightly from it; the comparisons are what matter here.

**Collision metrics** are reported three ways:

| Variant | What it does |
|---|---|
| **legacy** | Exact replica of `scripts/eval_collision.py`. It reproduces every published ColObj/ColScene value, including 0.0129 / 0.1913 for work 41 and 0.0179 / 0.2550 for EchoScene. |
| **corrected** | Fixes three bugs in that evaluator (below). |
| **corrected + yaw fix** | Also mirrors the object rotation (see `navigability_v2.md`, *Orientation fix*). |

The three bugs in `eval_collision.py`:

1. **Wardrobes are excluded.** Class index 13 is the wardrobe, not "floor/outer_room".
2. **The height check is shifted.** It treats y as the box centre, but y is the bottom.
3. **Rotated footprints are distorted.** It computes the z-coordinate as `z + dz·sin + dz·cos` (a typo for `dx·sin`).

## work 41 vs EchoScene (no guidance)

| Metric | Stage | EchoScene | work 41 | Δ [95% CI] |
|---|---|---|---|---|
| ColObj, legacy (published) | RAW | 0.229 | 0.162 | −0.067 [−0.088, −0.045] * |
| ColObj, corrected | RAW | 0.459 | 0.357 | −0.102 [−0.125, −0.079] * |
| ColObj, corrected + yaw fix | RAW | 0.457 | 0.362 | −0.096 [−0.119, −0.072] * |
| ColScene, corrected | RAW | 0.811 | 0.711 | −0.100 [−0.138, −0.062] * |
| OOB objects / scene | RAW | 1.214 | 0.478 | −0.735 [−0.854, −0.622] * |
| ColObj, corrected | PP | 0.031 | 0.012 | −0.019 [−0.031, −0.008] * |
| ColObj, corrected + yaw fix | PP | 0.050 | 0.033 | −0.017 [−0.032, −0.002] * |
| ColScene, corrected | PP | 0.070 | 0.030 | −0.041 [−0.065, −0.016] * |
| OOB objects / scene | PP | 0.254 | 0.024 | −0.230 [−0.297, −0.165] * |
| PP mean move (cm) | — | 14.3 | 4.6 | −9.7 [−10.7, −8.7] * |
| Objects moved > 30 cm by PP | — | 17.1% | 3.5% | −13.6 pts [−15.6, −11.6] * |
| Relation accuracy, total | RAW | 96.2 | 96.9 | +0.8 [+0.5, +1.1] * |
| Relation accuracy, macro | RAW | 91.9 | 93.3 | +1.4 [+0.8, +1.9] * |
| Close by | RAW | 73.5 | 78.4 | +4.9 [+2.2, +7.7] * |
| Symmetrical to | RAW | 47.8 | 52.1 | +4.4 [−2.9, +11.4] |
| Relation accuracy, total | PP | 96.4 | 96.9 | +0.6 [+0.3, +0.9] * |
| Relation accuracy, macro | PP | 92.0 | 93.3 | +1.3 [+0.7, +1.9] * |
| Close by | PP | 74.1 | 78.1 | +4.1 [+1.5, +6.6] * |
| Symmetrical to | PP | 43.4 | 52.1 | +8.7 [+1.2, +16.1] * |
| Taller / shorter | both | 95.9 | 97.3 | +1.4 [+0.3, +2.8] * |
| Navigability (all metrics, `navigability_v2.md`) | PP | — | — | tie (all CIs contain 0) |

## work 41 vs Gaussian-v1 (WACV main model)

| Metric | Δ work 41 − Gaussian-v1 [95% CI] |
|---|---|
| Relation total (PP) | +1.9 [+1.4, +2.4] * |
| Relation macro (PP) | +2.5 [+1.7, +3.3] * |
| Close by (PP) | +11.2 [+7.5, +15.0] * |
| OOB / scene (RAW) | −0.235 [−0.332, −0.143] * |
| PP mean move (cm) | −1.0 [−1.7, −0.3] * |
| Objects moved > 30 cm | −1.4 pts [−2.5, −0.2] * |
| ColObj, PP (corrected; corrected + yaw fix) | tie (+0.000; −0.000) |
| ColObj, RAW (corrected) | **+0.079 [+0.052, +0.106]**: work 41 collides more before PP |
| Navigability: functional access / ≥0.8 m corridor reach / usable-zone clear | −2.3 * / −11.4 * / −2.5 *: Gaussian-v1 better |

## Claims the data supports for work 41

1. **More faithful to the input scene graph.** It has the best relational accuracy of all runs: +1.3 macro vs EchoScene and +2.5 vs Gaussian-v1. The biggest gains are on the hard relations, *close by* (+4 vs EchoScene, +11 vs Gaussian-v1) and *symmetrical to* (+9 vs EchoScene on PP).
2. **Physically cleaner raw layouts than EchoScene.** It has 22% fewer colliding objects before any post-processing and 61% fewer out-of-bounds objects. This holds under the published evaluator, the corrected one, and the yaw-fixed one.
3. **Needs far less post-processing.** The resolver moves objects 4.6 cm on average vs 14.3 cm for EchoScene. Only 3.5% of objects move more than 30 cm, vs 17%. So the final PP layout is close to what the model generated, and PP does not destroy its relations: relation accuracy is essentially unchanged by PP.
4. **After PP:** fewer collisions and almost no out-of-bounds objects (0.02 per scene vs 0.25), with navigability unchanged.

## Claims to avoid

- **Navigability or accessibility gains for work 41.** There are none (see `navigability_v2.md`); that is work 37's trade-off.
- **"Fewer raw collisions than Gaussian-v1."** It has more. Its advantage over Gaussian-v1 is relations and less PP movement at equal post-PP collision rates.
- **Published ColObj values without a caveat.** `eval_collision.py` excludes wardrobes and has a footprint typo. The ranking work 41 < EchoScene survives every variant, but absolute values roughly double for RAW when wardrobes are counted.
