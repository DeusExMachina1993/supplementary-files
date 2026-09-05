# Post-hoc routed BWT results — PatchExpertFFN_refresh2_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6058 | -0.079593 | YES | 0.3409 | -0.3445 |
| 1 | 0.6698 | 0.6314 | -0.038372 | YES | 0.3351 | -0.3346 |
| 2 | 0.7137 | 0.7137 | +0.000000 | YES | 0.3969 | -0.3168 |
| 3 | 0.7088 | 0.7088 | +0.000000 | YES | 0.5213 | -0.1876 |
| 4 | 0.7073 | — | — | — | — | — |

**BWT_routed (seed 1)** = -0.029491

**BWT_unrouted (seed 1)** = -0.2959

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6217 | -0.063723 | YES | 0.3367 | -0.3487 |
| 1 | 0.6676 | 0.6407 | -0.026840 | YES | 0.3283 | -0.3393 |
| 2 | 0.7077 | 0.7077 | +0.000000 | YES | 0.3911 | -0.3166 |
| 3 | 0.7036 | 0.7036 | +0.000000 | YES | 0.5157 | -0.1879 |
| 4 | 0.7005 | — | — | — | — | — |

**BWT_routed (seed 2)** = -0.022641

**BWT_unrouted (seed 2)** = -0.2981

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6302 | -0.055194 | YES | 0.3413 | -0.3441 |
| 1 | 0.6713 | 0.6375 | -0.033841 | YES | 0.3337 | -0.3376 |
| 2 | 0.7116 | 0.7116 | +0.000000 | YES | 0.3953 | -0.3163 |
| 3 | 0.7016 | 0.7016 | +0.000000 | YES | 0.5166 | -0.1850 |
| 4 | 0.7041 | — | — | — | — | — |

**BWT_routed (seed 3)** = -0.022259

**BWT_unrouted (seed 3)** = -0.2957

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | -0.024797 | 0.003323 | ['-0.029491', '-0.022641', '-0.022259'] |
| BWT_unrouted | -0.2966 | 0.0011 | ['-0.2959', '-0.2981', '-0.2957'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

