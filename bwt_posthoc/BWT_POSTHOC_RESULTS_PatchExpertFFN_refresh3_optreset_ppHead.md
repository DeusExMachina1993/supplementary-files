# Post-hoc routed BWT results — PatchExpertFFN_refresh3_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5727 | -0.112731 | YES | 0.3389 | -0.3465 |
| 1 | 0.6698 | 0.6157 | -0.054091 | YES | 0.3272 | -0.3426 |
| 2 | 0.7035 | 0.6272 | -0.076298 | YES | 0.3764 | -0.3271 |
| 3 | 0.7053 | 0.7053 | +0.000000 | YES | 0.5246 | -0.1807 |
| 4 | 0.7031 | — | — | — | — | — |

**BWT_routed (seed 1)** = -0.060780

**BWT_unrouted (seed 1)** = -0.2992

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5613 | -0.124056 | YES | 0.3328 | -0.3526 |
| 1 | 0.6676 | 0.6238 | -0.043795 | YES | 0.3233 | -0.3442 |
| 2 | 0.6988 | 0.6319 | -0.066853 | YES | 0.3650 | -0.3337 |
| 3 | 0.7055 | 0.7055 | +0.000000 | YES | 0.5175 | -0.1880 |
| 4 | 0.7032 | — | — | — | — | — |

**BWT_routed (seed 2)** = -0.058676

**BWT_unrouted (seed 2)** = -0.3046

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5648 | -0.120561 | YES | 0.3351 | -0.3503 |
| 1 | 0.6713 | 0.6096 | -0.061711 | YES | 0.3227 | -0.3486 |
| 2 | 0.6970 | 0.6344 | -0.062539 | YES | 0.3691 | -0.3279 |
| 3 | 0.7062 | 0.7062 | +0.000000 | YES | 0.5300 | -0.1762 |
| 4 | 0.6968 | — | — | — | — | — |

**BWT_routed (seed 3)** = -0.061203

**BWT_unrouted (seed 3)** = -0.3008

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | -0.060220 | 0.001105 | ['-0.060780', '-0.058676', '-0.061203'] |
| BWT_unrouted | -0.3015 | 0.0023 | ['-0.2992', '-0.3046', '-0.3008'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

