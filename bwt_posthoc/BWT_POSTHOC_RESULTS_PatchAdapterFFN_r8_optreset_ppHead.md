# Post-hoc routed BWT results — PatchAdapterFFN_r8_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3394 | -0.3460 |
| 1 | 0.6606 | 0.6606 | +0.000000 | YES | 0.3088 | -0.3517 |
| 2 | 0.6799 | 0.6799 | +0.000000 | YES | 0.3592 | -0.3207 |
| 3 | 0.6711 | 0.6711 | +0.000000 | YES | 0.4658 | -0.2053 |
| 4 | 0.6589 | — | — | — | — | — |

**BWT_routed (seed 1)** = +0.000000

**BWT_unrouted (seed 1)** = -0.3059

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3436 | -0.3418 |
| 1 | 0.6579 | 0.6579 | +0.000000 | YES | 0.3175 | -0.3404 |
| 2 | 0.6767 | 0.6767 | +0.000000 | YES | 0.3599 | -0.3168 |
| 3 | 0.6700 | 0.6700 | +0.000000 | YES | 0.4610 | -0.2089 |
| 4 | 0.6562 | — | — | — | — | — |

**BWT_routed (seed 2)** = +0.000000

**BWT_unrouted (seed 2)** = -0.3020

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3446 | -0.3408 |
| 1 | 0.6629 | 0.6629 | +0.000000 | YES | 0.3129 | -0.3500 |
| 2 | 0.6800 | 0.6800 | +0.000000 | YES | 0.3561 | -0.3239 |
| 3 | 0.6725 | 0.6725 | +0.000000 | YES | 0.4701 | -0.2024 |
| 4 | 0.6572 | — | — | — | — | — |

**BWT_routed (seed 3)** = +0.000000

**BWT_unrouted (seed 3)** = -0.3043

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | +0.000000 | 0.000000 | ['+0.000000', '+0.000000', '+0.000000'] |
| BWT_unrouted | -0.3041 | 0.0016 | ['-0.3059', '-0.3020', '-0.3043'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

