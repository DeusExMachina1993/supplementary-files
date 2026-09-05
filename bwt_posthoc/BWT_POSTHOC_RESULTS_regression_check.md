# Post-hoc routed BWT results — PatchExpertFFN_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3391 | -0.3463 |
| 1 | 0.6698 | 0.6698 | +0.000000 | YES | 0.3175 | -0.3522 |
| 2 | 0.7035 | 0.7035 | +0.000000 | YES | 0.3705 | -0.3330 |
| 3 | 0.6935 | 0.6935 | +0.000000 | YES | 0.5005 | -0.1930 |
| 4 | 0.6983 | — | — | — | — | — |

**BWT_routed (seed 1)** = +0.000000

**BWT_unrouted (seed 1)** = -0.3061

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3372 | -0.3482 |
| 1 | 0.6676 | 0.6676 | +0.000000 | YES | 0.3185 | -0.3491 |
| 2 | 0.6988 | 0.6988 | +0.000000 | YES | 0.3652 | -0.3335 |
| 3 | 0.6917 | 0.6917 | +0.000000 | YES | 0.5004 | -0.1913 |
| 4 | 0.6978 | — | — | — | — | — |

**BWT_routed (seed 2)** = +0.000000

**BWT_unrouted (seed 2)** = -0.3055

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3324 | -0.3530 |
| 1 | 0.6713 | 0.6713 | +0.000000 | YES | 0.3226 | -0.3488 |
| 2 | 0.6970 | 0.6970 | +0.000000 | YES | 0.3682 | -0.3288 |
| 3 | 0.6962 | 0.6962 | +0.000000 | YES | 0.5048 | -0.1914 |
| 4 | 0.6993 | — | — | — | — | — |

**BWT_routed (seed 3)** = +0.000000

**BWT_unrouted (seed 3)** = -0.3055

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | +0.000000 | 0.000000 | ['+0.000000', '+0.000000', '+0.000000'] |
| BWT_unrouted | -0.3057 | 0.0003 | ['-0.3061', '-0.3055', '-0.3055'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

