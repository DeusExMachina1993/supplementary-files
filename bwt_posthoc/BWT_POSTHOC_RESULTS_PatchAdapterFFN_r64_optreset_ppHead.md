# Post-hoc routed BWT results — PatchAdapterFFN_r64_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3391 | -0.3463 |
| 1 | 0.6670 | 0.6670 | +0.000000 | YES | 0.3138 | -0.3532 |
| 2 | 0.6959 | 0.6959 | +0.000000 | YES | 0.3628 | -0.3331 |
| 3 | 0.6926 | 0.6926 | +0.000000 | YES | 0.4969 | -0.1956 |
| 4 | 0.6879 | — | — | — | — | — |

**BWT_routed (seed 1)** = +0.000000

**BWT_unrouted (seed 1)** = -0.3071

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3432 | -0.3422 |
| 1 | 0.6689 | 0.6689 | +0.000000 | YES | 0.3188 | -0.3501 |
| 2 | 0.6952 | 0.6952 | +0.000000 | YES | 0.3609 | -0.3343 |
| 3 | 0.6864 | 0.6864 | +0.000000 | YES | 0.4951 | -0.1913 |
| 4 | 0.6903 | — | — | — | — | — |

**BWT_routed (seed 2)** = +0.000000

**BWT_unrouted (seed 2)** = -0.3045

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3439 | -0.3415 |
| 1 | 0.6717 | 0.6717 | +0.000000 | YES | 0.3176 | -0.3541 |
| 2 | 0.6989 | 0.6989 | +0.000000 | YES | 0.3640 | -0.3348 |
| 3 | 0.6879 | 0.6879 | +0.000000 | YES | 0.4934 | -0.1945 |
| 4 | 0.6889 | — | — | — | — | — |

**BWT_routed (seed 3)** = +0.000000

**BWT_unrouted (seed 3)** = -0.3062

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | +0.000000 | 0.000000 | ['+0.000000', '+0.000000', '+0.000000'] |
| BWT_unrouted | -0.3059 | 0.0011 | ['-0.3071', '-0.3045', '-0.3062'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

