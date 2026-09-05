# Post-hoc routed BWT results — PatchAdapterFFN_r32_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3390 | -0.3464 |
| 1 | 0.6671 | 0.6671 | +0.000000 | YES | 0.3163 | -0.3508 |
| 2 | 0.6910 | 0.6910 | +0.000000 | YES | 0.3613 | -0.3296 |
| 3 | 0.6800 | 0.6800 | +0.000000 | YES | 0.4848 | -0.1952 |
| 4 | 0.6803 | — | — | — | — | — |

**BWT_routed (seed 1)** = +0.000000

**BWT_unrouted (seed 1)** = -0.3055

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3409 | -0.3445 |
| 1 | 0.6636 | 0.6636 | +0.000000 | YES | 0.3154 | -0.3482 |
| 2 | 0.6941 | 0.6941 | +0.000000 | YES | 0.3658 | -0.3282 |
| 3 | 0.6844 | 0.6844 | +0.000000 | YES | 0.4896 | -0.1948 |
| 4 | 0.6818 | — | — | — | — | — |

**BWT_routed (seed 2)** = +0.000000

**BWT_unrouted (seed 2)** = -0.3039

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.6854 | +0.000000 | YES | 0.3369 | -0.3485 |
| 1 | 0.6667 | 0.6667 | +0.000000 | YES | 0.3084 | -0.3583 |
| 2 | 0.6931 | 0.6931 | +0.000000 | YES | 0.3589 | -0.3342 |
| 3 | 0.6912 | 0.6912 | +0.000000 | YES | 0.4945 | -0.1967 |
| 4 | 0.6818 | — | — | — | — | — |

**BWT_routed (seed 3)** = +0.000000

**BWT_unrouted (seed 3)** = -0.3094

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | +0.000000 | 0.000000 | ['+0.000000', '+0.000000', '+0.000000'] |
| BWT_unrouted | -0.3063 | 0.0023 | ['-0.3055', '-0.3039', '-0.3094'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

