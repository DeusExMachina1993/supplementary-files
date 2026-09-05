# Post-hoc routed BWT results — PatchExpertFFN_refresh4_optreset_ppHead (winrate signal)

Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. See `TASK_BWT_POSTHOC.md` for the procedure.

**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every mask position deterministically (no RNG at construction), so the test datasets built here are byte-identical every run; any small difference from the W&B-logged diagonal values reflects using the SAME dataset instance across both terms of BWT (as required), not stochastic mask draws.


## Seed 1

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5541 | -0.131292 | YES | 0.3271 | -0.3583 |
| 1 | 0.6698 | 0.5888 | -0.080931 | YES | 0.3159 | -0.3539 |
| 2 | 0.7035 | 0.6128 | -0.090641 | YES | 0.3649 | -0.3385 |
| 3 | 0.6935 | 0.6384 | -0.055154 | YES | 0.5208 | -0.1727 |
| 4 | 0.7189 | — | — | — | — | — |

**BWT_routed (seed 1)** = -0.089504

**BWT_unrouted (seed 1)** = -0.3058

**Weight-equality assertion (seed 1)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 2

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5735 | -0.111857 | YES | 0.3367 | -0.3487 |
| 1 | 0.6676 | 0.6100 | -0.057523 | YES | 0.3263 | -0.3413 |
| 2 | 0.6988 | 0.6264 | -0.072334 | YES | 0.3767 | -0.3221 |
| 3 | 0.6917 | 0.6472 | -0.044497 | YES | 0.5175 | -0.1741 |
| 4 | 0.7131 | — | — | — | — | — |

**BWT_routed (seed 2)** = -0.071553

**BWT_unrouted (seed 2)** = -0.2966

**Weight-equality assertion (seed 2)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)

## Seed 3

| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |
|---|---|---|---|---|---|---|
| 0 | 0.6854 | 0.5686 | -0.116820 | YES | 0.3378 | -0.3476 |
| 1 | 0.6713 | 0.6021 | -0.069261 | YES | 0.3193 | -0.3520 |
| 2 | 0.6970 | 0.6153 | -0.081701 | YES | 0.3659 | -0.3311 |
| 3 | 0.6962 | 0.6405 | -0.055756 | YES | 0.5195 | -0.1767 |
| 4 | 0.7209 | — | — | — | — | — |

**BWT_routed (seed 3)** = -0.080885

**BWT_unrouted (seed 3)** = -0.3018

**Weight-equality assertion (seed 3)**: ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)


## Aggregate over seeds

| Metric | mean | stdev | values |
|---|---|---|---|
| BWT_routed | -0.080647 | 0.007331 | ['-0.089504', '-0.071553', '-0.080885'] |
| BWT_unrouted | -0.3014 | 0.0038 | ['-0.3058', '-0.2966', '-0.3018'] |

**Cross-seed weight-equality assertion**: ALL PASSED across all seeds and j=0..3

Expectation check: BWT_routed should be ~0 (by construction — routing makes old-patch scoring insensitive to later training), and BWT_unrouted should be close to the previously logged W&B value (~-0.29), cross-validating the original artifact diagnosis.

