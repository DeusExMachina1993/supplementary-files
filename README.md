## supplementary files

Per-experience model checkpoints, their parameter-level SHA-256 digests,
the routed backward-transfer verification outputs, and the per-step
recovery records behind the manuscript's figures and restoration tables.
Tensor state dicts only; no optimizer state, no paths, no environment
metadata.

## Contents

- `checkpoints/` -- `<arm>_seed<N>_exp<M>.pt`, N in {1,2,3}, M the
  experience index, plus the shared patch-0 model
  `base_exp0__PatchExpertFFNBert4Rec.pt` (trained once, shared by every arm).
  Arms:
  - `Base` -- sequential fine-tuning, exp1-4
  - `PatchExpertFFN_ppHead` -- the routed arm with per-patch head, exp1-4
  - `PatchAdapterFFN_r8_ppHead`, `PatchAdapterFFN_r32_ppHead`,
    `PatchAdapterFFN_r64_ppHead` -- per-patch LoRA adapters on the same
    frozen trunk and routing, exp1-4
  - `PatchExpertFFN_refresh2_ppHead`, `..._refresh3_...`, `..._refresh4_...`
    -- the routed arm with a trunk refresh at that experience; files exist
    from the refresh experience onward, the earlier experiences being
    bit-identical to `PatchExpertFFN_ppHead`'s (the arm branches from its
    saved state).
  Internal experiment-launcher tokens have been stripped from the filenames.
- `checkpoint_hashes.csv` -- one row per parameter tensor of every
  checkpoint: file, arm, seed, exp, param, shape, sha256 (SHA-256 of the
  tensor's raw bytes in row-major order, native dtype, i.e.
  `hashlib.sha256(tensor.numpy().tobytes()).hexdigest()`; see
  `verification_log.txt` for the rule actually applied).
- `bwt_posthoc/` -- the routed / unrouted backward-transfer reports
  (Online Resource 1, S4 and S9) as produced by `code/eval_bwt_posthoc.py`.
- `curves/` -- per-step held-out HR@10 records for every arm of the
  revision (baselines, adapters, refresh, shared-weight reference arms and
  the core arms), behind the recovery figure and the restoration tables;
  `recovery_curves_all_seeds.csv` is the core-arm file of the first
  submission.
- `code/` -- the routed-BWT evaluation script, the PatchExpertFFN plugin
  and the strategy registration.
- `verification_log.txt` -- output of the deposit-preparation run,
  including the bit-identity checks below.

## Verifying zero forgetting without the dataset

For each routed arm, seed N and each j in {1,2,3}, every tensor whose name
carries the patch index j (`...patch_experts.<j>.` and `patch_heads.<j>.`)
is bit-identical between `<arm>_seed<N>_exp<j>.pt` and
`<arm>_seed<N>_exp4.pt`. For a refresh arm with refresh experience R, the
exp<j> file for j < R is `PatchExpertFFN_ppHead_seed<N>_exp<j>.pt`. Compare
the tensors directly, or compare the SHA-256 rows in
`checkpoint_hashes.csv`. This requires no dataset and no forward pass.
For the refresh arms the experts are identical but the trunk is not, which
is the "superseded trunk discarded" retention cost reported in the
manuscript's refresh table.
