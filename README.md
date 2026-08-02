#supplementary files

Per-experience model checkpoints for both arms and all three
seeds of the submitted manuscript, plus the shared patch-0
model. Tensor state dicts only; no optimizer state, no paths,
no environment metadata.

## Naming

`<arm>_seed<N>_exp<M>.pt`, where arm is `Base` (sequential
fine-tuning) or `PatchExpertFFN_ppHead` (the method, with the
per-patch output head), N is the seed (1, 2, 3) and M is the
experience index. Internal experiment-launcher tokens have been
stripped from the filenames; they encoded optimizer handling and
a drift-signal slot that this family does not use.

## Verifying zero forgetting without the dataset

For each seed and each j in {1,2,3}, the tensors named
`patch_experts.<j>.*` and `patch_heads.<j>.*` are bit-identical
between `..._exp<j>.pt` and `..._exp4.pt`. Compare them directly,
or check the SHA-256 digests in `checkpoint_hashes.csv`.
This requires no dataset and no forward pass.
