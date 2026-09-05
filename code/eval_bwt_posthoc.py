"""
Post-hoc routed BWT from saved per_exp checkpoints (no retraining).

See TASK_BWT_POSTHOC.md for the full spec. This is a NEW standalone script;
it does not modify any existing module (backward-compat rule in CLAUDE.md).

What this computes, for PatchExpertFFN_optreset_ppHead (winrate signal):

  1. HR_diag[j]         : hit@10 on patch j's held-out test set, evaluated
                          through expert j / head j, loaded from the
                          checkpoint saved right after patch j finished
                          training (exp0 file for j=0, per_exp exp{j} for
                          j=1..4).
  2. HR_final_routed[j] : hit@10 on patch j's test set (j=0..3), evaluated
                          through expert j / head j, but loaded from the
                          FINAL (exp4) checkpoint. Because every checkpoint
                          allocates and freezes every expert/head slot,
                          expert j / head j inside the exp4 checkpoint are
                          bit-identical to the same slots inside the exp_j
                          checkpoint -- this is verified directly, not
                          assumed.
  3. BWT_routed         = mean_j (HR_final_routed[j] - HR_diag[j]), j=0..3.
                          Expected to be EXACTLY 0 given (2).
  4. HR_unrouted[j]     : hit@10 on patch j's test set (j=0..3), evaluated
                          through the exp4 checkpoint with active_patch=4,
                          active_head=4 (i.e. what the runner's retention
                          loop actually measured -- it never called
                          set_active_patch/set_active_head before eval).
  5. BWT_unrouted       = mean_j (HR_unrouted[j] - HR_diag[j]), j=0..3.
                          Expected to reproduce the logged W&B BWT (~-0.29).

Run:
    python eval_bwt_posthoc.py

Requires torch + pandas (avalanche is stubbed out below if not installed --
it is only ever imported transitively by cl_localize.utils for a function
this script never calls).
"""

from __future__ import annotations

import os
import re
import sys
import types
import statistics
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# Optional avalanche stub. cl_localize.utils does
#   `from avalanche.benchmarks.utils import make_avalanche_dataset`
# at module import time (needed elsewhere in the repo, never called by this
# script). If avalanche isn't installed in this environment, inject a stub so
# cl_localize.utils / cl_localize.metrics.causal_eval can still be imported
# unmodified. If avalanche IS installed, this block is a no-op.
# ---------------------------------------------------------------------------
try:
    import avalanche  # noqa: F401
except ImportError:
    _av = types.ModuleType("avalanche")
    _av_bm = types.ModuleType("avalanche.benchmarks")
    _av_bm_utils = types.ModuleType("avalanche.benchmarks.utils")

    def _stub_make_avalanche_dataset(*_a, **_k):
        raise RuntimeError(
            "avalanche stub: make_avalanche_dataset was called, but "
            "eval_bwt_posthoc.py should never need it."
        )

    _av_bm_utils.make_avalanche_dataset = _stub_make_avalanche_dataset
    _av_bm.utils = _av_bm_utils
    _av.benchmarks = _av_bm
    sys.modules["avalanche"] = _av
    sys.modules["avalanche.benchmarks"] = _av_bm
    sys.modules["avalanche.benchmarks.utils"] = _av_bm_utils

# ---------------------------------------------------------------------------
# argv guard. MUST stay above the cl_localize imports.
#
# cl_localize/config.py calls argparse.parse_args() at MODULE IMPORT time to
# set MODEL_TYPE. Its parser knows exactly one flag, --model, with choices
# {sasrec, bert4rec, gbert}. So merely importing config makes that parser read
# THIS script's command line and abort the process:
#     error: unrecognized arguments: --out ... / --run-name ...
#     error: argument --model: invalid choice: 'adapter'
# before main() is ever reached.
#
# Fix: stash our argv, hand config.py an empty one, and have main() parse from
# the stash. config.py then falls back to its own default, which is inert here
# -- MODEL_TYPE is never read by this script nor by any module it imports; this
# script selects its model class explicitly via build_model(). Editing
# config.py instead would change every training entry point, which the
# backward-compat rule in CLAUDE.md forbids.
# ---------------------------------------------------------------------------
_OUR_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

from cl_localize.config import (
    NUM_ITEMS, SEQ_LEN, D_MODEL, N_HEAD, N_LAYERS, DROPOUT, MASK_TOKEN_ID,
    DATASET_DIR, device as CONFIG_DEVICE,
)
from cl_localize.models.patch_expert_ffn_bert4rec import PatchExpertFFNBert4Rec
from cl_localize.datasets.new_hero_action_bert_dataset import NewHeroActionBERTDataset
from cl_localize.metrics.causal_eval import evaluate_model
from cl_localize.utils import (
    load_inter_file,
    build_shared_hero_vocab,
    build_shared_match_id_map,
    split_sequences,
)

# ============================================================
# Fixed run parameters (mirrors the runner's main() / strategy config)
# ============================================================
SEEDS = [1, 2, 3]
NUM_PATCHES = 5
PATCH_VERSIONS = [54, 55, 56, 57, 58]
K = 10
EVAL_BATCH_SIZE = 1024

# Defaults preserve the original single-purpose behaviour; every value is
# overridable from the CLI (revision 2026-09-01, R1.5/R2.3/R2.4 analyses).
RUN_NAME = "PatchExpertFFN_optreset_ppHead"
SIGNAL_TAG = "winrate"
# E3 refresh arms branch mid-stream, so their per_exp files only exist from
# the refresh patch on. Pre-refresh checkpoints are bit-identical to the
# source arm's by construction; --fallback-run-name names that source arm
# and is consulted ONLY when the primary file is absent.
FALLBACK_RUN_NAME = None
# "expert" = PatchExpertFFNBert4Rec (default). "adapter" = the E2d
# PatchAdapterFFNBert4Rec (per-patch LoRA on the shared frozen FFN); needs
# --lora-rank to match the arm being analysed.
MODEL_KIND = "expert"
LORA_RANK = 8

CKPT_DIR = os.path.join("checkpoints", "per_exp")
EXP0_CKPT = os.path.join("checkpoints", "base_exp0__PatchExpertFFNBert4Rec.pt")


def per_exp_ckpt(seed: int, exp: int, run_name: str = None) -> str:
    rn = run_name or RUN_NAME
    return os.path.join(CKPT_DIR, f"{SIGNAL_TAG}_{rn}_seed{seed}_exp{exp}.pt")


def ckpt_for(seed: int, exp: int) -> str:
    if exp == 0:
        return EXP0_CKPT
    path = per_exp_ckpt(seed, exp)
    if not os.path.exists(path) and FALLBACK_RUN_NAME:
        fb = per_exp_ckpt(seed, exp, FALLBACK_RUN_NAME)
        if os.path.exists(fb):
            print(f"[ckpt] '{os.path.basename(path)}' absent; using fallback "
                  f"'{os.path.basename(fb)}' (pre-branch checkpoints are "
                  f"bit-identical by construction)")
            return fb
    return path


DEVICE = torch.device(CONFIG_DEVICE)


# ============================================================
# 1. Build the 5 test datasets ONCE (shared hero vocab / match-id map, same
#    train/valid/test split logic as main()). Reused for every seed and
#    every checkpoint load so both terms of BWT see the identical dataset
#    instance (masks/banned-ids constructed once).
# ============================================================
def build_test_datasets():
    files = [
        os.path.join(DATASET_DIR, f"patch_{v}", f"patch_{v}.inter")
        for v in PATCH_VERSIONS
    ]
    hero_vocab = build_shared_hero_vocab(files)
    match_map = build_shared_match_id_map(files)

    test_datasets = []
    for v in PATCH_VERSIONS:
        f = os.path.join(DATASET_DIR, f"patch_{v}", f"patch_{v}.inter")
        seq = load_inter_file(f, hero_action_vocab=hero_vocab, match_id_map=match_map)
        _train, _valid, test_data = split_sequences(seq, dataset_size=1.0)
        ds = NewHeroActionBERTDataset(test_data, mask_token_id=MASK_TOKEN_ID)
        test_datasets.append(ds)
        print(f"[data] patch_{v} test set: {len(ds)} masked samples")
    return test_datasets


# ============================================================
# 2. Model construction + checkpoint loading
# ============================================================
def build_model() -> PatchExpertFFNBert4Rec:
    kwargs = dict(
        num_items=NUM_ITEMS, max_seq_len=SEQ_LEN,
        d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, dropout=DROPOUT,
        num_patches=NUM_PATCHES,
    )
    if MODEL_KIND == "adapter":
        from cl_localize.models.patch_adapter_ffn_bert4rec import (
            PatchAdapterFFNBert4Rec)
        model = PatchAdapterFFNBert4Rec(lora_rank=LORA_RANK, **kwargs).to(DEVICE)
    else:
        model = PatchExpertFFNBert4Rec(**kwargs).to(DEVICE)
    model.eval()
    return model


def strip_and_filter_state_dict(raw_sd: Dict[str, torch.Tensor], prefix: str = "base_model.") -> Dict[str, torch.Tensor]:
    """The saved checkpoint is `wrapped_model.state_dict()` where wrapped_model
    is `DomainIncrementalBERT4Rec(base_transformer_model, ...)`. Its keys are
    therefore `base_model.<...>` (the PatchExpertFFNBert4Rec params we want)
    plus `classifier.<...>` (the wrapper's unused fixed-head classifier --
    never read in forward(), not part of PatchExpertFFNBert4Rec).

    Strips the `base_model.` prefix and drops everything else. If NO key has
    the prefix, assumes the checkpoint was already saved unwrapped and
    returns it unchanged.
    """
    has_prefix = any(k.startswith(prefix) for k in raw_sd.keys())
    if not has_prefix:
        return dict(raw_sd)
    return {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}


def load_checkpoint(model: PatchExpertFFNBert4Rec, ckpt_path: str) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    raw_sd = ckpt["model_state_dict"]
    sd = strip_and_filter_state_dict(raw_sd)
    strict = not (MODEL_KIND == "adapter" and ckpt_path == EXP0_CKPT)
    try:
        result = model.load_state_dict(sd, strict=strict)
        if not strict:
            # Adapter arms borrow the expert arm's exp0 checkpoint; the
            # adapter's own lora/head-slot keys are legitimately absent
            # there, and the expert checkpoint's pre-allocated slots
            # patch_experts.{k>=1}.* have no counterpart in the adapter
            # model (its slots k>=1 are LoRA deltas over slot 0). This is
            # the same contract the training runner asserts (see the
            # PatchAdapterFFNBert4Rec docstring). Anything ELSE missing or
            # unexpected is fatal.
            _spare_slot = re.compile(r"\.patch_experts\.([1-9]\d*)\.")
            unexpected = [k for k in result.unexpected_keys
                          if not _spare_slot.search(k)]
            n_spare = len(result.unexpected_keys) - len(unexpected)
            bad_missing = [k for k in result.missing_keys
                           if "lora" not in k and "head" not in k]
            if unexpected or bad_missing:
                raise RuntimeError(
                    f"non-adapter key diff on borrowed exp0 load: "
                    f"unexpected={unexpected[:5]} missing={bad_missing[:5]}")
            print(f"  [exp0 borrow] loaded strict=False: "
                  f"{len(result.missing_keys)} adapter-only keys left at init, "
                  f"{n_spare} unused expert-slot tensors (patch_experts.k>=1) ignored")
    except RuntimeError as e:
        print(f"\n[FATAL] load_state_dict FAILED for '{ckpt_path}'.")
        print("Key diff:")
        print(str(e))
        raise
    return ckpt


def expert_and_head_tensors(model: PatchExpertFFNBert4Rec, patch_id: int) -> Dict[str, torch.Tensor]:
    """Pull out expert `patch_id`'s FFN+norm2 weights (every layer) and head
    `patch_id`'s weight+bias, keyed by their state_dict name, detached clones
    (so later loads into the same model object can't mutate them)."""
    sd = model.state_dict()
    out = {}
    for L in range(N_LAYERS):
        prefix = f"patch_expert_layers.{L}.patch_experts.{patch_id}."
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k] = v.detach().clone()
    head_prefix = f"patch_heads.{patch_id}."
    for k, v in sd.items():
        if k.startswith(head_prefix):
            out[k] = v.detach().clone()
    return out


def tensors_equal(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    assert set(a.keys()) == set(b.keys()), (
        f"Tensor key sets differ:\n  only in a: {set(a) - set(b)}\n"
        f"  only in b: {set(b) - set(a)}"
    )
    return {k: bool(torch.equal(a[k], b[k])) for k in a}


# ============================================================
# 3. Per-seed procedure
# ============================================================
def eval_one(model, ds, label: str) -> float:
    res = evaluate_model(model, ds, DEVICE, k=K, batch_size=EVAL_BATCH_SIZE, label=label)
    return res[f"hit@{K}"]


def run_seed(seed: int, test_datasets: List[NewHeroActionBERTDataset]) -> dict:
    print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")

    # ---- Step 3: diagonal ----
    hr_diag = [None] * NUM_PATCHES
    diag_expert_head_tensors = {}
    for j in range(NUM_PATCHES):
        m = build_model()
        ckpt_path = ckpt_for(seed, j)
        print(f"[diag] loading {ckpt_path} for patch {j}")
        load_checkpoint(m, ckpt_path)
        m.set_active_patch(j)
        m.set_active_head(j)
        hr_diag[j] = eval_one(m, test_datasets[j], label=f"diag_seed{seed}_exp{j}")
        diag_expert_head_tensors[j] = expert_and_head_tensors(m, j)
        print(f"  HR_diag[{j}] = {hr_diag[j]:.4f}")
        del m

    # ---- Step 4: routed final (load exp4 checkpoint once) ----
    model_final = build_model()
    exp4_path = ckpt_for(seed, NUM_PATCHES - 1)
    print(f"[routed-final] loading {exp4_path}")
    load_checkpoint(model_final, exp4_path)

    hr_final_routed = [None] * (NUM_PATCHES - 1)
    final_expert_head_tensors = {}
    for j in range(NUM_PATCHES - 1):
        model_final.set_active_patch(j)
        model_final.set_active_head(j)
        hr_final_routed[j] = eval_one(model_final, test_datasets[j], label=f"routed_final_seed{seed}_exp{j}")
        final_expert_head_tensors[j] = expert_and_head_tensors(model_final, j)
        print(f"  HR_final_routed[{j}] = {hr_final_routed[j]:.4f}")

    # ---- Step 5: weight-equality assertion + routed BWT ----
    weight_equal = {}
    routed_delta = [None] * (NUM_PATCHES - 1)
    for j in range(NUM_PATCHES - 1):
        eq_map = tensors_equal(diag_expert_head_tensors[j], final_expert_head_tensors[j])
        all_equal = all(eq_map.values())
        weight_equal[j] = all_equal
        routed_delta[j] = hr_final_routed[j] - hr_diag[j]
        if not all_equal:
            bad_keys = [k for k, v in eq_map.items() if not v]
            print(f"  [WARNING] expert/head {j}: weight mismatch on keys: {bad_keys}")
        else:
            print(f"  [OK] expert/head {j}: bit-identical between exp{j} and exp4 checkpoints")

    bwt_routed = sum(routed_delta) / len(routed_delta)

    # ---- Step 6: unrouted ablation (reuse model_final; force patch/head=4) ----
    hr_unrouted = [None] * (NUM_PATCHES - 1)
    unrouted_delta = [None] * (NUM_PATCHES - 1)
    model_final.set_active_patch(NUM_PATCHES - 1)
    model_final.set_active_head(NUM_PATCHES - 1)
    for j in range(NUM_PATCHES - 1):
        hr_unrouted[j] = eval_one(model_final, test_datasets[j], label=f"unrouted_seed{seed}_exp{j}")
        unrouted_delta[j] = hr_unrouted[j] - hr_diag[j]
        print(f"  HR_unrouted[{j}] (routed through exp4 expert/head) = {hr_unrouted[j]:.4f}"
              f"  (delta {unrouted_delta[j]:+.4f})")

    bwt_unrouted = sum(unrouted_delta) / len(unrouted_delta)

    del model_final

    return {
        "seed": seed,
        "hr_diag": hr_diag,
        "hr_final_routed": hr_final_routed,
        "routed_delta": routed_delta,
        "bwt_routed": bwt_routed,
        "weight_equal": weight_equal,
        "hr_unrouted": hr_unrouted,
        "unrouted_delta": unrouted_delta,
        "bwt_unrouted": bwt_unrouted,
    }


# ============================================================
# 4. Output
# ============================================================
def write_report(all_results: List[dict], out_path: str = None):
    if out_path is None:
        out_path = (f"BWT_POSTHOC_RESULTS_{RUN_NAME}.md"
                    if RUN_NAME != "PatchExpertFFN_optreset_ppHead"
                    else "BWT_POSTHOC_RESULTS.md")
    lines = []
    lines.append(f"# Post-hoc routed BWT results — {RUN_NAME} ({SIGNAL_TAG} signal)\n")
    lines.append(
        "Computed from saved `checkpoints/per_exp/` checkpoints, no retraining. "
        "See `TASK_BWT_POSTHOC.md` for the procedure.\n"
    )
    lines.append(
        "**Note on absolute HRs:** `NewHeroActionBERTDataset` enumerates every "
        "mask position deterministically (no RNG at construction), so the test "
        "datasets built here are byte-identical every run; any small difference "
        "from the W&B-logged diagonal values reflects using the SAME dataset "
        "instance across both terms of BWT (as required), not stochastic "
        "mask draws.\n"
    )

    for r in all_results:
        s = r["seed"]
        lines.append(f"\n## Seed {s}\n")
        lines.append("| Patch j | HR_diag[j] | HR_final_routed[j] | routed delta | weight-equal | HR_unrouted[j] | unrouted delta |")
        lines.append("|---|---|---|---|---|---|---|")
        for j in range(NUM_PATCHES):
            diag = r["hr_diag"][j]
            if j < NUM_PATCHES - 1:
                fr = r["hr_final_routed"][j]
                rd = r["routed_delta"][j]
                we = r["weight_equal"][j]
                un = r["hr_unrouted"][j]
                ud = r["unrouted_delta"][j]
                lines.append(
                    f"| {j} | {diag:.4f} | {fr:.4f} | {rd:+.6f} | "
                    f"{'YES' if we else 'NO — MISMATCH'} | {un:.4f} | {ud:+.4f} |"
                )
            else:
                lines.append(f"| {j} | {diag:.4f} | — | — | — | — | — |")

        lines.append(f"\n**BWT_routed (seed {s})** = {r['bwt_routed']:+.6f}")
        lines.append(f"\n**BWT_unrouted (seed {s})** = {r['bwt_unrouted']:+.4f}")
        all_weights_ok = all(r["weight_equal"].values())
        lines.append(
            f"\n**Weight-equality assertion (seed {s})**: "
            f"{'ALL PASSED (expert j / head j bit-identical between exp_j and exp4 checkpoints for all j=0..3)' if all_weights_ok else 'FAILED — see WARNING lines above / console log'}"
        )

    # ---- Aggregate across seeds ----
    bwt_routed_vals = [r["bwt_routed"] for r in all_results]
    bwt_unrouted_vals = [r["bwt_unrouted"] for r in all_results]
    lines.append("\n\n## Aggregate over seeds\n")
    lines.append("| Metric | mean | stdev | values |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| BWT_routed | {statistics.mean(bwt_routed_vals):+.6f} | "
        f"{statistics.pstdev(bwt_routed_vals):.6f} | "
        f"{[f'{v:+.6f}' for v in bwt_routed_vals]} |"
    )
    lines.append(
        f"| BWT_unrouted | {statistics.mean(bwt_unrouted_vals):+.4f} | "
        f"{statistics.pstdev(bwt_unrouted_vals):.4f} | "
        f"{[f'{v:+.4f}' for v in bwt_unrouted_vals]} |"
    )

    all_ok = all(all(r["weight_equal"].values()) for r in all_results)
    lines.append(
        f"\n**Cross-seed weight-equality assertion**: "
        f"{'ALL PASSED across all seeds and j=0..3' if all_ok else 'FAILED for at least one (seed, j) — see per-seed sections above'}"
    )
    lines.append(
        "\nExpectation check: BWT_routed should be ~0 (by construction — routing "
        "makes old-patch scoring insensitive to later training), and BWT_unrouted "
        "should be close to the previously logged W&B value (~-0.29), "
        "cross-validating the original artifact diagnosis.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[write_report] wrote {out_path}")


def inspect_checkpoint_keys(ckpt_path: str) -> None:
    """One-time diagnostic (task step 2): print key-prefix structure of a
    checkpoint before doing any loading, so a bad assumption about
    base_model./classifier. prefixes surfaces immediately."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    prefixes = sorted({k.split(".")[0] for k in sd.keys()})
    print(f"[inspect] {ckpt_path}")
    print(f"  top-level ckpt keys: {list(ckpt.keys())}")
    print(f"  model_state_dict top-level prefixes: {prefixes}")
    n_base = sum(1 for k in sd if k.startswith("base_model."))
    n_other = len(sd) - n_base
    print(f"  {len(sd)} total tensors -> {n_base} under 'base_model.', {n_other} other")


def main():
    # This `global` must precede the argparse block, because the `default=`
    # expressions below read RUN_NAME / SIGNAL_TAG / LORA_RANK. Python rejects
    # a `global` that follows any use of the name in the same scope, and does
    # so at compile time ("name 'RUN_NAME' is used prior to global
    # declaration"), so the module will not even import if this moves back down.
    global RUN_NAME, SIGNAL_TAG, FALLBACK_RUN_NAME, MODEL_KIND, LORA_RANK

    import argparse
    ap = argparse.ArgumentParser(
        description="Post-hoc ROUTED BWT from saved per_exp checkpoints. "
                    "The runner's inline retention/hit@10_* logs are "
                    "UNROUTED (newest expert on old data) -- use this "
                    "script for any routed arm's retention numbers.")
    ap.add_argument("--run-name", default=RUN_NAME,
                    help="strategy/run name in the checkpoint stem "
                         "(default: %(default)s)")
    ap.add_argument("--signal-tag", default=SIGNAL_TAG,
                    help="signal tag prefix of the checkpoint files "
                         "(default: %(default)s)")
    ap.add_argument("--fallback-run-name", default=None,
                    help="E3 branched arms: source arm whose per_exp file is "
                         "used when the primary is absent (pre-branch "
                         "checkpoints are bit-identical). E.g. "
                         "PatchExpertFFN_optreset_ppHead")
    ap.add_argument("--model", choices=["expert", "adapter"], default="expert",
                    help="model class: expert=PatchExpertFFNBert4Rec, "
                         "adapter=E2d PatchAdapterFFNBert4Rec")
    ap.add_argument("--lora-rank", type=int, default=8,
                    help="adapter arms only: LoRA rank of the arm being "
                         "analysed (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="report path (default: BWT_POSTHOC_RESULTS_<run>.md)")
    # Parse the argv stashed before cl_localize.config was imported (see the
    # argv guard near the top), not sys.argv, which was emptied there.
    args = ap.parse_args(_OUR_ARGV)

    RUN_NAME = args.run_name
    SIGNAL_TAG = args.signal_tag
    FALLBACK_RUN_NAME = args.fallback_run_name
    MODEL_KIND = args.model
    LORA_RANK = args.lora_rank

    print(f"[run] {RUN_NAME} (signal={SIGNAL_TAG}, model={MODEL_KIND}"
          + (f", r={LORA_RANK}" if MODEL_KIND == "adapter" else "")
          + (f", fallback={FALLBACK_RUN_NAME}" if FALLBACK_RUN_NAME else "")
          + ")")
    print(f"[device] {DEVICE}")
    inspect_checkpoint_keys(EXP0_CKPT)
    test_datasets = build_test_datasets()

    all_results = []
    for seed in SEEDS:
        all_results.append(run_seed(seed, test_datasets))

    write_report(all_results, out_path=args.out)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"seed {r['seed']}: BWT_routed={r['bwt_routed']:+.6f}  "
              f"BWT_unrouted={r['bwt_unrouted']:+.4f}  "
              f"weights_ok={all(r['weight_equal'].values())}")


if __name__ == "__main__":
    main()
