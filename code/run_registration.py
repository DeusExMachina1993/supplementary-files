        # -- PatchExpertFFN: MoE-style per-patch FFN(+norm2) expert library.
        #    Model class swapped via FORWARD_INJECTION_MODELS (PatchExpertFFNBert4Rec).
        #    exp0 trains the whole model (expert 0 == the ordinary backbone FFN);
        #    exp t>=1 copies expert (t-1)->t, freezes everything but expert t, and
        #    routes forward through it. Selection is by patch id, like PerPatchLoRA
        #    ,  it does NOT read a signal folder either. The _optreset variant adds
        #    GlobalOptimizerResetPlugin as a second plugin (same two-plugin
        #    convention as FreezeExceptFFN_optreset ,  independent state, order-
        #    insensitive: PatchExpertFFNPlugin only touches requires_grad/weights,
        #    GlobalOptimizerResetPlugin only touches optimizer.state).
        #
        #    per_patch_head toggle (2026-07-28): the model always pre-allocates a
        #    per-patch output-head library too (head 0 == original final_projection),
        #    but head routing stays pinned at head 0 forever (today's exact
        #    behaviour) UNLESS PatchExpertFFNPlugin is built with
        #    per_patch_head=True. Same keys, no new architecture/checkpoint
        #    change ,  toggled here via the SAME UI/YAML "forward:" passthrough
        #    already used for patch_center/freeze_backbone (set
        #    forward: {per_patch_head: true} in a run_configs.yml preset, or
        #    answer the forward-injection UI prompt, to turn it on). Default
        #    (no _UI_FORWARD_CFG, or key absent) = False = unchanged behaviour.
        #    copy_init toggle (PATCH_EXPERT_METRICS_PLAN.md M4 ablation): same
        #    "forward:" passthrough, forward: {copy_init: false} disables the
        #    warm-start copy (expert/head t keeps its own fresh random init)
        #    to isolate the copy rule's contribution via fwt/hit@10_exp{t}.
        #    Default True = unchanged behaviour. eval_datasets=recovery_eval_datasets
        #    reuses the SAME held-out sets RecoveryTimePlugin already built, so
        #    fwt/hit@10_exp{t} needs no new data plumbing.
        "PatchExpertFFN":                  lambda: (NaiveWithBannedIds, "PatchExpertFFN",
            lambda: [PatchExpertFFNPlugin(
                per_patch_head=bool(_UI_FORWARD_CFG and _UI_FORWARD_CFG.get("per_patch_head")),
                copy_init=bool(_UI_FORWARD_CFG.get("copy_init", True)) if _UI_FORWARD_CFG else True,
                eval_datasets=recovery_eval_datasets)],
            'CL', PLASTICITY_LR),
        "PatchExpertFFN_optreset":         lambda: (NaiveWithBannedIds, "PatchExpertFFN_optreset",
            lambda: [PatchExpertFFNPlugin(
                per_patch_head=bool(_UI_FORWARD_CFG and _UI_FORWARD_CFG.get("per_patch_head")),
                copy_init=bool(_UI_FORWARD_CFG.get("copy_init", True)) if _UI_FORWARD_CFG else True,
                eval_datasets=recovery_eval_datasets),
                     GlobalOptimizerResetPlugin(enabled=True, reset_step=True)],
            'CL', PLASTICITY_LR),
