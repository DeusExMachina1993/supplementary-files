def expert_and_head_tensors(model: PatchExpertFFNBert4Rec, patch_id: int) -> Dict[str, torch.Tensor]:
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


def eval_one(model, ds, label: str) -> float:
    res = evaluate_model(model, ds, DEVICE, k=K, batch_size=EVAL_BATCH_SIZE, label=label)
    return res[f"hit@{K}"]


def run_seed(seed: int, test_datasets: List[NewHeroActionBERTDataset]) -> dict:
    print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")

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
