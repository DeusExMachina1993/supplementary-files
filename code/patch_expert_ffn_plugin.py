from avalanche.training.plugins import SupervisedPlugin


class PatchExpertFFNPlugin(SupervisedPlugin):
    def __init__(self, verbose: bool = True, per_patch_head: bool = False,
                copy_init: bool = True, eval_datasets=None, k: int = 10,
                batch_size: int = 1024):
        super().__init__()
        self.verbose = verbose
        self.per_patch_head = per_patch_head
        self.copy_init = copy_init
        self.eval_datasets = eval_datasets or {}
        self.k = k
        self.batch_size = batch_size

    def _resolve_model(self, strategy):
        model = getattr(strategy, "model", None)
        if model is None:
            return None
        if hasattr(model, "set_active_patch") and hasattr(model, "copy_expert_weights"):
            return model
        base = getattr(model, "base_model", None)
        if base is not None and hasattr(base, "set_active_patch") and hasattr(base, "copy_expert_weights"):
            return base
        return None

    def _wandb_log(self, payload: dict) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(payload)

    def _wandb_summary(self, payload: dict) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            for key, value in payload.items():
                wandb.run.summary[key] = value

    def _log_forward_transfer(self, strategy, m, t: int) -> None:
        ds = self.eval_datasets.get(t)
        if ds is None:
            return
        m.set_active_patch(t - 1)
        if self.per_patch_head:
            m.set_active_head(t - 1)
        from cl_localize.metrics.causal_eval import evaluate_model
        res = evaluate_model(
            strategy.model, ds, strategy.device,
            k=self.k, batch_size=self.batch_size,
            label=f"fwt_exp{t}",
        )
        hr = res.get(f"hit@{self.k}", float("nan"))
        self._wandb_log({f"fwt/hit@{self.k}_exp{t}": hr})
        if self.verbose:
            print(f"[PatchExpertFFN] FWT exp {t}: hit@{self.k}={hr:.4f} "
                f"(patch {t-1}'s state, pre-copy/pre-training).")

    def _log_param_counts(self, m, t: int) -> None:
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        stored = sum(p.numel() for layer in m.patch_expert_layers
                    for p in layer.patch_experts[t].parameters())
        if self.per_patch_head:
            stored += sum(p.numel() for p in m.patch_heads[t].parameters())
        self._wandb_summary({
            "params/trainable_per_patch": trainable,
            "params/stored_per_patch": stored,
        })
        if self.verbose:
            print(f"[PatchExpertFFN] params: trainable_per_patch={trainable:,} "
                f"stored_per_patch={stored:,}")

    def before_training_exp(self, strategy, *args, **kwargs):
        m = self._resolve_model(strategy)
        if m is None:
            return
        t = strategy.experience.current_experience
        if t == 0:
            m.set_active_patch(0)
            m.set_active_head(0)
            if self.verbose:
                print("[PatchExpertFFN] train Exp 0: full model "
                    "(expert 0 == backbone FFN, head 0 == backbone final_projection).")
        else:
            self._log_forward_transfer(strategy, m, t)

            if self.copy_init:
                m.copy_expert_weights(src=t - 1, dst=t)
            m.freeze_all_except_experts()
            m.set_trainable_expert(t)
            m.set_active_patch(t)
            if self.per_patch_head:
                if self.copy_init:
                    m.copy_head_weights(src=t - 1, dst=t)
                m.set_trainable_head(t)
                m.set_active_head(t)

            self._log_param_counts(m, t)

            if self.verbose:
                head_msg = f", copied head {t-1} -> {t} (trainable)" if self.per_patch_head else " (head pinned at 0)"
                copy_msg = "" if self.copy_init else " [copy_init=False: fresh random init, no warm start]"
                print(f"[PatchExpertFFN] train Exp {t}: copied expert {t-1} -> {t}, "
                    f"froze everything but expert {t}{head_msg}{copy_msg}.")

    def after_training_exp(self, strategy, *args, **kwargs):
        m = self._resolve_model(strategy)
        if m is None:
            return
        t = strategy.experience.current_experience
        if t == 0:
            return

        payload = {}
        total_sq = 0.0
        for i, layer in enumerate(m.patch_expert_layers):
            cur = layer.patch_experts[t]
            prev = layer.patch_experts[t - 1]
            sq = 0.0
            for (_, p_cur), (_, p_prev) in zip(cur.named_parameters(), prev.named_parameters()):
                diff = p_cur.detach() - p_prev.detach()
                sq += float((diff * diff).sum())
            l2 = sq ** 0.5
            payload[f"expert_diff/l{i}_l2"] = l2
            total_sq += sq
        payload["expert_diff/total_l2"] = total_sq ** 0.5
        payload["expert_diff/exp"] = t

        if self.per_patch_head:
            h_cur, h_prev = m.patch_heads[t], m.patch_heads[t - 1]
            hsq = float(((h_cur.weight.detach() - h_prev.weight.detach()) ** 2).sum())
            hsq += float(((h_cur.bias.detach() - h_prev.bias.detach()) ** 2).sum())
            payload["head_diff/l2"] = hsq ** 0.5

        self._wandb_log(payload)
        if self.verbose:
            print(f"[PatchExpertFFN] exp {t} drift vs expert {t-1}: "
                f"total_l2={payload['expert_diff/total_l2']:.4f}"
                + (f", head_l2={payload.get('head_diff/l2', float('nan')):.4f}" if self.per_patch_head else ""))

    def before_eval_exp(self, strategy, *args, **kwargs):
        m = self._resolve_model(strategy)
        if m is None:
            return
        t = strategy.experience.current_experience
        m.set_active_patch(t)
        if self.per_patch_head:
            m.set_active_head(t)
        if self.verbose:
            head_tag = f", head {t}" if self.per_patch_head else ", head 0"
            print(f"[PatchExpertFFN] eval Exp {t}: expert {t}{head_tag}.")
