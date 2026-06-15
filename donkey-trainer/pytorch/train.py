"""Donkey v3 trainer — Path-2 JEPA + detached CE.

Stage 4: plumbing-only. Runs --steps forward+backward pairs, writes metrics.
No LR schedule, no checkpointing yet — those land in stage 5.

Usage:
    python train.py <trace_dir> <out_dir> [--steps N] [--batch B] ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from donkey_config import DonkeyConfig
from donkey_world import DonkeyWorldRef
from trace_dataset import DonkeyTraceDataset, split_datasets
from losses import (jepa_loss, stochastic_donkey_loss, sinkhorn_token_wasserstein,
                    topk_distillation_loss, sorted_logit_wasserstein,
                    multimode_loss)
from donkey_world import DonkeyWorldStochastic
from forensics import MetricsLogger, ForensicsDumper, AdaptiveHuberDelta


def main():
    DEFAULT_DATASETS_ROOT = Path("/Volumes/TB5/donkey/dataset")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Names under <datasets-root>/traces/ to train on")
    ap.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    ap.add_argument("--run-name", type=str, default=None,
                    help="Run dir under <datasets-root>/runs/. Auto if omitted.")
    ap.add_argument("--spec", type=Path,
                    default=Path(__file__).parents[2] / "spec/donkey_v2_default.json")
    ap.add_argument("--lm-head", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_lm_head_fp16.pt")
    # Training loop control
    ap.add_argument("--max-steps", type=int, default=50000,
                    help="Hard cap; usually patience stops first")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-batch", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=200,
                    help="Validation cadence in training steps")
    ap.add_argument("--val-windows", type=int, default=2048,
                    help="How many val windows to evaluate per validation pass")
    ap.add_argument("--patience", type=int, default=5,
                    help="Stop after N consecutive validations without val_lce improvement")
    ap.add_argument("--min-improvement", type=float, default=0.005,
                    help="Minimum L_ce drop to count as 'improvement' (fraction, default 0.5%)")
    # Optimizer
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="0 -> auto = max(50, max_steps/100)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    # Loss
    ap.add_argument("--lam-reg", type=float, default=0.01)
    ap.add_argument("--lam-ce", type=float, default=0.5)
    ap.add_argument("--use-distillation", action="store_true",
                    help="Replace CE loss with top-K distillation loss")
    ap.add_argument("--distill-alpha", type=float, default=1.0,
                    help="Weight on L_outside (mass outside trunk's top-K)")
    ap.add_argument("--distill-beta", type=float, default=0.5,
                    help="Weight on L_within (shape match inside top-K)")
    ap.add_argument("--lam-wass-uld", type=float, default=0.0,
                    help="Weight for ULD-style sorted-logit Wasserstein loss "
                         "(K=3 distillation path only). 0 = disabled (default). "
                         "ULD paper recommends 1.5.")
    ap.add_argument("--m-modes", type=int, default=1,
                    help="Predictions per slot. 1=legacy, >1=multi-mode for tree-spec.")
    ap.add_argument("--lam-mode-div", type=float, default=0.5,
                    help="Diversity penalty weight on softmax overlap between modes.")
    # === Stochastic action-conditioned donkey (LeWM-faithful) ===
    ap.add_argument("--stochastic-donkey", action="store_true",
                    help="Use stochastic action-conditioned donkey (LeWM-faithful). "
                         "Drops K=3 parallel prediction; predicts single next latent "
                         "per call conditioned on random ε. Trained with Wasserstein "
                         "to trunk top-K + anchor Huber + SIGReg.")
    ap.add_argument("--eps-dim", type=int, default=64,
                    help="Dimension of ε noise vector (stochastic donkey)")
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Number of ε samples per training example (stochastic donkey)")
    ap.add_argument("--blue-noise-path", type=Path,
                    default=Path(__file__).parents[1] / "weights/blue_noise_eps64_1M.pt",
                    help="Path to precomputed blue noise dictionary")
    ap.add_argument("--embed-tokens-path", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_embed_tokens_fp16.pt",
                    help="Path to MiMo's embed_tokens for cosine ground metric")
    ap.add_argument("--lam-anchor", type=float, default=1.0,
                    help="Huber-anchor loss weight (stochastic donkey)")
    ap.add_argument("--lam-wass", type=float, default=1.0,
                    help="Wasserstein loss weight (stochastic donkey)")
    # Data split
    ap.add_argument("--split-train", type=float, default=0.90)
    ap.add_argument("--split-val",   type=float, default=0.05)
    ap.add_argument("--split-verify",type=float, default=0.05)
    ap.add_argument("--split-seed",  type=int, default=42)
    # Bookkeeping
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--single-batch", action="store_true",
                    help="Freeze one batch and reuse it (overfit gate)")
    ap.add_argument("--ckpt-every", type=int, default=2000,
                    help="Periodic checkpoint frequency")
    ap.add_argument("--ckpt-keep-last", type=int, default=3)
    ap.add_argument("--forensics-every", type=int, default=500,
                    help="Per-example parquet dump cadence")
    args = ap.parse_args()
    if args.warmup_steps == 0:
        args.warmup_steps = max(50, args.max_steps // 100)
    if args.data_seed is None:
        args.data_seed = args.seed
    assert abs(args.split_train + args.split_val + args.split_verify - 1.0) < 1e-6, \
        "splits must sum to 1.0"

    torch.manual_seed(args.seed)

    # Resolve out_dir from datasets_root + run_name
    if args.run_name is None:
        import datetime
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
        mix = "_".join(args.datasets)
        args.run_name = f"{stamp}_{mix}_s{args.steps}_b{args.batch}"
    out_dir = args.datasets_root / "runs" / args.run_name
    if out_dir.exists():
        print(f"[train] FAIL: run dir already exists: {out_dir}")
        sys.exit(2)
    out_dir.mkdir(parents=True)
    args.out_dir = out_dir

    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] device: {device}")
    print(f"[train] run dir: {out_dir}")

    cfg = DonkeyConfig.from_json(args.spec)
    print(f"[train] cfg: D={cfg.architecture.hidden_dim} W={cfg.sequence.window_size} "
          f"K={cfg.sequence.draft_size} trunk_H={cfg.trunk.hidden_dim}")

    trace_dirs = [args.datasets_root / "traces" / d for d in args.datasets]
    for td in trace_dirs:
        if not (td / "meta.json").exists():
            print(f"[train] FAIL: trace not found: {td}"); sys.exit(2)

    train_ds, val_ds, verify_ds = split_datasets(
        trace_dirs, cfg.sequence.window_size, cfg.sequence.draft_size,
        trunk_hidden=cfg.trunk.hidden_dim,
        fractions=(args.split_train, args.split_val, args.split_verify),
        seed=args.split_seed)
    print(f"[train] split (seed={args.split_seed}):")
    print(f"[train]   train:  {len(train_ds):>7} windows")
    print(f"[train]   val:    {len(val_ds):>7} windows")
    print(f"[train]   verify: {len(verify_ds):>7} windows")
    for name, n_t, n_v, n_ve in zip(train_ds.trace_names,
                                     train_ds.trace_sizes, val_ds.trace_sizes, verify_ds.trace_sizes):
        print(f"[train]     {name:30s} train={n_t:>7} val={n_v:>6} verify={n_ve:>6}")
    if len(train_ds) == 0:
        print("[train] FAIL: zero train windows"); sys.exit(2)

    data_gen = torch.Generator().manual_seed(args.data_seed)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, generator=data_gen)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch, shuffle=False, num_workers=0)
    # Verify_ds saved for end-of-run measurement; not used during training.
    # Save the split metadata so we can re-evaluate later.
    import json as _json
    (args.out_dir / "split.json").write_text(_json.dumps({
        "split_seed": args.split_seed,
        "fractions": [args.split_train, args.split_val, args.split_verify],
        "datasets": args.datasets,
        "train_size": len(train_ds), "val_size": len(val_ds), "verify_size": len(verify_ds),
    }, indent=2))

    if args.m_modes > 1:
        cfg.architecture.m_modes = args.m_modes
        print(f"[train] M-mode enabled: M={args.m_modes} modes per slot")
    if args.stochastic_donkey:
        print(f"[train] === STOCHASTIC DONKEY: ε_dim={args.eps_dim}, N={args.n_samples}, "
              f"K=1 (single-step prediction), Wasserstein+anchor+SIGReg loss ===")
        model = DonkeyWorldStochastic(cfg, eps_dim=args.eps_dim).to(device)
        model.init_for_training(seed=args.seed)
        # Load blue noise dictionary
        bn_data = torch.load(args.blue_noise_path, map_location="cpu", weights_only=False)
        blue_noise = bn_data["eps"].to(device).float()  # [N_dict, eps_dim]
        print(f"[train] loaded blue noise: {tuple(blue_noise.shape)} from {args.blue_noise_path.name}")
        # Load embed_tokens
        emb_data = torch.load(args.embed_tokens_path, map_location="cpu", weights_only=False)
        embed_tokens = emb_data["weight"].to(device).float()  # [V, H]
        print(f"[train] loaded embed_tokens: {tuple(embed_tokens.shape)}")
        # Load MiMo's final RMSNorm γ — required to decode donkey's raw output
        # via lm_head. Without this, donkey's output magnitude is wrong (~52 vs
        # MiMo's post-norm ~200) and softmax becomes near-uniform, killing
        # any gradient signal from token-space loss.
        from safetensors import safe_open
        TRUNK_SHARD = Path("/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M/model-00144-of-00144.safetensors")
        with safe_open(str(TRUNK_SHARD), framework="pt") as f:
            mimo_norm_gamma = f.get_tensor("model.norm.weight").to(device).float()
        mimo_norm_eps = 1e-5
        print(f"[train] loaded MiMo's final norm γ: {tuple(mimo_norm_gamma.shape)} "
              f"mean={mimo_norm_gamma.mean():.3f}")
    else:
        model = DonkeyWorldRef(cfg).to(device)
        model.init_for_training(seed=args.seed)
        blue_noise = None
        embed_tokens = None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] donkey params: {n_params/1e6:.2f}M")

    print(f"[train] loading lm_head from {args.lm_head}")
    lm_head_data = torch.load(args.lm_head, map_location="cpu", weights_only=False)
    lm_head = lm_head_data["weight"].to(device).to(torch.float32)
    lm_head.requires_grad_(False)
    print(f"[train] lm_head: {tuple(lm_head.shape)} {lm_head.dtype} on {lm_head.device}")

    # Param-group weight decay: matrices yes, norms/embeddings no.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_norm_or_embed = (
            "gamma" in name or "beta" in name or
            "ln_" in name or
            name.endswith("draft_queries") or name.endswith("pos_bias") or
            name.endswith("draft_query") or
            # ε-path params: weight decay actively shrinks them, killing
            # the stochastic latent variable. Exclude.
            "adaln" in name.lower() or
            "eps_encoder" in name
        )
        (no_decay_params if is_norm_or_embed else decay_params).append(p)
    print(f"[train] param groups: decay={sum(p.numel() for p in decay_params)/1e6:.2f}M  "
          f"no-decay={sum(p.numel() for p in no_decay_params)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        import math
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_final + (args.lr - args.lr_final) * cos

    def save_checkpoint(label: str, step: int):
        from safetensors.torch import save_file
        ckpt_dir = args.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        target = ckpt_dir / f"{label}.safetensors"
        save_file(sd, str(target), metadata={"step": str(step), "label": label})
        meta = {
            "step": step,
            "label": label,
            "loss_last_total": float(L.detach().item()) if "L" in dir() else None,
            "huber_delta": huber_delta.current_delta,
            "lr_at_save": lr_at(step),
        }
        (ckpt_dir / f"{label}.json").write_text(__import__("json").dumps(meta, indent=2))
        return target

    def run_validation(step):
        """Compute val L_pred, L_ce, L_reg, K=0 accept rate.

        Caps at args.val_windows to keep validation fast even on huge val sets.
        """
        model.eval()
        if args.stochastic_donkey:
            # Stochastic val: track Wasserstein, anchor, sig, and per-bucket
            # Wasserstein for collapse diagnostic. Also report N-sample tree-1
            # accept rate (mean accept rate across N samples).
            from losses import sigreg as _sigreg_eval
            totals = {"L": 0.0, "Lw": 0.0, "La": 0.0, "Ls": 0.0,
                      "var_eps": 0.0, "pair_dist": 0.0,
                      "wass_low": 0.0, "wass_low_n": 0,
                      "wass_med": 0.0, "wass_med_n": 0,
                      "wass_high": 0.0, "wass_high_n": 0,
                      "count": 0}
            top1_correct_any = 0  # any of N samples got top-1 correct
            top1_correct_best = 0  # the highest-prob sample got top-1 correct
            with torch.no_grad():
                seen = 0
                for vbatch in val_loader:
                    if seen >= args.val_windows: break
                    vh = vbatch["history"].to(device)
                    vt = vbatch["targets"].to(device)
                    vtok = vbatch["target_tokens"].to(device)
                    Bv = vh.shape[0]
                    Nv = args.n_samples
                    # Sample N ε from blue noise (deterministic-ish for val: use fixed first chunk)
                    eps_v = blue_noise[:Bv * Nv].reshape(Bv, Nv, -1)
                    z_pred_n = model.forward_n_samples(vh, eps_v)  # [B, N, trunk_hidden]
                    z_target_th = vt[:, :, 0]                       # [B, trunk_hidden]
                    # Apply MiMo's RMSNorm before lm_head decode (see comment in train step).
                    z_for_decode_v = z_pred_n.float()
                    rms_v = z_for_decode_v.pow(2).mean(dim=-1, keepdim=True).sqrt()
                    z_normed_v = z_for_decode_v * mimo_norm_gamma / (rms_v + mimo_norm_eps)
                    donkey_logits = z_normed_v @ lm_head.float().t()  # [B, N, V]
                    tk_ids   = vbatch["target_topk_ids"].to(device)
                    tk_probs = vbatch["target_topk_probs"].to(device)
                    z_pred_flat = z_pred_n.reshape(Bv * Nv, -1, 1)
                    z_pred_latent = model.input_proj(z_pred_flat).squeeze(-1).view(Bv, Nv, -1)
                    z_target_latent = model.input_proj(z_target_th.unsqueeze(-1)).squeeze(-1)
                    z_history_latent = model.input_proj(vh).reshape(-1, model.input_proj.W.shape[0])

                    vloss = stochastic_donkey_loss(
                        z_pred_samples=z_pred_latent, z_target=z_target_latent,
                        donkey_logits=donkey_logits,
                        trunk_topk_ids=tk_ids, trunk_topk_probs=tk_probs,
                        embed_tokens=embed_tokens, sigreg_fn=_sigreg_eval,
                        history_latents=z_history_latent,
                        lam_anchor=args.lam_anchor, lam_wass=args.lam_wass,
                        lam_sig=args.lam_reg, huber_delta=huber_delta.current_delta)

                    # Per-position Wasserstein for bucketed diagnostic
                    wass_pp = sinkhorn_token_wasserstein(
                        donkey_logits, tk_ids, tk_probs, embed_tokens)["per_pos"]  # [B]
                    # Compute trunk entropy on top-K as the bucketing key
                    trunk_entropy = -(tk_probs.clamp(min=1e-12) * tk_probs.clamp(min=1e-12).log()).sum(-1)
                    low_mask  = trunk_entropy < 0.5
                    high_mask = trunk_entropy > 2.0
                    med_mask  = ~(low_mask | high_mask)
                    totals["wass_low"]    += wass_pp[low_mask].sum().item() if low_mask.any() else 0.0
                    totals["wass_low_n"]  += int(low_mask.sum().item())
                    totals["wass_med"]    += wass_pp[med_mask].sum().item() if med_mask.any() else 0.0
                    totals["wass_med_n"]  += int(med_mask.sum().item())
                    totals["wass_high"]   += wass_pp[high_mask].sum().item() if high_mask.any() else 0.0
                    totals["wass_high_n"] += int(high_mask.sum().item())

                    b = vh.shape[0]
                    totals["L"]    += float(vloss["total"].item())         * b
                    totals["Lw"]   += float(vloss["wass"].item())          * b
                    totals["La"]   += float(vloss["anchor"].item())        * b
                    totals["Ls"]   += float(vloss["sig"].item())           * b
                    totals["var_eps"]   += float(vloss["var_over_eps"].item()) * b
                    totals["pair_dist"] += float(vloss["pairwise_dist"].item()) * b
                    totals["count"] += b

                    # Accept rate: any of N samples produces correct top-1 token?
                    sample_argmax = donkey_logits.argmax(dim=-1)        # [B, N]
                    correct_per_sample = (sample_argmax == vtok[:, 0:1])  # [B, N]
                    top1_correct_any += int(correct_per_sample.any(dim=-1).sum().item())
                    # "Best sample": the one whose decoded distribution has highest peak prob
                    max_prob_per_sample = donkey_logits.softmax(-1).max(-1).values  # [B, N]
                    best_idx = max_prob_per_sample.argmax(dim=-1)                    # [B]
                    best_argmax = sample_argmax[torch.arange(Bv), best_idx]          # [B]
                    top1_correct_best += int((best_argmax == vtok[:, 0]).sum().item())
                    seen += b
            model.train()
            if totals["count"] == 0: return None
            n = totals["count"]
            out = {
                "step": step,
                "val_examples": n,
                "val/L":  totals["L"]  / n,
                "val/Lw": totals["Lw"] / n,
                "val/La": totals["La"] / n,
                "val/Ls": totals["Ls"] / n,
                "val/var_over_eps":  totals["var_eps"]   / n,
                "val/pairwise_dist": totals["pair_dist"] / n,
                "val/wass_low":   (totals["wass_low"]  / max(totals["wass_low_n"], 1)),
                "val/wass_med":   (totals["wass_med"]  / max(totals["wass_med_n"], 1)),
                "val/wass_high":  (totals["wass_high"] / max(totals["wass_high_n"], 1)),
                "val/wass_low_n":  totals["wass_low_n"],
                "val/wass_med_n":  totals["wass_med_n"],
                "val/wass_high_n": totals["wass_high_n"],
                "val/top1_any":  top1_correct_any  / n,
                "val/top1_best": top1_correct_best / n,
            }
            return out
        # End stochastic val branch; legacy K=3 path below.
        totals = {"L": 0.0, "Lp": 0.0, "Lr": 0.0, "Lce": 0.0, "count": 0}
        correct = [0, 0, 0]; total_per_k = [0, 0, 0]
        with torch.no_grad():
            seen = 0
            for vbatch in val_loader:
                if seen >= args.val_windows: break
                vh = vbatch["history"].to(device)
                vt = vbatch["targets"].to(device)
                vtok = vbatch["target_tokens"].to(device)
                vout = model.forward_for_training(vh)
                z_pred = vout["z_pred"]; pred_decoded = vout["pred_decoded"]
                M_v = vout.get("m_modes", 1)
                b = vh.shape[0]

                if M_v > 1:
                    # Multi-mode val path.
                    vtk_ids = vbatch.get("target_topk_ids")
                    vtk_probs = vbatch.get("target_topk_probs")
                    if vtk_ids is not None:
                        vtk_ids = vtk_ids.to(device)
                    if vtk_probs is not None:
                        vtk_probs = vtk_probs.to(device)
                    vloss = multimode_loss(
                        pred_decoded, vtok, lm_head,
                        trunk_topk_ids=vtk_ids,
                        trunk_topk_probs=vtk_probs,
                        lam_reg_l2=args.lam_reg,
                    )
                    totals["L"]   += float(vloss["total"].item()) * b
                    totals["Lp"]  += 0.0
                    totals["Lr"]  += float(vloss["reg"].item())   * b
                    totals["Lce"] += float(vloss["any"].item())   * b
                    totals["count"] += b
                    # Best-mode K-slot accept rate: for each (b, k), pick the
                    # mode with the lowest CE-to-target; check whether THAT
                    # mode's argmax matches the target.
                    Bv, Hv, Kv, Mv = pred_decoded.shape
                    pf = pred_decoded.permute(0, 2, 3, 1).reshape(
                        Bv * Kv * Mv, Hv).float()
                    logits = (pf @ lm_head.float().t()).view(Bv, Kv, Mv, -1)
                    # Best mode per (b,k): the one whose argmax IS the target
                    argmax_mode = logits.argmax(dim=-1)  # [B, K, M]
                    correct_mode = (argmax_mode == vtok.unsqueeze(-1))  # [B, K, M]
                    any_mode_correct = correct_mode.any(dim=-1)  # [B, K]
                    for k in range(Kv):
                        correct[k] += int(any_mode_correct[:, k].sum().item())
                        total_per_k[k] += b
                else:
                    # Legacy single-mode val path (K=3 parallel).
                    z_target = model.input_proj(vt)
                    vloss = jepa_loss(
                        z_pred=z_pred, z_target=z_target,
                        pred_decoded=pred_decoded, target_tokens=vtok,
                        lm_head=lm_head, huber_delta_tracker=huber_delta,
                        lam_reg=args.lam_reg, lam_ce=args.lam_ce)
                    if args.use_distillation and "target_topk_ids" in vbatch:
                        Bv, Kv = pred_decoded.shape[0], pred_decoded.shape[2]
                        pfv = pred_decoded.permute(0, 2, 1).reshape(Bv * Kv, -1).float()
                        dlogits = pfv @ lm_head.float().t()
                        tk_ids_v   = vbatch["target_topk_ids"].to(device).view(Bv * Kv, -1)
                        tk_probs_v = vbatch["target_topk_probs"].to(device).view(Bv * Kv, -1).float()
                        distill_v = topk_distillation_loss(
                            dlogits, tk_ids_v, tk_probs_v,
                            alpha=args.distill_alpha, beta=args.distill_beta)
                        vloss["ce"] = distill_v["total"]
                    totals["L"]   += float(vloss["total"].item()) * b
                    totals["Lp"]  += float(vloss["pred"].item())  * b
                    totals["Lr"]  += float(vloss["reg"].item())   * b
                    totals["Lce"] += float(vloss["ce"].item())    * b
                    totals["count"] += b
                    logits = (pred_decoded.permute(0, 2, 1).float() @ lm_head.float().t())
                    argmax = logits.argmax(dim=-1)
                    for k in range(argmax.shape[1]):
                        correct[k] += int((argmax[:, k] == vtok[:, k]).sum().item())
                        total_per_k[k] += b
                seen += b
        model.train()
        if totals["count"] == 0: return None
        n = totals["count"]
        out = {
            "step": step,
            "val_examples": n,
            "val/L":   totals["L"]   / n,
            "val/Lp":  totals["Lp"]  / n,
            "val/Lr":  totals["Lr"]  / n,
            "val/Lce": totals["Lce"] / n,
            "val/K0_acc": (correct[0] / max(1, total_per_k[0])) if total_per_k[0] else 0.0,
            "val/K1_acc": (correct[1] / max(1, total_per_k[1])) if total_per_k[1] else 0.0,
            "val/K2_acc": (correct[2] / max(1, total_per_k[2])) if total_per_k[2] else 0.0,
        }
        return out
    

    # Save run config
    run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_config["train_size"] = len(train_ds)
    run_config["val_size"] = len(val_ds)
    run_config["verify_size"] = len(verify_ds)
    (args.out_dir / "config.json").write_text(__import__("json").dumps(run_config, indent=2))
    print(f"[train] wrote run config -> {args.out_dir / 'config.json'}")

    metrics_log = MetricsLogger(args.out_dir / "metrics.jsonl")
    huber_delta = AdaptiveHuberDelta()
    forensics = (ForensicsDumper(args.out_dir / "forensics", dump_every=args.forensics_every)
                 if args.forensics_every > 0 else None)
    if forensics:
        print(f"[train] forensics every {args.forensics_every} steps -> {forensics.dump_dir}")

    print(f"[train] === step loop: max {args.max_steps} steps, val every {args.val_every}, patience {args.patience} ===")
    model.train()
    t_start = time.monotonic()
    val_log = MetricsLogger(args.out_dir / "val_metrics.jsonl")
    best_val_metric = float("inf")  # was best_val_metric; works for both modes
    best_step = 0
    patience_counter = 0

    frozen_batch = None
    if args.single_batch:
        for b in loader:
            frozen_batch = b
            break
        print(f"[train] SINGLE-BATCH MODE: batch_size={frozen_batch['history'].shape[0]}, "
              f"target_tokens={frozen_batch['target_tokens'][:2].tolist()}")

    step = 0
    while step < args.max_steps:
        if args.single_batch:
            batch = frozen_batch
        else:
            try:
                batch = next(loader_iter)
            except (StopIteration, NameError):
                loader_iter = iter(loader)
                batch = next(loader_iter)

        history = batch["history"].to(device)
        targets = batch["targets"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        if args.stochastic_donkey:
            # Stochastic: sample N ε from blue noise dictionary; predict N latents
            # per position; Wasserstein + anchor + SIGReg loss.
            B = history.shape[0]
            N = args.n_samples
            # Sample N indices from the blue noise pool
            bn_idx = torch.randint(0, blue_noise.shape[0], (B, N), device=device)
            eps_batch = blue_noise[bn_idx]  # [B, N, eps_dim]
            # Run donkey N times (broadcast history × N)
            z_pred_n = model.forward_n_samples(history, eps_batch)  # [B, N, trunk_hidden]
            # Project to latent space using same Φ_O the model uses on history.
            # Predicted z_pred_n is in trunk_hidden space (output of W_head); we
            # need to compare it to z_target also in trunk_hidden space — and we
            # ALSO need to decode through lm_head for Wasserstein.
            # Use position W (first target slot) for single-step prediction.
            # targets shape: [B, trunk_hidden, K]; first slot:
            z_target_th = targets[:, :, 0]  # [B, trunk_hidden]  (first predicted slot)
            # Decode N samples through lm_head — with proper MiMo RMSNorm first.
            # Donkey emits raw residual-stream-like vectors (~50 norm). MiMo's
            # final norm scales these to ~200 norm before lm_head. Skipping it
            # gives near-uniform softmax → no useful gradient signal.
            z_for_decode = z_pred_n.float()  # [B, N, trunk_hidden]
            rms = z_for_decode.pow(2).mean(dim=-1, keepdim=True).sqrt()
            z_normed = z_for_decode * mimo_norm_gamma / (rms + mimo_norm_eps)
            donkey_logits = z_normed @ lm_head.float().t()  # [B, N, V]
            target_topk_ids = batch["target_topk_ids"].to(device)     # [B, K_topk]
            target_topk_probs = batch["target_topk_probs"].to(device) # [B, K_topk]
            # Project everything through input_proj (Φ_O) for latent-space anchor.
            # z_pred_n is in [B, N, trunk_hidden]; flatten then project.
            z_pred_flat = z_pred_n.reshape(B * N, -1, 1)  # [B*N, trunk_hidden, 1]
            z_pred_latent = model.input_proj(z_pred_flat).squeeze(-1).view(B, N, -1)  # [B, N, D]
            with torch.no_grad():
                z_target_latent = model.input_proj(z_target_th.unsqueeze(-1)).squeeze(-1)  # [B, D]
                z_history_latent = model.input_proj(history).reshape(-1, model.input_proj.W.shape[0])
                # ^^^ [B*W, D] for SIGReg pool
            def _sigreg_fn(z):
                from losses import sigreg as _sig
                return _sig(z)
            losses = stochastic_donkey_loss(
                z_pred_samples=z_pred_latent,
                z_target=z_target_latent,
                donkey_logits=donkey_logits,
                trunk_topk_ids=target_topk_ids,
                trunk_topk_probs=target_topk_probs,
                embed_tokens=embed_tokens,
                sigreg_fn=_sigreg_fn,
                history_latents=z_history_latent,
                lam_anchor=args.lam_anchor,
                lam_wass=args.lam_wass,
                lam_sig=args.lam_reg,
                huber_delta=huber_delta.current_delta,
            )
            L = losses["total"]
        else:
            out = model.forward_for_training(history)
            z_pred = out["z_pred"]
            pred_decoded = out["pred_decoded"]
            M = out.get("m_modes", 1)

            if M > 1:
                tk_ids = batch.get("target_topk_ids")
                tk_probs = batch.get("target_topk_probs")
                if tk_ids is not None:
                    tk_ids = tk_ids.to(device)
                if tk_probs is not None:
                    tk_probs = tk_probs.to(device)
                losses = multimode_loss(
                    pred_decoded, target_tokens, lm_head,
                    trunk_topk_ids=tk_ids,
                    trunk_topk_probs=tk_probs,
                    lam_reg_l2=args.lam_reg,
                )
                L = losses["total"]
                losses["pred"] = torch.zeros((), device=L.device)
                losses["ce"] = losses["any"]
            else:
                with torch.no_grad():
                    z_target = model.input_proj(targets)

                losses = jepa_loss(
                z_pred=z_pred, z_target=z_target,
                pred_decoded=pred_decoded, target_tokens=target_tokens,
                lm_head=lm_head, huber_delta_tracker=huber_delta,
                lam_reg=args.lam_reg, lam_ce=args.lam_ce,
            )
            L = losses["total"]

            # Optional: KL distillation on top-K trunk targets (replaces CE if enabled).
            # The original "KL distill" run had this in val only — fix that.
            if args.use_distillation and "target_topk_ids" in batch:
                B_tr, K_tr = pred_decoded.shape[0], pred_decoded.shape[2]
                pf = pred_decoded.permute(0, 2, 1).reshape(B_tr * K_tr, -1).float()
                dlogits = pf @ lm_head.float().t()  # [B*K, V]
                tk_ids = batch["target_topk_ids"].to(device).view(B_tr * K_tr, -1)
                tk_probs = batch["target_topk_probs"].to(device).view(B_tr * K_tr, -1).float()
                distill = topk_distillation_loss(
                    dlogits, tk_ids, tk_probs,
                    alpha=args.distill_alpha, beta=args.distill_beta)
                # Replace CE component of L with distill loss; remove CE; add distill.
                L = L - args.lam_ce * losses["ce"] + distill["total"]
                losses["distill_outside"] = distill["outside"]
                losses["distill_within"] = distill["within"]
                losses["distill_total"] = distill["total"].detach()

                # Optional: ULD-style sorted-logit Wasserstein on top of distillation.
                if args.lam_wass_uld > 0.0:
                    wuld = sorted_logit_wasserstein(dlogits, tk_ids, tk_probs)
                    L = L + args.lam_wass_uld * wuld["total"]
                    losses["wass_uld"] = wuld["total"].detach()


        lr_now = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad()
        L.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics: stochastic mode has different keys (anchor/wass/sig)
        # while CE/distill mode has pred/reg/ce. Log both shapes appropriately.
        if args.stochastic_donkey:
            metrics = {
                "loss/total":         L,
                "loss/anchor":        losses["anchor"],
                "loss/wass":          losses["wass"],
                "loss/sig":           losses["sig"],
                "metric/pairwise_dist": losses["pairwise_dist"],
                "metric/var_over_eps":  losses["var_over_eps"],
                "metric/huber_delta":   huber_delta.current_delta,
            }
        else:
            metrics = {
                "loss/total": L,
                "loss/pred":  losses["pred"],
                **({"loss/distill": losses["distill_total"]}
                   if "distill_total" in losses else {}),
                **({"loss/wass_uld": losses["wass_uld"]}
                   if "wass_uld" in losses else {}),
                **({"loss/div": losses["div"]}
                   if "div" in losses else {}),
                **({"loss/mode_collapse_frac": losses["mode_collapse_frac"]}
                   if "mode_collapse_frac" in losses else {}),
                **({"loss/best_mode_acc": losses["best_mode_acc"]}
                   if "best_mode_acc" in losses else {}),
                **({"loss/mode0_acc": losses["mode0_acc"]}
                   if "mode0_acc" in losses else {}),
                **({"loss/ce_mode_0": losses["ce_mode_0"]}
                   if "ce_mode_0" in losses else {}),
                **({"loss/ce_mode_1": losses["ce_mode_1"]}
                   if "ce_mode_1" in losses else {}),
                "loss/reg":   losses["reg"],
                "loss/ce":    losses["ce"],
                "metric/huber_delta":   huber_delta.current_delta,
            "metric/grad_norm":     grad_norm,
            "metric/z_pred_norm":   z_pred.detach().norm(dim=1).mean(),
            "metric/z_target_norm": (z_target.detach().norm(dim=1).mean()
                                     if "z_target" in dir() else
                                     torch.zeros((), device=L.device)),
            "lr": optimizer.param_groups[0]["lr"],
        }
        metrics_log.write(step, metrics)
        if args.stochastic_donkey:
            print(f"[train] step {step}: L={L.item():.4f}  "
                  f"La={losses['anchor'].item():.4f}  Lw={losses['wass'].item():.4f}  "
                  f"Ls={losses['sig'].item():.4f}  "
                  f"varε={losses['var_over_eps'].item():.5f}  "
                  f"pd={losses['pairwise_dist'].item():.3f}  "
                  f"lr={lr_now:.2e}  |g|={grad_norm:.3f}")
        else:
            print(f"[train] step {step}: L={L.item():.4f}  "
                  f"Lp={losses['pred'].item():.4f}  Lr={losses['reg'].item():.4f}  "
                  f"Lce={losses['ce'].item():.4f}  δ={huber_delta.current_delta:.3f}  "
                  f"lr={lr_now:.2e}  |g|={grad_norm:.3f}")

        if forensics and forensics.should_dump(step) and "z_target" in dir():
            # Skip forensics in multi-mode path (no z_target computed).
            dpath = forensics.collect_and_dump(
                step=step, batch=batch,
                z_pred=z_pred, z_target=z_target, pred_decoded=pred_decoded,
                lm_head=lm_head, losses_per_ex=None)
            print(f"[train]   forensics -> {dpath.name}")

        if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            ck = save_checkpoint(f"step_{step+1:06d}", step + 1)
            print(f"[train]   ckpt -> {ck.name}")
            ckpts = sorted((args.out_dir / "checkpoints").glob("step_*.safetensors"))
            for old_p in ckpts[:-args.ckpt_keep_last]:
                old_p.unlink(missing_ok=True)
                old_p.with_suffix(".json").unlink(missing_ok=True)

        # Validation pass + patience
        if args.val_every > 0 and (step + 1) % args.val_every == 0:
            vres = run_validation(step + 1)
            if vres is not None:
                val_log.write(step + 1, vres)
                if args.stochastic_donkey:
                    print(f"[train]   VAL step {step+1}: Lw={vres['val/Lw']:.4f}  "
                          f"La={vres['val/La']:.4f}  Ls={vres['val/Ls']:.4f}  "
                          f"top1_any={100*vres['val/top1_any']:.1f}%  "
                          f"top1_best={100*vres['val/top1_best']:.1f}%  "
                          f"varε={vres['val/var_over_eps']:.5f}  "
                          f"pd={vres['val/pairwise_dist']:.3f}  "
                          f"(best Lw: {best_val_metric:.4f})")
                    print(f"[train]   per-entropy buckets: "
                          f"low={vres['val/wass_low']:.4f} (n={vres['val/wass_low_n']})  "
                          f"med={vres['val/wass_med']:.4f} (n={vres['val/wass_med_n']})  "
                          f"high={vres['val/wass_high']:.4f} (n={vres['val/wass_high_n']})")
                else:
                    print(f"[train]   VAL step {step+1}: Lce={vres['val/Lce']:.4f}  "
                          f"Lp={vres['val/Lp']:.4f}  "
                          f"K0={100*vres['val/K0_acc']:.1f}% K1={100*vres['val/K1_acc']:.1f}% "
                          f"K2={100*vres['val/K2_acc']:.1f}%  "
                          f"(best Lce: {best_val_metric:.4f})")
                # Patience metric: val/Lw (Wasserstein) in stochastic mode,
                # val/Lce (CE/distill) in legacy mode.
                patience_key = "val/Lw" if args.stochastic_donkey else "val/Lce"
                if vres[patience_key] < best_val_metric * (1.0 - args.min_improvement):
                    best_val_metric = vres[patience_key]
                    patience_counter = 0
                    best_step = step + 1
                    bpath = save_checkpoint("best", step + 1)
                    print(f"[train]     NEW BEST val {('Lw' if args.stochastic_donkey else 'Lce')}={best_val_metric:.4f} -> {bpath.name}")
                else:
                    patience_counter += 1
                    print(f"[train]     no improvement ({patience_counter}/{args.patience} "
                          f"since best at step {best_step})")
                if patience_counter >= args.patience:
                    print(f"[train] STOP: patience exhausted at step {step+1}. "
                          f"Best val {('Lw' if args.stochastic_donkey else 'Lce')}={best_val_metric:.4f} at step {best_step}.")
                    break

        step += 1

    final_ckpt = save_checkpoint("final", step)
    print(f"[train] final ckpt -> {final_ckpt}")
    metrics_log.close()
    print(f"[train] done in {time.monotonic()-t_start:.1f}s  "
          f"metrics: {args.out_dir/'metrics.jsonl'}")


if __name__ == "__main__":
    main()
