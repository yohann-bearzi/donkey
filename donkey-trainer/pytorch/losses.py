"""Donkey v3 Path-2 JEPA loss.

L = L_pred + lam_reg * L_reg + lam_ce * L_ce

  L_pred = Huber(z_pred, z_target.detach(), delta=adaptive)   in 1024-d latent
  L_reg  = SIGReg(z_pred ∪ z_target.detach())                  anti-collapse
  L_ce   = CE(lm_head(pred_decoded), target_token)             pred_decoded already
                                                                detaches latent at the
                                                                model boundary
"""
import torch
import torch.nn.functional as F


def sigreg(z: torch.Tensor, n_sketches: int = 16) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularizer (LeJEPA, moment-matching).

    Project z onto random unit vectors; penalize deviation from N(0, 1)
    per projection via mean^2 + (std - 1)^2. Single hyperparameter
    (n_sketches). Full LeJEPA SIGReg uses empirical CDF distance; this
    moment-matching variant is cheaper and adequate for our pilot.
    """
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    N, D = z.shape
    g = torch.randn(D, n_sketches, device=z.device, dtype=z.dtype)
    g = g / g.norm(dim=0, keepdim=True).clamp(min=1e-8)
    proj = z @ g
    mean = proj.mean(dim=0)
    std = proj.std(dim=0)
    return (mean ** 2).mean() + ((std - 1.0) ** 2).mean()


def jepa_loss(
    z_pred,             # [B, D, K]  with gradient
    z_target,           # [B, D, K]  detached upstream
    pred_decoded,       # [B, H, K]  grad to decoder only (detached upstream)
    target_tokens,      # [B, K]
    lm_head,            # [V, H]     frozen
    huber_delta_tracker,
    lam_reg: float = 0.01,
    lam_ce: float = 0.5,
):
    B, D, K = z_pred.shape
    H = pred_decoded.shape[1]
    V = lm_head.shape[0]

    z_pred_flat   = z_pred.permute(0, 2, 1).reshape(B * K, D)
    z_target_flat = z_target.permute(0, 2, 1).reshape(B * K, D)

    abs_err = (z_pred_flat - z_target_flat).abs()
    delta = max(huber_delta_tracker.update(abs_err), 1e-3)
    L_pred = F.huber_loss(z_pred_flat, z_target_flat, delta=delta)

    z_union = torch.cat([z_pred_flat, z_target_flat], dim=0)
    L_reg = sigreg(z_union)

    pred_flat = pred_decoded.permute(0, 2, 1).reshape(B * K, H)
    # Compute logits in fp32 for numerical stability with 152K vocab
    logits = (pred_flat.float() @ lm_head.float().t())          # [B*K, V]
    L_ce = F.cross_entropy(logits, target_tokens.reshape(-1).long())

    total = L_pred + lam_reg * L_reg + lam_ce * L_ce
    return {"total": total, "pred": L_pred, "reg": L_reg, "ce": L_ce}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, K, H, V = 4, 1024, 3, 4096, 152576

    class FakeTracker:
        def __init__(self): self.current_delta = 1.0
        def update(self, x):
            self.current_delta = float(x.detach().quantile(0.95))
            return self.current_delta

    z_pred = torch.randn(B, D, K, requires_grad=True)
    z_target = torch.randn(B, D, K)
    pred_decoded = (torch.randn(B, H, K) * 0.01).requires_grad_(True)
    target_tokens = torch.randint(0, V, (B, K))
    lm_head = torch.randn(V, H) * 0.01

    out = jepa_loss(z_pred, z_target, pred_decoded, target_tokens,
                    lm_head, FakeTracker())
    print(f"[loss] total={out['total'].item():.4f}  pred={out['pred'].item():.4f}  "
          f"reg={out['reg'].item():.4f}  ce={out['ce'].item():.4f}")
    out["total"].backward()
    assert z_pred.grad is not None and pred_decoded.grad is not None
    print(f"[loss] z_pred grad norm={z_pred.grad.norm().item():.4f}")
    print(f"[loss] pred_decoded grad norm={pred_decoded.grad.norm().item():.4f}")
    # SIGReg standalone sanity: standard normal data should give ~0 loss
    z_normal = torch.randn(2000, 64)
    z_collapsed = torch.zeros(2000, 64) + torch.randn(64) * 0.01  # all same
    print(f"[sigreg] normal -> {sigreg(z_normal).item():.6f}  (expect ~0)")
    print(f"[sigreg] collapsed -> {sigreg(z_collapsed).item():.4f}  (expect >> 0)")
    print("[loss] OK")



def topk_distillation_loss(donkey_logits, trunk_topk_ids, trunk_topk_probs,
                            alpha=1.0, beta=0.5):
    """Distillation loss: penalize donkey for mass outside trunk's top-K candidates,
    and mismatched shape within them.

    donkey_logits:     [B*K_donkey, V]  donkey's full vocab logits at each draft slot
    trunk_topk_ids:    [B*K_donkey, K_t] trunk's top-K_trunk token IDs (from trace)
    trunk_topk_probs:  [B*K_donkey, K_t] trunk's normalized top-K_trunk probabilities

    Returns dict with 'total', 'outside', 'within'.
    """
    import torch.nn.functional as F
    donkey_logp = F.log_softmax(donkey_logits, dim=-1)              # [N, V]
    donkey_logp_at_topk = donkey_logp.gather(-1, trunk_topk_ids)    # [N, K_t]
    donkey_prob_at_topk = donkey_logp_at_topk.exp()                 # [N, K_t]

    # L_outside: penalize mass that landed outside trunk's top-K set
    mass_inside = donkey_prob_at_topk.sum(dim=-1).clamp(min=1e-8)   # [N]
    L_outside = (-mass_inside.log()).mean()

    # L_within: shape match inside the top-K (renormalize donkey within trunk's top-K)
    donkey_norm = donkey_prob_at_topk / mass_inside.unsqueeze(-1)    # [N, K_t], sums to 1
    # KL(trunk || donkey) on the top-K renormalized distributions
    # F.kl_div expects (log_input, target); target is trunk_topk_probs.
    # Trunk's top-K probs may not sum to exactly 1 (some tail mass); renormalize.
    trunk_norm = trunk_topk_probs / trunk_topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    L_within = F.kl_div(
        donkey_norm.clamp(min=1e-8).log(),
        trunk_norm,
        reduction='batchmean'
    )

    return {
        "total":   alpha * L_outside + beta * L_within,
        "outside": L_outside.detach(),
        "within":  L_within.detach(),
    }



# ============================================================================
# Stochastic donkey loss (Wasserstein + anchor + SIGReg)
# ============================================================================

def sinkhorn_token_wasserstein(donkey_logits, trunk_topk_ids, trunk_topk_probs,
                                embed_tokens, temp=0.05, n_iters=20):
    """Empirical Wasserstein between donkey's N samples and trunk's top-K.

    Source side (donkey): N samples. Each sample i contributes mass 1/N at
    position s_i = softmax(logits_i) @ embed_tokens — the differentiable
    expectation of the argmax in embedding space.

    Target side (trunk): K top-K tokens. Token k contributes mass π_k at
    position embed_tokens[trunk_topk_ids[k]].

    Cost: N×K matrix with c[i,k] = 1 - cos(s_i, target_k).

    Sinkhorn solves OT honoring per-sample masses. Key property:
    if all N samples produce identical softmaxes, all source positions
    collapse to one point. The transport cost from that point to the
    K-spread target is FIXED regardless of N (since total source mass = 1
    must move to K targets in proportion π_k). Donkey is incentivized
    to spread sources to cover trunk's distribution.

    Args:
        donkey_logits:    [B, N, V]
        trunk_topk_ids:   [B, K]      trunk token IDs
        trunk_topk_probs: [B, K]      trunk probs (each row sums to 1.0)
        embed_tokens:     [V, H]      token embedding matrix
        temp:             Sinkhorn entropy temperature
        n_iters:          Sinkhorn iterations

    Returns:
        dict with 'total' (mean OT cost) and 'per_pos' (per-batch costs).
    """
    import torch.nn.functional as F
    B, N, V = donkey_logits.shape
    K = trunk_topk_ids.shape[1]
    H = embed_tokens.shape[1]

    # ---- Source positions: softmax-weighted embedding mean per sample.
    donkey_probs = F.softmax(donkey_logits.float(), dim=-1)        # [B, N, V]
    source_emb = donkey_probs @ embed_tokens.float()                # [B, N, H]

    # ---- Target positions: embeddings of trunk's top-K tokens.
    target_emb = embed_tokens[trunk_topk_ids].float()               # [B, K, H]

    # ---- Cost matrix: cosine distance, NxK.
    source_n = F.normalize(source_emb, dim=-1)                      # [B, N, H]
    target_n = F.normalize(target_emb, dim=-1)                      # [B, K, H]
    cos_sim = torch.einsum('bnh,bkh->bnk', source_n, target_n)     # [B, N, K]
    cost = 1.0 - cos_sim                                            # [B, N, K]

    # ---- Source mass: 1/N per sample. Target mass: trunk's probs.
    source_mass = torch.full((B, N), 1.0 / N, device=cost.device, dtype=cost.dtype)
    target_mass = trunk_topk_probs.float()                          # [B, K]
    log_a = source_mass.clamp(min=1e-8).log()                       # [B, N]
    log_b = target_mass.clamp(min=1e-8).log()                       # [B, K]

    # ---- Sinkhorn in log space.
    u = torch.zeros_like(log_a)                                     # [B, N]
    v = torch.zeros_like(log_b)                                     # [B, K]
    neg_cost_over_temp = -cost / temp                               # [B, N, K]

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(neg_cost_over_temp + v.unsqueeze(1), dim=-1)
        v = log_b - torch.logsumexp(neg_cost_over_temp + u.unsqueeze(-1), dim=-2)

    # ---- Compute transport cost.
    log_pi = neg_cost_over_temp + u.unsqueeze(-1) + v.unsqueeze(1)  # [B, N, K]
    transport_cost = (log_pi.exp() * cost).sum(dim=(-2, -1))        # [B]

    return {
        "total":   transport_cost.mean(),
        "per_pos": transport_cost.detach(),
    }


def stochastic_donkey_loss(z_pred_samples, z_target, donkey_logits,
                            trunk_topk_ids, trunk_topk_probs, embed_tokens,
                            sigreg_fn, history_latents=None,
                            lam_anchor=1.0, lam_wass=1.0, lam_sig=0.01,
                            huber_delta=1.0):
    """Stochastic donkey loss.

    Args:
        z_pred_samples:   [B, N, D]   N predicted latents per position
        z_target:         [B, D]      true next latent (from trace, projected)
        donkey_logits:    [B, N, V]   decoded logits per sample
        trunk_topk_ids:   [B, K]
        trunk_topk_probs: [B, K]
        embed_tokens:     [V, H]
        sigreg_fn:        callable(z) -> scalar (operates on [N_latents, D] tensor)
        history_latents:  [B, D]      optionally include history in SIGReg pool
        lam_anchor, lam_wass, lam_sig: loss weights
        huber_delta:      Huber delta

    Returns:
        dict with 'total' (loss for backward) + diagnostics (all detached).
    """
    import torch
    import torch.nn.functional as F

    B, N, D = z_pred_samples.shape

    # ANCHOR: Huber on closest sample to z_target
    dists = (z_pred_samples - z_target.unsqueeze(1)).pow(2).sum(-1)  # [B, N]
    closest_idx = dists.argmin(dim=-1)                                # [B]
    closest = z_pred_samples[torch.arange(B), closest_idx]            # [B, D]
    L_anchor = F.huber_loss(closest, z_target.detach(), delta=huber_delta)

    # WASSERSTEIN: token-space OT to trunk's top-K
    wass = sinkhorn_token_wasserstein(
        donkey_logits, trunk_topk_ids, trunk_topk_probs, embed_tokens)

    # SIGREG: on joint latent set
    latent_pool = z_pred_samples.reshape(B * N, D)
    latent_pool = torch.cat([latent_pool, z_target], dim=0)
    if history_latents is not None:
        latent_pool = torch.cat([latent_pool, history_latents], dim=0)
    L_sig = sigreg_fn(latent_pool)

    L_total = lam_anchor * L_anchor + lam_wass * wass["total"] + lam_sig * L_sig

    return {
        "total":         L_total,
        "anchor":        L_anchor.detach(),
        "wass":          wass["total"].detach(),
        "sig":           L_sig.detach(),
        "pairwise_dist": (z_pred_samples.unsqueeze(2)
                          - z_pred_samples.unsqueeze(1)
                          ).pow(2).sum(-1).sqrt().mean().detach(),
        "var_over_eps":  z_pred_samples.var(dim=1).mean().detach(),
    }



def sorted_logit_wasserstein(donkey_logits, trunk_topk_ids, trunk_topk_probs,
                              top_k_donkey=256):
    """ULD-style closed-form Wasserstein-1 between donkey and trunk distributions.

    Per Boizard et al. 2024 ("Towards Cross-Tokenizer Distillation"):
        W1 = sum_i |donkey_sorted[i] - trunk_sorted[i]|
    where both distributions are sorted in decreasing order. This is the
    closed-form Wasserstein-1 between two discrete distributions over the
    same ordered support (rank).

    Since trunk's distribution is given as top-K (the rest is zero),
    we sort donkey's softmax and compare its top-K_donkey to trunk's top-K_trunk
    via sorted-rank alignment, padding the shorter with zeros.

    Args:
        donkey_logits:    [N, V]    donkey's full vocab logits at each slot
        trunk_topk_ids:   [N, K_t]  trunk's top-K_t token IDs (not used here;
                                    we only need probs since we're aligning by rank)
        trunk_topk_probs: [N, K_t]  trunk's top-K_t probs (sum to 1.0 per row)
        top_k_donkey:     int       how many top donkey tokens to consider

    Returns:
        dict with 'total' (scalar W1) and 'per_pos' [N] distances.
    """
    import torch.nn.functional as F
    N, V = donkey_logits.shape
    K_t = trunk_topk_probs.shape[-1]

    # Donkey probs, sorted descending — take top K_donkey
    donkey_probs = F.softmax(donkey_logits.float(), dim=-1)
    donkey_top, _ = donkey_probs.topk(top_k_donkey, dim=-1)         # [N, K_donkey]

    # Trunk probs sorted descending (they should already be, but enforce)
    trunk_sorted, _ = trunk_topk_probs.float().sort(dim=-1, descending=True)  # [N, K_t]

    # Pad both to the same length so we can compute |donkey_sorted - trunk_sorted|
    K_max = max(top_k_donkey, K_t)
    if top_k_donkey < K_max:
        donkey_top = F.pad(donkey_top, (0, K_max - top_k_donkey), value=0.0)
    if K_t < K_max:
        trunk_sorted = F.pad(trunk_sorted, (0, K_max - K_t), value=0.0)

    # Closed-form W1 between two sorted discrete distributions on rank-aligned support
    w1_per_pos = (donkey_top - trunk_sorted).abs().sum(dim=-1)      # [N]

    return {
        "total":   w1_per_pos.mean(),
        "per_pos": w1_per_pos.detach(),
    }



def multimode_loss(pred_decoded, target_tokens, lm_head,
                   trunk_topk_ids=None, trunk_topk_probs=None,
                   lam_reg_l2=0.01, **_kwargs):
    """Per-mode-target multi-mode loss.

    Each of M modes at slot k trains to predict trunk's rank-m token at slot k.
    M=2 → mode 0 supervised on trunk argmax (rank 0), mode 1 on rank 1.

    This gives every mode an explicit, different supervision target.
    Diversity is natural (different targets), no L_div term needed.

    Args:
        pred_decoded:     [B, trunk_H, K, M]
        target_tokens:    [B, K]              — trunk argmax tokens
                          (mode 0's supervision target)
        trunk_topk_ids:   [B, K, K_topk]      — trunk's top-K_topk token IDs
                          (mode m's supervision target = trunk_topk_ids[:,:,m])
        trunk_topk_probs: [B, K, K_topk]      — (unused here; could weight CE)
        lam_reg_l2:       weight on output L2 norm regularizer

    Returns:
        dict with: total, mode_losses (list per-mode CE means),
                   reg, best_mode_acc, mode_collapse_frac.
    """
    import torch.nn.functional as F
    B, H, K, M = pred_decoded.shape
    V = lm_head.shape[0]

    pf = pred_decoded.permute(0, 2, 3, 1).reshape(B * K * M, H).float()
    logits_flat = pf @ lm_head.float().t()                            # [B*K*M, V]
    logits = logits_flat.view(B, K, M, V)

    # Build per-mode targets.
    # Fall back to argmax-replicated targets if topk_ids missing (legacy).
    if trunk_topk_ids is None:
        # Mode m is supervised on target_tokens[:, k] for all m (fallback).
        per_mode_target = target_tokens.unsqueeze(-1).expand(-1, -1, M)   # [B, K, M]
    else:
        # trunk_topk_ids: [B, K, K_topk]. Mode m supervises on rank m.
        K_topk = trunk_topk_ids.shape[-1]
        if M > K_topk:
            raise ValueError(f"M={M} but trunk_topk_ids has only {K_topk} ranks")
        per_mode_target = trunk_topk_ids[:, :, :M]                       # [B, K, M]

    # CE per (b, k, m): donkey at (b,k,m) vs trunk's rank-m token
    ce_per_mode = F.cross_entropy(
        logits.reshape(B * K * M, V),
        per_mode_target.reshape(B * K * M).long(),
        reduction="none",
    ).view(B, K, M)                                                  # [B, K, M]

    # Total CE = sum over modes, mean over (B, K).
    # Equal weight per mode; can introduce probability-weighted version later.
    L_modes = ce_per_mode.mean(dim=(0, 1))                            # [M]
    L_total_ce = L_modes.sum()

    L_reg = (pred_decoded.float() ** 2).mean()

    L_total = L_total_ce + lam_reg_l2 * L_reg

    # Diagnostics: best-mode acceptance (does ANY mode's argmax match trunk argmax?)
    with torch.no_grad():
        argmax_per_mode = logits.argmax(dim=-1)                       # [B, K, M]
        any_mode_correct = (argmax_per_mode == target_tokens.unsqueeze(-1)
                            ).any(dim=-1).float().mean()
        # mode_collapse_frac = fraction of (b,k) where all modes have same argmax
        mode_collapse_frac = (
            argmax_per_mode == argmax_per_mode[:, :, :1]
        ).all(dim=-1).float().mean()
        # mode 0 alone accuracy (for comparison to single-mode baseline)
        mode0_acc = (argmax_per_mode[:, :, 0] == target_tokens).float().mean()

    out = {
        "total": L_total,
        "any":   L_total_ce.detach(),     # alias for backward compat with metrics
        "div":   torch.zeros((), device=L_total.device),  # placeholder for log
        "reg":   L_reg.detach(),
        "best_mode_acc":       any_mode_correct.detach(),
        "mode0_acc":           mode0_acc.detach(),
        "mode_collapse_frac":  mode_collapse_frac.detach(),
    }
    # Per-mode CE for diagnostics
    for m_idx in range(M):
        out[f"ce_mode_{m_idx}"] = L_modes[m_idx].detach()
    return out
