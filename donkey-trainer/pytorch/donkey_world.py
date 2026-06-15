"""Donkey v2 PyTorch reference — windowed world-model drafter.

Architecture mirrors what the Swift orchestrator will run on ANE. Every size
is derived from DonkeyConfig; nothing is hardcoded.

Per-call data flow:
  1. history input: [W] vectors of trunk.hidden_dim (cold-start: replicate h[0])
  2. HistoryProjection: trunk.hidden_dim -> architecture.hidden_dim (linear1x1)
  3. concat draft-slot query embeddings (K learned vectors)
  4. positional bias added per slot
  5. N transformer layers (RMSNorm + multi-head SDPA + RMSNorm + SiLU FFN)
  6. final RMSNorm
  7. output head: architecture.hidden_dim -> (trunk.hidden_dim + out_conf_dim)
  8. split: pred_hidden [W+K, trunk_hidden] | confidence [W+K, out_conf]
  9. return only the K draft-slot outputs (positions W..W+K-1)

Loss at training time (computed externally, not here): L2 + lambda*(1-cos)
on pred_hidden vs trunk.lastHiddenState[t+1..t+K], mu*CE through lm_head,
nu*calibration on confidence (all training-time only).
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from donkey_config import DonkeyConfig


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., C, SP]. Reduces over C (channel). gamma: [C]."""
    ms = (x * x).mean(dim=-2, keepdim=True)
    rrms = torch.rsqrt(ms + eps)
    return x * rrms * gamma.view(-1, 1)


def causal_mask(sp: int, device, dtype) -> torch.Tensor:
    """Additive mask [1, 1, sp, sp]: 0 at and below diagonal, -65504 above."""
    m = torch.full((sp, sp), -65504.0, device=device, dtype=dtype)
    return torch.tril(torch.zeros_like(m)).masked_fill(
        torch.triu(torch.ones_like(m, dtype=torch.bool), diagonal=1), -65504.0
    ).view(1, 1, sp, sp)


class HistoryProjection(nn.Module):
    """Project trunk hidden states to donkey's working dim, then LayerNorm.

    Input:  x [B, trunk_hidden, SP]   row-major [C, SP] per donkey convention
    Weight: W [hidden_dim, trunk_hidden]
    Output: y [B, hidden_dim, SP]     y = LN(W @ x) over the channel axis

    Channel-major LayerNorm: normalize each [hidden_dim] vector at each
    SP position (subtract per-position mean across channels, divide by
    per-position std across channels). Affine gamma/beta of shape
    [hidden_dim] applied per channel.

    The LayerNorm is what makes donkey's latent unit-ish-scale so SIGReg
    and Huber both operate in well-conditioned regimes. Without it, trunk
    hidden-state norms in the 100-1e6 range (massive activations) feed
    directly into the regression objective.
    """
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(cfg.architecture.hidden_dim,
                                          cfg.trunk.hidden_dim))
        self.ln_gamma = nn.Parameter(torch.ones(cfg.architecture.hidden_dim))
        self.ln_beta  = nn.Parameter(torch.zeros(cfg.architecture.hidden_dim))
        self.ln_eps = float(cfg.architecture.rms_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.W @ x                                       # [..., D, SP]
        # LayerNorm over channel dim (dim=-2 = D for [..., D, SP] layout).
        mean = y.mean(dim=-2, keepdim=True)
        var  = y.var(dim=-2, keepdim=True, unbiased=False)
        y = (y - mean) / torch.sqrt(var + self.ln_eps)
        # Broadcast gamma/beta over batch + SP, multiply on channel axis.
        # ln_gamma: [D] -> [D, 1] for [..., D, SP] layout
        y = y * self.ln_gamma.unsqueeze(-1) + self.ln_beta.unsqueeze(-1)
        return y


class DonkeyAttention(nn.Module):
    """Multi-head causal self-attention.

    Input:  x [B, hidden_dim, SP]
    Output: y [B, hidden_dim, SP]
    """
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        D, H, HD = cfg.architecture.hidden_dim, cfg.architecture.heads, cfg.architecture.head_dim
        self.Wq = nn.Parameter(torch.zeros(D, D))
        self.Wk = nn.Parameter(torch.zeros(D, D))
        self.Wv = nn.Parameter(torch.zeros(D, D))
        self.Wo = nn.Parameter(torch.zeros(D, D))
        self.heads, self.head_dim, self.dim = H, HD, D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, SP = x.shape
        H, HD = self.heads, self.head_dim
        q = self.Wq @ x; k = self.Wk @ x; v = self.Wv @ x          # [B, D, SP] each
        # Reshape to [B, H, HD, SP] -> [B, H, SP, HD] for attention.
        q = q.view(B, H, HD, SP).transpose(2, 3)
        k = k.view(B, H, HD, SP).transpose(2, 3)
        v = v.view(B, H, HD, SP).transpose(2, 3)
        scale = 1.0 / math.sqrt(HD)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale       # [B, H, SP, SP]
        scores = scores + causal_mask(SP, x.device, x.dtype)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                  # [B, H, SP, HD]
        # Back to [B, D, SP]:
        out = out.transpose(2, 3).contiguous().view(B, D, SP)
        return self.Wo @ out


class DonkeyFFN(nn.Module):
    """SiLU-activated FFN.  hidden_dim -> ffn_dim -> hidden_dim."""
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        D, F_ = cfg.architecture.hidden_dim, cfg.architecture.ffn_dim
        self.W_up = nn.Parameter(torch.zeros(F_, D))
        self.W_down = nn.Parameter(torch.zeros(D, F_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W_up @ x
        h = h * torch.sigmoid(h)  # silu
        return self.W_down @ h


class DonkeyLayer(nn.Module):
    """One transformer layer: pre-norm attention + residual + pre-norm FFN + residual."""
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        D = cfg.architecture.hidden_dim
        self.gamma_att = nn.Parameter(torch.ones(D))
        self.attn = DonkeyAttention(cfg)
        self.gamma_ffn = nn.Parameter(torch.ones(D))
        self.ffn = DonkeyFFN(cfg)
        self.rms_eps = cfg.architecture.rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x, self.gamma_att, self.rms_eps))
        x = x + self.ffn(rmsnorm(x, self.gamma_ffn, self.rms_eps))
        return x


class DonkeyWorldRef(nn.Module):
    """Full donkey v2 world-model drafter (PyTorch reference).

    Parameter shapes are entirely a function of DonkeyConfig. The Swift
    orchestrator will instantiate kernels of these exact shapes from the same
    config; trained weights round-trip via safetensors.
    """
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.architecture.hidden_dim
        SP = cfg.sequence.spatial_pad

        self.input_proj = HistoryProjection(cfg)
        # Learned query embeddings for the K draft slots.
        # Stored as [D, K] for natural broadcast into channel-major layout.
        # M modes per slot (default M=1 = legacy single-mode K=3).
        # Read from cfg.architecture.m_modes if set, default 1.
        M = getattr(cfg.architecture, "m_modes", 1)
        self.m_modes = M
        self.draft_queries = nn.Parameter(
            torch.zeros(D, cfg.sequence.draft_size * M))
        # Positional bias added to all SP positions (learned).
        self.pos_bias = nn.Parameter(torch.zeros(D, SP))

        self.layers = nn.ModuleList(
            [DonkeyLayer(cfg) for _ in range(cfg.architecture.n_layers)])
        self.gamma_final = nn.Parameter(torch.ones(D))
        # Output head: D -> trunk_hidden + out_conf.
        out_ch = cfg.trunk.hidden_dim + cfg.architecture.out_conf_dim
        self.W_head = nn.Parameter(torch.zeros(out_ch, D))

    def forward(self, history: torch.Tensor) -> tuple:
        """
        Args:
            history: [trunk_hidden, W]  the W most recent trunk lastHiddenStates,
                     channel-major.

        Returns:
            (pred_hidden [trunk_hidden, K], confidence [K])  - the K draft slots.
        """
        cfg = self.cfg
        D = cfg.architecture.hidden_dim
        W = cfg.sequence.window_size
        K = cfg.sequence.draft_size
        SP = cfg.sequence.spatial_pad
        assert history.shape == (cfg.trunk.hidden_dim, W), \
            f"history shape {history.shape} != ({cfg.trunk.hidden_dim}, {W})"

        # Add batch dim for matmul compatibility with the modules.
        h = history.unsqueeze(0)                                 # [1, trunk_hidden, W]
        hp = self.input_proj(h)                                  # [1, D, W]

        # Build full SP-wide input: history slots 0..W-1, draft slots W..W+K-1.
        # Slots W+K..SP-1 are padding (zero), masked-out by causal mask anyway.
        x = torch.zeros(1, D, SP, device=hp.device, dtype=hp.dtype)
        x[:, :, :W] = hp[0]
        x[:, :, W:W+K] = self.draft_queries
        x = x + self.pos_bias.unsqueeze(0)

        for layer in self.layers:
            x = layer(x)

        x = rmsnorm(x, self.gamma_final, cfg.architecture.rms_eps)
        head_out = self.W_head @ x                                # [1, out_ch, SP]

        pred_full = head_out[0, :cfg.trunk.hidden_dim, :]         # [trunk_hidden, SP]
        conf_full = head_out[0, cfg.trunk.hidden_dim:, :]         # [out_conf, SP]

        # Return only the K draft-slot outputs.
        pred_draft = pred_full[:, W:W+K]                          # [trunk_hidden, K]
        conf_draft = torch.sigmoid(conf_full[0, W:W+K])           # [K]
        return pred_draft, conf_draft

    def init_for_training(self, seed: int = 42) -> None:
        """Initialize weights for PyTorch training.

        Swift loads safetensors directly, so deployed donkey never sees these
        values. Training only. Standard transformer init: Kaiming uniform on
        linears (PyTorch nn.Linear default), small-normal on learned
        embeddings/positions, leave RMSNorm gammas at 1.
        """
        import math
        g = torch.Generator(device="cpu").manual_seed(seed)

        def kaiming_(p):
            # PyTorch nn.Linear default: kaiming_uniform_ with a=sqrt(5),
            # which matches what we want for the linear-style 2D matrices.
            fan_in = p.shape[-1]
            bound = math.sqrt(6.0 / fan_in) / math.sqrt(3.0)  # uniform std
            with torch.no_grad():
                tmp = torch.empty(p.shape).uniform_(-bound, bound, generator=g)
                p.copy_(tmp)

        kaiming_(self.input_proj.W)
        kaiming_(self.W_head)
        for layer in self.layers:
            kaiming_(layer.attn.Wq)
            kaiming_(layer.attn.Wk)
            kaiming_(layer.attn.Wv)
            kaiming_(layer.attn.Wo)
            kaiming_(layer.ffn.W_up)
            kaiming_(layer.ffn.W_down)

        with torch.no_grad():
            tmp = torch.empty(self.draft_queries.shape).normal_(0.0, 0.02, generator=g)
            self.draft_queries.copy_(tmp)
            tmp = torch.empty(self.pos_bias.shape).normal_(0.0, 0.02, generator=g)
            self.pos_bias.copy_(tmp)
        # gamma_final, layers.*.gamma_att, layers.*.gamma_ffn stay at 1.0

    def forward_for_training(self, history: torch.Tensor) -> dict:
        """Batched training-mode forward for v3 JEPA loss.

        Returns the internal 1024-d latent (gradient enabled for JEPA terms)
        and the trunk-space decoded prediction (computed with detached
        latent, so its gradient flows only into the decoder W_head).

        Args:
            history: [B, trunk_hidden, W]

        Returns:
            {z_pred:       [B, D, K]   latent;  grad -> input_proj, layers,
             pred_decoded: [B, trunk_hidden, K]  decoded;  grad -> W_head only}
        """
        cfg = self.cfg
        D = cfg.architecture.hidden_dim
        W = cfg.sequence.window_size
        K = cfg.sequence.draft_size
        SP = cfg.sequence.spatial_pad
        B = history.shape[0]
        assert history.shape == (B, cfg.trunk.hidden_dim, W), \
            f"history {tuple(history.shape)} != (B, {cfg.trunk.hidden_dim}, {W})"

        hp = self.input_proj(history)
        x = torch.zeros(B, D, SP, device=hp.device, dtype=hp.dtype)
        x[:, :, :W] = hp
        M = self.m_modes
        KM = K * M
        assert SP >= W + KM, (
            f"spatial_pad={SP} too small for W={W} + K*M={KM}; "
            f"increase spec.sequence.spatial_pad")
        x[:, :, W:W+KM] = self.draft_queries.unsqueeze(0).expand(B, -1, -1)
        x = x + self.pos_bias.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = rmsnorm(x, self.gamma_final, cfg.architecture.rms_eps)

        # z_pred shape: [B, D, K*M]; reshape for downstream
        z_pred_flat = x[:, :, W:W+KM]                                  # [B, D, K*M]
        z_pred_d = z_pred_flat.detach()
        if M == 1:
            head_out = torch.einsum("od,bdk->bok", self.W_head, z_pred_d)
            pred_decoded = head_out[:, :cfg.trunk.hidden_dim, :]      # [B, trunk_H, K]
            z_pred = z_pred_flat                                       # [B, D, K]
        else:
            head_out = torch.einsum(
                "od,bdkm->bokm", self.W_head, z_pred_d.view(B, D, K, M))
            pred_decoded = head_out[:, :cfg.trunk.hidden_dim, :, :]    # [B, trunk_H, K, M]
            z_pred = z_pred_flat.view(B, D, K, M)
        return {"z_pred": z_pred, "pred_decoded": pred_decoded, "m_modes": M}


def fill_history_cold_start(h_first: torch.Tensor, real_history: list, W: int) -> torch.Tensor:
    """Build the W-slot history buffer with cold-start replication.

    Per donkey-v2 design: positions 0..(W - n - 1) replicate the very first
    hidden h_first (cold-start padding); positions (W - n - 1)..(W - 1) hold
    the n most recent real history entries. Once n >= W, only real entries
    fill the buffer.

    Args:
        h_first:      [trunk_hidden] - the first lastHiddenState ever seen
                      this session (used as cold-start padding)
        real_history: list of recent [trunk_hidden] tensors, oldest first.
                      May be shorter than W (cold-start) or longer (we take the
                      most recent W).
        W:            window size

    Returns:
        history: [trunk_hidden, W] channel-major.
    """
    n = min(len(real_history), W)
    D = h_first.shape[0]
    out = torch.zeros(D, W, dtype=h_first.dtype, device=h_first.device)
    # Cold-start padding.
    if n < W:
        for i in range(W - n):
            out[:, i] = h_first
    # Real recent history.
    if n > 0:
        recent = real_history[-n:]
        for i, h in enumerate(recent):
            out[:, W - n + i] = h
    return out



# ============================================================================
# Stochastic action-conditioned donkey (LeWM-faithful)
# ============================================================================

class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioned on a vector c.

    Replaces plain RMSNorm in the stochastic predictor. Same scale-shift trick
    used in DiT (Peebles & Xie). Critically: the output projection of the
    conditioning MLP is zero-initialized, so AdaLN starts as identity-on-ε
    and gradient learns when ε helps. This is the LeWM stability recipe.

    Input shapes (channel-major, [B, D, SP]):
        x:  [B, D, SP]
        c:  [B, D]  - conditioning vector (single per example; broadcast over SP)

    Output: [B, D, SP]
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma_base = nn.Parameter(torch.ones(dim))
        # Output projection (gamma, beta scale-shift) from conditioning vector.
        # Will be zero-initialized in init_for_training. Until then, no effect.
        # Small non-zero init lets gradient flow through both ε paths from step 1.
        # (Pure zero would freeze eps_encoder's contribution until proj first moves.)
        self.proj = nn.Parameter(torch.zeros(2 * dim, dim))
        # Will be filled in init_for_training with small std=0.01 normal.

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # RMSNorm with base gamma (will be modulated by adaptive scale below).
        ms = (x * x).mean(dim=-2, keepdim=True)
        rrms = torch.rsqrt(ms + self.eps)
        x_normed = x * rrms                                       # [B, D, SP]

        # Compute (γ_delta, β) from conditioning c. Shape: [B, 2D].
        # γ_delta: scale modulation (1 + γ_delta), β: shift.
        # Layout: c is [B, D], proj is [2D, D] → [B, 2D] = c @ proj.T
        gb = c @ self.proj.t()                                    # [B, 2D]
        gamma_delta, beta = gb.chunk(2, dim=-1)                   # each [B, D]

        # Apply: x * (gamma_base + γ_delta) + β, broadcasting over SP.
        # gamma_base: [D], gamma_delta: [B, D]. Effective gamma: [B, D].
        gamma_eff = (self.gamma_base.unsqueeze(0) + gamma_delta)  # [B, D]
        # x_normed: [B, D, SP], gamma_eff: [B, D, 1] for broadcast over SP.
        return x_normed * gamma_eff.unsqueeze(-1) + beta.unsqueeze(-1)


class DonkeyLayerStochastic(nn.Module):
    """One transformer layer with AdaLN-modulated attention and FFN."""
    def __init__(self, cfg: DonkeyConfig):
        super().__init__()
        D = cfg.architecture.hidden_dim
        self.adaln_att = AdaLN(D, eps=cfg.architecture.rms_eps)
        self.attn      = DonkeyAttention(cfg)
        self.adaln_ffn = AdaLN(D, eps=cfg.architecture.rms_eps)
        self.ffn       = DonkeyFFN(cfg)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.adaln_att(x, c))
        x = x + self.ffn(self.adaln_ffn(x, c))
        return x


class EpsilonEncoder(nn.Module):
    """Map ε (64-d random Gaussian) → conditioning vector c (D-d) for AdaLN.

    Two-layer MLP. Output projection zero-initialized (combined with AdaLN's
    zero-init proj, this gives identity-on-ε at step 0).
    """
    def __init__(self, eps_dim: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(hidden_dim, eps_dim))
        self.W2 = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        # eps: [B, eps_dim] → [B, hidden_dim]
        h = eps @ self.W1.t()           # [B, hidden_dim]
        h = h * torch.sigmoid(h)        # SiLU
        return h @ self.W2.t()          # [B, hidden_dim]


class DonkeyWorldStochastic(nn.Module):
    """Stochastic action-conditioned donkey (LeWM-faithful).

    Per call:
        donkey(history, ε) → ẑ_next   single-step prediction

    Inputs:
        history: [B, trunk_hidden, W]   history of W trunk hidden states
        ε:       [B, eps_dim]            random Gaussian per example

    Output:
        ẑ_pred: [B, trunk_hidden, 1]    single predicted next hidden

    Internally:
        1. Φ_O (HistoryProjection) maps trunk_hidden → D, applies LayerNorm
        2. Append ONE draft slot (learned query) at position W
        3. EpsilonEncoder maps ε → c (conditioning vector)
        4. N transformer layers with AdaLN(·, c) modulation
        5. Final norm + output head: D → trunk_hidden
        6. Return the single draft slot

    At inference for multi-step rollout: chain by feeding ẑ_pred back into
    history, generating fresh ε per step.
    """
    def __init__(self, cfg: DonkeyConfig, eps_dim: int = 64):
        super().__init__()
        self.cfg = cfg
        self.eps_dim = eps_dim
        D = cfg.architecture.hidden_dim
        W = cfg.sequence.window_size
        SP_eff = W + 1                                            # window + 1 draft slot

        self.input_proj = HistoryProjection(cfg)
        # Single draft query (no K=3 parallel slots).
        self.draft_query = nn.Parameter(torch.zeros(D, 1))
        # Positional bias for the W+1 effective positions.
        self.pos_bias = nn.Parameter(torch.zeros(D, SP_eff))

        self.eps_encoder = EpsilonEncoder(eps_dim, D)
        self.layers = nn.ModuleList(
            [DonkeyLayerStochastic(cfg) for _ in range(cfg.architecture.n_layers)])
        self.gamma_final = nn.Parameter(torch.ones(D))
        # Output head: D → trunk_hidden. Drop confidence head (not needed in
        # world-model framing — Wasserstein on the predicted distribution does
        # the calibration work).
        self.W_head = nn.Parameter(torch.zeros(cfg.trunk.hidden_dim, D))

    def forward(self, history: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [B, trunk_hidden, W]
            eps:     [B, eps_dim]
        Returns:
            z_pred:  [B, trunk_hidden, 1]
        """
        cfg = self.cfg
        D = cfg.architecture.hidden_dim
        W = cfg.sequence.window_size
        B = history.shape[0]
        SP_eff = W + 1

        # 1. Encode history.
        hp = self.input_proj(history)                            # [B, D, W]

        # 2. Build full input: W history slots + 1 draft slot at W.
        x = torch.zeros(B, D, SP_eff, device=hp.device, dtype=hp.dtype)
        x[:, :, :W] = hp
        x[:, :, W:W+1] = self.draft_query.unsqueeze(0).expand(B, -1, -1)
        x = x + self.pos_bias.unsqueeze(0)                       # broadcast over batch

        # 3. Conditioning vector from ε.
        c = self.eps_encoder(eps)                                # [B, D]

        # 4. Stochastic transformer layers.
        for layer in self.layers:
            x = layer(x, c)

        # 5. Final norm + projection.
        # Use plain rmsnorm here (not AdaLN — final norm is unconditional).
        ms = (x * x).mean(dim=-2, keepdim=True)
        x = x * torch.rsqrt(ms + cfg.architecture.rms_eps) * self.gamma_final.view(-1, 1)
        head_out = self.W_head @ x                                # [B, trunk_hidden, SP_eff]

        # 6. Return only the draft slot.
        return head_out[:, :, W:W+1]                              # [B, trunk_hidden, 1]

    def forward_n_samples(self, history: torch.Tensor,
                          eps_batch: torch.Tensor) -> torch.Tensor:
        """Convenience: run N ε samples for the same history.

        Args:
            history:    [B, trunk_hidden, W]
            eps_batch:  [B, N, eps_dim]
        Returns:
            z_preds:    [B, N, trunk_hidden]     (squeezed last dim)
        """
        B, N, _ = eps_batch.shape
        # Expand history to [B*N, ...] and ε to [B*N, eps_dim]
        history_exp = history.unsqueeze(1).expand(-1, N, -1, -1).reshape(
            B * N, history.shape[1], history.shape[2])
        eps_flat = eps_batch.reshape(B * N, -1)
        z_pred = self.forward(history_exp, eps_flat)             # [B*N, trunk_hidden, 1]
        return z_pred.squeeze(-1).view(B, N, -1)                  # [B, N, trunk_hidden]

    def init_for_training(self, seed: int = 42) -> None:
        """Initialize weights. Critically: ε path (eps_encoder.W2 + AdaLN.proj)
        is zero-initialized → AdaLN modulation = 0 at step 0 → predictor starts
        as identity-on-ε. Gradient learns when ε is useful.
        """
        import math
        g = torch.Generator(device="cpu").manual_seed(seed)

        def kaiming_(p):
            fan_in = p.shape[-1]
            bound = math.sqrt(6.0 / fan_in) / math.sqrt(3.0)
            with torch.no_grad():
                tmp = torch.empty(p.shape).uniform_(-bound, bound, generator=g)
                p.copy_(tmp)

        # Standard Kaiming on linear weights.
        kaiming_(self.input_proj.W)
        kaiming_(self.W_head)
        kaiming_(self.eps_encoder.W1)  # first layer of ε MLP: standard init
        kaiming_(self.eps_encoder.W2)  # standard init; AdaLN.proj does the zero-init
        for layer in self.layers:
            kaiming_(layer.attn.Wq)
            kaiming_(layer.attn.Wk)
            kaiming_(layer.attn.Wv)
            kaiming_(layer.attn.Wo)
            kaiming_(layer.ffn.W_up)
            kaiming_(layer.ffn.W_down)
            # AdaLN.proj: tiny init breaks gradient symmetry; effectively starts
            # near-identity but lets gradient flow both ways through ε from step 1.
            with torch.no_grad():
                # AdaLN.proj init: std=0.1 is large enough that different ε
                # vectors produce visibly different ẑ predictions at step 0.
                # Smaller (0.01) collapses to anchor; larger (1.0) destabilizes
                # the first few hundred steps.
                tmp = torch.empty(layer.adaln_att.proj.shape).normal_(0.0, 0.3, generator=g)
                layer.adaln_att.proj.copy_(tmp)
                tmp = torch.empty(layer.adaln_ffn.proj.shape).normal_(0.0, 0.3, generator=g)
                layer.adaln_ffn.proj.copy_(tmp)
            # AdaLN.gamma_base stays at 1.0 (default initialization)

        # Small init for learned queries/positions.
        with torch.no_grad():
            tmp = torch.empty(self.draft_query.shape).normal_(0.0, 0.02, generator=g)
            self.draft_query.copy_(tmp)
            tmp = torch.empty(self.pos_bias.shape).normal_(0.0, 0.02, generator=g)
            self.pos_bias.copy_(tmp)
