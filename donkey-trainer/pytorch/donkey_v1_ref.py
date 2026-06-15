"""Donkey v1 PyTorch reference.

Architectural constants must stay in lockstep with donkey-trainer/models/donkey_v1.h
and donkey-runtime/Sources/DonkeyRuntime/DonkeyForward.swift::DonkeyV1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DIM        = 1024
HIDDEN     = 4096
HEADS      = 16
HD         = 64
NLAYERS    = 2
SP         = 16
TRUNK_DIM  = 4096
N_TAPS     = 3
TAP_IN_TOTAL = N_TAPS * TRUNK_DIM
OUT_HIDDEN = 4096
OUT_CONF   = 1
OUT_CH     = OUT_HIDDEN + OUT_CONF
RMS_EPS    = 1e-6


def rmsnorm(x, gamma, eps=RMS_EPS):
    ms = (x * x).mean(dim=1, keepdim=True)
    rrms = torch.rsqrt(ms + eps)
    return x * rrms * gamma.view(1, -1, 1)


def silu(x):
    return x * torch.sigmoid(x)


def causal_mask(sp, device, dtype):
    m = torch.zeros(sp, sp, device=device, dtype=dtype)
    for r in range(sp):
        for c in range(sp):
            if c > r:
                m[r, c] = -65504.0
    return m.view(1, 1, sp, sp)


def sdpa(qkv, dim=DIM, heads=HEADS, hd=HD, sp=SP):
    q = qkv[:, 0:dim, 0, :]
    k = qkv[:, dim:2*dim, 0, :]
    v = qkv[:, 2*dim:3*dim, 0, :]
    q = q.view(1, heads, hd, sp).transpose(2, 3)
    k = k.view(1, heads, hd, sp).transpose(2, 3)
    v = v.view(1, heads, hd, sp).transpose(2, 3)
    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = causal_mask(sp, q.device, q.dtype)
    scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(2, 3).contiguous().view(1, dim, sp)
    return out


class DonkeyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_att = nn.Parameter(torch.ones(DIM))
        self.Wq = nn.Parameter(torch.zeros(DIM, DIM))
        self.Wk = nn.Parameter(torch.zeros(DIM, DIM))
        self.Wv = nn.Parameter(torch.zeros(DIM, DIM))
        self.Wo = nn.Parameter(torch.zeros(DIM, DIM))
        self.gamma_ffn = nn.Parameter(torch.ones(DIM))
        self.W_up   = nn.Parameter(torch.zeros(HIDDEN, DIM))
        self.W_down = nn.Parameter(torch.zeros(DIM, HIDDEN))

    def linear(self, W, x):
        return W @ x

    def forward(self, x):
        residual = x
        xn = rmsnorm(x, self.gamma_att)
        q = self.linear(self.Wq, xn)
        k = self.linear(self.Wk, xn)
        v = self.linear(self.Wv, xn)
        qkv = torch.cat([q, k, v], dim=1).unsqueeze(2)
        attn = sdpa(qkv)
        oo = self.linear(self.Wo, attn)
        x = residual + oo
        residual = x
        xn = rmsnorm(x, self.gamma_ffn)
        h_up = self.linear(self.W_up, xn)
        h_silu = silu(h_up)
        h_down = self.linear(self.W_down, h_silu)
        x = residual + h_down
        return x


class DonkeyV1Ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_tap = nn.Parameter(torch.zeros(DIM, TAP_IN_TOTAL))
        self.layers = nn.ModuleList([DonkeyLayer() for _ in range(NLAYERS)])
        self.gamma_final = nn.Parameter(torch.ones(DIM))
        self.W_head = nn.Parameter(torch.zeros(OUT_CH, DIM))

    def forward(self, tap_lo, tap_mid, tap_hi):
        taps = torch.cat([tap_lo, tap_mid, tap_hi], dim=0).unsqueeze(0)
        x = self.W_tap @ taps
        for layer in self.layers:
            x = layer(x)
        xn = rmsnorm(x, self.gamma_final)
        head_out = self.W_head @ xn
        pred_hidden = head_out[0, :OUT_HIDDEN, :]
        confidence  = torch.sigmoid(head_out[0, OUT_HIDDEN, :])
        return pred_hidden, confidence
