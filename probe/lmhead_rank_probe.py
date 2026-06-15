#!/usr/bin/env python3
"""Estimate lm_head effective rank + test low-rank token recovery on real harvested hiddens.
Loads norm+lm_head from shards (NOT the trunk). Compares rank-r lm_head's argmax to the full
lm_head's argmax (and to the actually-emitted token) on real hiddens. The rank where token
agreement plateaus = the effective rank for the spectral/low-rank Channel-2 head.
Usage: python3 lmhead_rank_probe.py [corpus] [n_hiddens]"""
import sys, os, json, glob, time
import numpy as np

BASE  = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
MODEL = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
DMODEL = 4096
CORPORA = {"humaneval":"humaneval","mbpp":"mbpp","codealpaca":"codealpaca_20k"}
corpus = sys.argv[1] if len(sys.argv) > 1 else "mbpp"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

import mlx.core as mx

# --- load norm + lm_head from shards ---
print("[load] scanning shards for norm.weight + lm_head.weight ...")
cfg = json.load(open(os.path.join(MODEL, "config.json")))
eps = float(cfg.get("rms_norm_eps", 1e-6))
norm_w = lm_w = emb_w = None
for s in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
    w = mx.load(s)
    for k, v in w.items():
        if k.endswith("model.norm.weight") or k == "norm.weight": norm_w = v
        if k.endswith("lm_head.weight"): lm_w = v
        if k.endswith("embed_tokens.weight"): emb_w = v
if lm_w is None: lm_w = emb_w
assert lm_w is not None and norm_w is not None, "couldn't find lm_head/norm in shards"
lm = np.array(lm_w.astype(mx.float32))      # [vocab, dmodel]
nw = np.array(norm_w.astype(mx.float32))    # [dmodel]
VOCAB = lm.shape[0]
print(f"[load] lm_head {lm.shape}  vocab={VOCAB}  dmodel={lm.shape[1]}")

# --- SVD spectrum of lm_head (the expensive step: 4096 x ~150k) ---
print("[svd] computing singular values of lm_head (this is the slow part) ...")
t0 = time.time()
# economy SVD; we only need singular values + the right factors for low-rank reconstruction.
# lm is [vocab, dmodel]; SVD gives U[vocab,dmodel] S[dmodel] Vt[dmodel,dmodel].
U, S, Vt = np.linalg.svd(lm, full_matrices=False)
print(f"[svd] done in {time.time()-t0:.1f}s; {len(S)} singular values")
energy = (S**2).cumsum() / (S**2).sum()
def rank_for(e): return int(np.searchsorted(energy, e)) + 1
print("[svd] lm_head effective rank by energy:")
for e in [0.9, 0.95, 0.99, 0.999, 0.9999]:
    print(f"   {e*100:6.2f}% energy -> rank {rank_for(e):5d}  (of {len(S)})")

# --- load real harvested hiddens + their emitted tokens ---
d = os.path.join(BASE, "traces", CORPORA[corpus])
nbytes = os.path.getsize(os.path.join(d, "lastHiddenState.bin"))
n_hid = nbytes // (4 * DMODEL)
tok = np.fromfile(os.path.join(d, "tokens.bin"), np.int32)
H_mm = np.memmap(os.path.join(d, "lastHiddenState.bin"), np.float32, "r", shape=(n_hid, DMODEL))
rng = np.random.default_rng(0)
idx = rng.choice(min(n_hid, len(tok)), min(N, n_hid), replace=False)
H = np.array(H_mm[idx], np.float32)         # real pre-norm hiddens
emitted = tok[idx]                          # the token actually emitted at each (sampled, temp=1)
print(f"[data] {len(H)} real hiddens from {corpus}")

def apply_norm(h):  # MiMo final RMSNorm then lm_head expects normed hidden
    return h * (1.0 / np.sqrt((h*h).mean(-1, keepdims=True) + eps)) * nw

Hn = apply_norm(H)                          # [N, dmodel]
full_logits = Hn @ lm.T                      # [N, vocab]  (full lm_head)
full_argmax = np.argmax(full_logits, axis=1)

# --- low-rank token agreement: rank-r lm_head argmax vs full argmax AND vs emitted token ---
print("\n[rank test] token agreement of rank-r lm_head:")
print(f"{'rank':>6}{'vs_full_argmax%':>16}{'vs_emitted%':>14}{'top5_vs_emit%':>15}")
# full vs emitted baseline (argmax!=sampled token at temp=1 sometimes)
base_emit = (full_argmax == emitted).mean()
for r in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    if r > len(S): break
    # rank-r reconstruction: project hidden onto top-r right singular dirs, then to vocab
    # logits_r = Hn @ Vt_r.T @ diag(S_r) @ U_r.T  == (Hn @ Vr) @ (Sr * Ur.T)
    Vr = Vt[:r].T                            # [dmodel, r]
    coeff = Hn @ Vr                          # [N, r]  <-- the SPECTRAL compression (r numbers/hidden)
    logits_r = (coeff * S[:r]) @ U[:, :r].T  # [N, vocab]
    am = np.argmax(logits_r, axis=1)
    agree_full = (am == full_argmax).mean()
    agree_emit = (am == emitted).mean()
    top5 = np.argpartition(-logits_r, 5, axis=1)[:, :5]
    top5_emit = np.mean([emitted[i] in top5[i] for i in range(len(am))])
    print(f"{r:>6}{agree_full*100:>16.1f}{agree_emit*100:>14.1f}{top5_emit*100:>15.1f}")
print(f"\n[baseline] full lm_head argmax == emitted token: {base_emit*100:.1f}%")
print("  (gap from 100% = temp=1 sampling picked a non-argmax token; argmax!=sampled)")

print("\n================= READING =================")
print("- lm_head 'effective rank' = where SVD energy saturates (the 99%/99.9% rows).")
print("- 'vs_full_argmax' = does rank-r preserve the SAME top token as full lm_head. The rank")
print("  where THIS plateaus near 100% = the rank your spectral/low-rank Channel-2 head needs.")
print("- 'vs_emitted' caps at the baseline (full lm_head vs emitted), since temp=1 sampled non-argmax.")
print("- If plateau rank << dmodel (4096), the low-rank head is much smaller than full lm_head:")
print("  params = dmodel*r (project) + r*vocab (expand) vs dmodel*vocab full.")
