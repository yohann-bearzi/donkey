#!/usr/bin/env python3
"""Does the ACTION (emitted token) predict the next hidden state?
Variance decomposition: unconditional var of h_{t+1} vs var WITHIN action-token groups.
High 'variance explained' => action determines next hidden => donkey's core bet holds.
Read-only, memmap + subsample, safe while harvest runs.
Usage: python3 action_test.py [corpus]"""
import sys, os, numpy as np

BASE = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
DMODEL = 4096; STOP = (151645, 151643)
SUB = 300000   # cap rows materialized
MIN_COUNT = 30 # min transitions per action group to count it
rng = np.random.default_rng(0)
corpus = sys.argv[1] if len(sys.argv) > 1 else "humaneval"
T = os.path.join(BASE, "traces", corpus)

hb = os.path.getsize(os.path.join(T,"lastHiddenState.bin")); n_hid = hb//(4*DMODEL)
tok = np.fromfile(os.path.join(T,"tokens.bin"), np.int32)
pid = np.fromfile(os.path.join(T,"prompt_idx.bin"), np.int32)
n = min(n_hid, len(tok), len(pid)); tok, pid = tok[:n], pid[:n]
H = np.memmap(os.path.join(T,"lastHiddenState.bin"), np.float32, "r", shape=(n_hid,DMODEL))[:n]
valid = pid < pid[-1]

# build transitions: action = tok[t]; target = h[t+1]; within-prompt only
a, b = [], []
for p in np.unique(pid[valid]):
    idx = np.where(pid == p)[0]
    if len(idx) < 2: continue
    a.append(idx[:-1]); b.append(idx[1:])
src = np.concatenate(a); tgt = np.concatenate(b)
action = tok[src]                       # token emitted at t (the action advancing t->t+1)
# subsample
if len(src) > SUB:
    s = rng.choice(len(src), SUB, replace=False); src, tgt, action = src[s], tgt[s], action[s]
Hn = np.asarray(H[tgt], np.float64)     # next hiddens
# WHITEN per-dim: the raw space is dominated by a few huge token-IRRELEVANT directions
# (84% var in 5 PCs), which drown out token-relevant signal in summed-variance. Standardize
# each dim so every direction counts equally.
Hn = (Hn - Hn.mean(0, keepdims=True)) / (Hn.std(0, keepdims=True) + 1e-6)
print(f"[{corpus}] {len(src)} transitions sampled, {len(np.unique(action))} distinct actions (whitened)")

def var_explained(Hn, grp, min_count):
    # Restrict to rows whose group has >= min_count, then compute BOTH total and within
    # over THAT SAME row set (otherwise the ratio is meaningless / can go negative).
    uniq, inv = np.unique(grp, return_inverse=True)
    counts = np.bincount(inv)
    keep_groups = set(np.where(counts >= min_count)[0].tolist())
    mask = np.array([g in keep_groups for g in inv])
    if mask.sum() < min_count:
        return float("nan"), float("nan"), float("nan"), 0, 0
    Hk = Hn[mask]; invk = inv[mask]
    total = Hk.var(0).sum()                      # total over the KEPT rows
    wvar = 0.0; ntot = 0; ngrp = 0
    for gi in np.unique(invk):
        rows = Hk[invk == gi]
        wvar += rows.var(0).sum() * len(rows); ntot += len(rows); ngrp += 1
    within = wvar / ntot
    return total, within, 1 - within/total, ngrp, ntot

# 1) variance explained by the ACTION (emitted token)
tot, wit, expl, ng, nt = var_explained(Hn, action, MIN_COUNT)
print(f"\n[ACTION] var explained by emitted token: {expl*100:5.1f}%  "
      f"(total {tot:.0f} -> within-action {wit:.0f}; {ng} groups, {nt} rows)")

# 2) CONTROL: variance explained by a RANDOM grouping (same #groups) -> should be ~0
ng_use = max(2, ng)
randgrp = rng.integers(0, ng_use, size=len(action))
_, _, expl_rand, _, _ = var_explained(Hn, randgrp, MIN_COUNT)
print(f"[CONTROL] var explained by RANDOM grouping ({ng_use} groups): {expl_rand*100:5.1f}%  (should be ~0)")

# 3) variance explained by NEXT token (tok[t+1]) -- sanity: the hidden h_{t+1} produces tok[t+1]
#    via lm_head, so grouping by the token h_{t+1} GENERATES should explain a lot (upper-ish ref)
next_tok = tok[tgt]
_, _, expl_next, ngn, _ = var_explained(Hn, next_tok, MIN_COUNT)
print(f"[REF]     var explained by NEXT token tok[t+1]: {expl_next*100:5.1f}%  ({ngn} groups)")
print("          (h_{t+1} -> tok[t+1] via lm_head, so this is how much the hidden encodes its own token)")

# 4) TOKEN-SPACE: does the emitted token predict the NEXT token's distribution?
#    For each action group, measure entropy of next-token; compare to unconditional entropy.
def entropy(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())
nt_all = tok[tgt]
uniq_a, inv_a = np.unique(action, return_inverse=True)
ca = np.bincount(inv_a)
keep = np.where(ca >= MIN_COUNT)[0]
H_uncond = entropy(np.bincount(np.unique(nt_all, return_inverse=True)[1]))
wH = 0.0; ntk = 0
for gi in keep:
    rows = nt_all[inv_a == gi]
    wH += entropy(np.bincount(np.unique(rows, return_inverse=True)[1])) * len(rows); ntk += len(rows)
H_cond = wH / ntk if ntk else float("nan")
print(f"\n[TOKEN] next-token entropy: unconditional {H_uncond:.2f} bits -> "
      f"conditional on action {H_cond:.2f} bits  (reduction {H_uncond-H_cond:.2f} bits = action info)")

print("\n================= READING =================")
print("[ACTION] is the core donkey bet: does the EMITTED token (at t) predict the NEXT hidden?")
print("  >> CONTROL  => action carries real predictive signal; donkey should work.")
print("  ~ CONTROL   => emitted token alone doesn't determine next hidden; signal is in the")
print("                 window/context, not the immediate action -> lean on history, not action.")
print("[REF] high => the hidden strongly encodes the token it will emit (expected); a loose")
print("      upper reference for how concentrated next-hiddens are when you know their own token.")
