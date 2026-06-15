#!/usr/bin/env python3
"""Verify harvested-hidden format/alignment correctness AND characterize correlation structure.
Read-only, no model load, memmap + subsample (safe while harvest runs).
Usage: python3 verify_hiddens.py [corpus]   (default: all completed)"""
import sys, os, numpy as np

BASE = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
DMODEL = 4096; STOP = (151645, 151643)
SUB = 100000  # max rows materialized per stat (subsample for big corpora)
rng = np.random.default_rng(0)

def load(corpus):
    T = os.path.join(BASE, "traces", corpus)
    if not os.path.exists(os.path.join(T, "tokens.bin")): return None
    hb = os.path.getsize(os.path.join(T, "lastHiddenState.bin")); n_hid = hb // (4*DMODEL)
    tok = np.fromfile(os.path.join(T,"tokens.bin"), np.int32)
    pid = np.fromfile(os.path.join(T,"prompt_idx.bin"), np.int32)
    n = min(n_hid, len(tok), len(pid)); tok, pid = tok[:n], pid[:n]
    H = np.memmap(os.path.join(T,"lastHiddenState.bin"), np.float32, "r", shape=(n_hid,DMODEL))[:n]
    return H, tok, pid, n_hid

def sub(idx):
    if len(idx) <= SUB: return idx
    return idx[rng.choice(len(idx), SUB, replace=False)]

def within_prompt_pairs(pid, k, valid):
    a, b = [], []
    for p in np.unique(pid[valid]):
        idx = np.where(pid == p)[0]
        if len(idx) > k: a.append(idx[:-k]); b.append(idx[k:])
    return np.concatenate(a), np.concatenate(b)

def pearson_perdim(H, i0, i1):
    s = sub(np.arange(len(i0)))
    z0 = np.asarray(H[i0[s]], np.float64); z1 = np.asarray(H[i1[s]], np.float64)
    z0c = z0-z0.mean(0); z1c = z1-z1.mean(0)
    r = (z0c*z1c).sum(0)/(np.sqrt((z0c**2).sum(0)*(z1c**2).sum(0))+1e-9)
    return float(r.mean()), float(r.std())

def cosine_consec(H, i0, i1):
    s = sub(np.arange(len(i0)))
    a = np.asarray(H[i0[s]], np.float64); b = np.asarray(H[i1[s]], np.float64)
    num = (a*b).sum(1); den = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9
    return float((num/den).mean()), float((num/den).std())

def analyze(corpus):
    r = load(corpus)
    if r is None: print(f"\n##### {corpus}: no traces #####"); return
    H, tok, pid, n_hid = r
    valid = pid < pid[-1]   # drop last (possibly mid-write)
    print(f"\n##### {corpus}: {len(pid)} positions, {len(np.unique(pid))} prompts (n_hid={n_hid}) #####")

    # --- FORMAT / ALIGNMENT CHECKS ---
    print("[FORMAT]")
    # 1. stream lengths consistent
    print(f"  lengths: hiddens {n_hid}, tokens {len(tok)}, pid {len(pid)}  "
          f"{'OK aligned' if n_hid>=len(tok) else 'MISMATCH'}")
    # 2. prompt_idx monotonic non-decreasing (generation order)
    mono = np.all(np.diff(pid) >= 0)
    print(f"  prompt_idx monotonic non-decreasing: {mono}  (False => stream not in order!)")
    # 3. each prompt contiguous (no interleaving)
    contig = len(np.unique(pid)) == (np.sum(np.diff(pid) != 0) + 1)
    print(f"  prompts contiguous (no interleave): {contig}")
    # 4. hiddens finite, not all-zero, not duplicated consecutively
    samp = sub(np.where(valid)[0])
    hs = np.asarray(H[samp], np.float64)
    print(f"  finite: {np.all(np.isfinite(hs))}  allzero rows: {int(np.all(hs==0,axis=1).sum())}")
    p0 = np.where(pid == pid[0])[0]
    dupfrac = np.mean([np.allclose(H[p0[i]], H[p0[i+1]]) for i in range(min(100,len(p0)-1))]) if len(p0)>1 else 0
    print(f"  exact-duplicate consecutive (prompt0): {dupfrac:.2f}  (should be 0)")
    # 5. terminal tokens: do prompts end in STOP? (sanity on segmentation)
    ends = [int(tok[np.where(pid==p)[0][-1]]) for p in np.unique(pid[valid])[:200]]
    stopfrac = np.mean([e in STOP for e in ends])
    print(f"  prompts ending in STOP (first 200): {stopfrac*100:.0f}%  (cap-hits lower this)")

    # --- CORRELATION STRUCTURE (lag-k, both pearson & cosine) ---
    print("[CORRELATION] within-prompt, lag-k:")
    for k in (1, 2, 4, 8, 16):
        try:
            i0, i1 = within_prompt_pairs(pid, k, valid)
            pm, ps = pearson_perdim(H, i0, i1)
            cm, cs = cosine_consec(H, i0, i1)
            print(f"  lag {k:2d}: pearson {pm:+.3f}(±{ps:.3f})  cosine {cm:+.3f}(±{cs:.3f})  npairs={len(i0)}")
        except ValueError:
            print(f"  lag {k:2d}: too few pairs")
    # control: SHUFFLED pairs (should be ~0) — confirms our ~0 isn't a bug in the estimator
    i0, i1 = within_prompt_pairs(pid, 1, valid)
    sh = i1.copy(); rng.shuffle(sh)
    pm, _ = pearson_perdim(H, i0, sh)
    print(f"  lag 1 SHUFFLED control: pearson {pm:+.3f}  (must be ~0; if lag1 real==shuffled, no structure)")

    # --- GEOMETRY: is the high cosine a single shared mean, or multi-PC slow structure? ---
    print("[GEOMETRY] consecutive cosine after removing shared structure:")
    s = sub(np.where(valid)[0])
    Hs = np.asarray(H[s], np.float64)
    mean_vec = Hs.mean(0, keepdims=True)
    # build consecutive within-prompt pairs restricted to sampled rows is complex; instead
    # measure consecutive cosine on a contiguous slice of one large prompt, raw vs centered vs PC-removed
    # pick a TYPICAL terminated prompt (median length, ends in STOP) — NOT the runaway outlier
    upids = np.unique(pid[valid])
    lens = {int(p): int(np.sum(pid == p)) for p in upids}
    terminated = [p for p in upids if int(tok[np.where(pid == p)[0][-1]]) in STOP]
    cand = terminated if terminated else list(upids)
    med = np.median([lens[p] for p in cand])
    longest = int(min(cand, key=lambda p: abs(lens[p] - med)))   # closest-to-median terminated prompt
    pidx = np.where(pid == longest)[0]
    Hp = np.asarray(H[pidx], np.float64)
    def consec_cos(M):
        a, b = M[:-1], M[1:]
        num = (a*b).sum(1); den = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9
        return float((num/den).mean())
    raw = consec_cos(Hp)
    cen = consec_cos(Hp - mean_vec)                      # remove GLOBAL mean (corpus-level)
    # remove top-k PCs (computed on the sampled corpus rows, applied to this prompt)
    Hc = Hs - mean_vec
    try:
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        def rm_topk(M, k):
            Mc = M - mean_vec
            return Mc - (Mc @ Vt[:k].T) @ Vt[:k]
        pc1 = consec_cos(rm_topk(Hp, 1)); pc5 = consec_cos(rm_topk(Hp, 5))
        var_top5 = float((Vt.shape[0] and (np.linalg.svd(Hc, compute_uv=False)[:5]**2).sum() /
                          (np.linalg.svd(Hc, compute_uv=False)**2).sum()))
    except Exception as e:
        pc1 = pc5 = float("nan"); var_top5 = float("nan")
    cen_local = consec_cos(Hp - Hp.mean(0, keepdims=True))   # subtract THIS prompt's own mean
    print(f"  typical terminated prompt {longest} ({len(pidx)} pos, median-length):")
    print(f"    raw cosine            {raw:+.3f}")
    print(f"    after center (global) {cen:+.3f}   (corpus mean)")
    print(f"    after center (local)  {cen_local:+.3f}   (this prompt's mean; if ~0 => per-prompt offset dominates)")
    print(f"    after remove top-1 PC {pc1:+.3f}")
    print(f"    after remove top-5 PC {pc5:+.3f}   (if center fails but this ~0 => multi-PC; whiten)")
    print(f"    top-5 PCs explain {var_top5*100:.1f}% of variance")

corpora = [sys.argv[1]] if len(sys.argv) > 1 else ["humaneval","mbpp","codealpaca_20k"]
for c in corpora: analyze(c)
print("\n================= READING =================")
print("FORMAT: monotonic+contiguous+finite+no-dups => stream is correct generation order.")
print("CORRELATION: if lag1 pearson ~ shuffled control ~0 across ALL corpora => uncorrelation is")
print("  real & universal (next hidden ~orthogonal to current). cosine confirms (scale-invariant).")
print("  If cosine >> pearson, there's a shared direction (mean component) worth centering out.")
