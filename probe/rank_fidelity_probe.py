#!/usr/bin/env python3
"""
rank_fidelity_probe.py -- find the rank at which a low-rank approximation of lm_head's ACTION
reproduces the true next-token DISTRIBUTION faithfully enough, ON THE DONKEY'S HIDDEN MANIFOLD.

Sizes the lm_head-free head. lm_head is globally full-rank (~3900 for 99% logit energy), but its
ACTION on the narrow manifold of hiddens the donkey produces may be far lower rank. We measure
rank vs three fidelity criteria that matter: JSD(q_r,p) (training loss), capture@X% (branch
selection: does q_r's top-X%-mass set contain the true emitted token), and set-size calibration.

Fast: avoids the giant [n,vocab] SVD (LAPACK SVD poorly threaded) via the Gram trick --
eig the small [n,n] (size independent of vocab), all heavy steps are all-core BLAS matmuls.

Usage:
  MIMO_DIR=... DONKEY_DATASET=... python3 rank_fidelity_probe.py mbpp --n 4000 --x 0.90
"""
import argparse, os, glob, json, time
import numpy as np

CORPORA = {"humaneval": "humaneval", "mbpp": "mbpp", "codealpaca": "codealpaca_20k"}
DMODEL = 4096


def load_lm_head(model_dir):
    import mlx.core as mx
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    norm_w = lm_w = emb_w = None
    for s in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        w = mx.load(s)
        for k, v in w.items():
            if k.endswith("model.norm.weight") or k == "norm.weight":
                norm_w = v
            if k.endswith("lm_head.weight"):
                lm_w = v
            if k.endswith("embed_tokens.weight"):
                emb_w = v
    if lm_w is None:
        lm_w = emb_w
    assert lm_w is not None and norm_w is not None
    norm_w = mx.array(norm_w).astype(mx.float32)
    lm_w = mx.array(lm_w).astype(mx.float32)
    mx.eval(norm_w, lm_w)
    return np.array(norm_w), np.array(lm_w), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--x", type=float, default=0.90)
    ap.add_argument("--ranks", type=str, default="8,16,32,48,64,96,128,192,256,384,512")
    args = ap.parse_args()

    base = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
    model_dir = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
    d = os.path.join(base, "traces", CORPORA[args.corpus])

    tok = np.fromfile(os.path.join(d, "tokens.bin"), np.int32)
    n_all = len(tok)
    H = np.memmap(os.path.join(d, "lastHiddenState.bin"), np.float32, "r", shape=(n_all, DMODEL))

    rng = np.random.default_rng(0)
    idx = rng.choice(n_all, min(args.n, n_all), replace=False)
    Hs = np.array(H[idx], np.float32)
    true_tok = tok[idx].astype(np.int64)

    print(f"[probe] {len(idx)} hiddens from {args.corpus}")
    print("[lm] loading norm + lm_head ...")
    norm_w, lm_w, eps = load_lm_head(model_dir)
    vocab = lm_w.shape[0]
    print(f"[lm] lm_head {lm_w.shape} vocab={vocab}")

    def rmsnorm(h):
        return h * (1.0 / np.sqrt((h * h).mean(-1, keepdims=True) + eps)) * norm_w

    Hn = rmsnorm(Hs)
    print("[probe] computing true logits (Hn @ lm_head.T; all-core BLAS) ...")
    true_logits = Hn @ lm_w.T
    tl = true_logits - true_logits.max(-1, keepdims=True)
    p = np.exp(tl); p /= p.sum(-1, keepdims=True)

    def capture(dist, x):
        order = np.argsort(-dist, axis=1)
        cs = np.cumsum(np.take_along_axis(dist, order, 1), axis=1)
        hit = 0; szs = np.empty(len(dist))
        for i in range(len(dist)):
            cut = int(np.searchsorted(cs[i], x)) + 1
            szs[i] = cut
            if true_tok[i] in order[i, :cut]:
                hit += 1
        return hit / len(dist), szs.mean()

    p_cap, p_sz = capture(p, args.x)
    argmax_hit = float((np.argmax(p, axis=1) == true_tok).mean())
    print(f"[truth] emitted-in-top{int(args.x*100)}%: {p_cap*100:.2f}% | set size {p_sz:.1f} | argmax-hit {argmax_hit*100:.2f}%")
    print()

    print("[probe] Gram-eig (vocab-independent; all-core BLAS) ...")
    t0 = time.time()
    mean = true_logits.mean(0, keepdims=True)
    Lc = true_logits - mean
    G = Lc @ Lc.T
    w, V = np.linalg.eigh(G)
    order = np.argsort(-w)
    w = np.clip(w[order], 0, None); V = V[:, order]
    S = np.sqrt(w); U = V
    print(f"[probe] Gram-eig done in {time.time()-t0:.1f}s; spectrum top: {S[:5].round(1)}")
    energy = (S**2).cumsum() / (S**2).sum()

    print()
    print(f"{'rank':>6}{'logit_energy':>13}{'JSD(qr,p)':>12}{'capture@'+str(int(args.x*100)):>12}{'setsize_qr':>12}{'setsize_p':>11}")
    for r in [int(x) for x in args.ranks.split(",")]:
        if r > len(S):
            continue
        approx = U[:, :r] @ (U[:, :r].T @ Lc) + mean
        al = approx - approx.max(-1, keepdims=True)
        q = np.exp(al); q /= q.sum(-1, keepdims=True)
        m = 0.5 * (p + q)
        jsd = (0.5 * (p * np.log(np.clip(p, 1e-9, 1) / np.clip(m, 1e-9, 1))).sum(1)
               + 0.5 * (q * np.log(np.clip(q, 1e-9, 1) / np.clip(m, 1e-9, 1))).sum(1)).mean()
        cap, sz = capture(q, args.x)
        print(f"{r:>6}{energy[r-1]*100:>12.1f}%{jsd:>12.4f}{cap*100:>11.2f}%{sz:>12.1f}{p_sz:>11.1f}")
    print()
    print("Read: smallest rank where JSD~0, capture ~ truth's capture, set size ~ truth's set size")
    print("= the rank for the lm_head-free head. rank x vocab = head's output-factor size.")
    print("Real hiddens; donkey's psi(zhat) may need slightly higher rank, but this tests the premise.")


if __name__ == "__main__":
    main()
