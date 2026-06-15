#!/usr/bin/env python3
"""Load a v4 checkpoint and report hit@1, hit@k, cap90 + set-size on a val subset.
Standalone -- does NOT touch the running training. Shows where you stand on the 90%-energy metric.
Usage: MIMO_DIR=... DONKEY_DATASET=... python3 cap_probe.py <checkpoint.npz> mbpp --n 8000"""
import argparse, os, glob, json
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
            if k.endswith("model.norm.weight") or k == "norm.weight": norm_w = v
            if k.endswith("lm_head.weight"): lm_w = v
            if k.endswith("embed_tokens.weight"): emb_w = v
    if lm_w is None: lm_w = emb_w
    norm_w = mx.array(norm_w).astype(mx.float32); lm_w = mx.array(lm_w).astype(mx.float32)
    mx.eval(norm_w, lm_w)
    return np.array(norm_w), np.array(lm_w), eps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--x", type=float, default=0.9)
    args = ap.parse_args()
    base = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
    model_dir = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
    d = os.path.join(base, "traces", CORPORA[args.corpus])
    tok = np.fromfile(os.path.join(d, "tokens.bin"), np.int32); n_all = len(tok)
    H = np.memmap(os.path.join(d, "lastHiddenState.bin"), np.float32, "r", shape=(n_all, DMODEL))
    rng = np.random.default_rng(0)
    idx = rng.choice(n_all, min(args.n, n_all), replace=False)
    Hs = np.array(H[idx], np.float32); true_tok = tok[idx].astype(np.int64)
    norm_w, lm_w, eps = load_lm_head(model_dir)
    # load checkpoint weights (skip __opt__ / __epoch__ keys)
    ck = np.load(args.ckpt)
    nweights = sum(1 for k in ck.files if not k.startswith("__"))
    print(f"[ckpt] {args.ckpt}: {nweights} weight tensors"
          + (f" (+opt state)" if any(k.startswith('__opt__') for k in ck.files) else ""))
    print("[note] This probe applies lm_head to the REAL hidden as an UPPER-BOUND sanity check of")
    print("       the metric. To eval the DONKEY's psi(zhat) you'd rebuild the model from these")
    print("       weights -- which needs the v4 model class. This probe shows the TRUE-hidden")
    print("       capture (the ceiling) + lets you verify the metric. For donkey cap90, read the")
    print("       training log once it flushes (or relaunch with -u).")
    Hn = Hs * (1.0/np.sqrt((Hs*Hs).mean(-1,keepdims=True)+eps)) * norm_w
    logits = Hn @ lm_w.T
    z = logits - logits.max(-1,keepdims=True); q = np.exp(z); q /= q.sum(-1,keepdims=True)
    order = np.argsort(-q, axis=1); cs = np.cumsum(np.take_along_axis(q,order,1),axis=1)
    cap=0; szs=[]
    for i in range(len(q)):
        cut = int(np.searchsorted(cs[i], args.x))+1; szs.append(cut)
        cap += true_tok[i] in order[i,:cut]
    rank = (q > q[np.arange(len(q)), true_tok][:,None]).sum(1)
    print(f"[TRUE-hidden ceiling on {args.corpus}, n={len(idx)}]")
    print(f"  hit@1 {100*np.mean(rank<1):.1f}%  hit@4 {100*np.mean(rank<4):.1f}%  hit@8 {100*np.mean(rank<8):.1f}%")
    print(f"  cap@{int(args.x*100)}% {100*cap/len(q):.1f}%  mean set-size {np.mean(szs):.1f}")

if __name__ == "__main__": main()
