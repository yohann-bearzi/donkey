#!/usr/bin/env python3
"""
quarot_int4_probe.py -- does int4-quantizing lm_head (with QuaRot rotation) preserve the
next-token distribution well enough for ANE deployment? Measures fidelity vs the true fp lm_head,
WITH and WITHOUT the QuaRot Hadamard rotation, BUCKETED BY POSITION ENTROPY (sharp vs flat),
since sharp (near-Dirac) positions are the sensitive ones.

NOT a speed test -- QuaRot speeds up quantized INFERENCE on int4 hardware, not fp training.
This tests whether the ~88MB int4-lm_head is FAITHFUL enough to deploy (the ANE-fit gate).

Usage:
  MIMO_DIR=... DONKEY_DATASET=... python3 quarot_int4_probe.py mbpp --n 4000 --x 0.90 --group 128
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


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(H.shape[0])).astype(np.float32)


def quant_int4_per_group(W, group=128):
    V, D = W.shape
    pad = (-D) % group
    if pad:
        W = np.concatenate([W, np.zeros((V, pad), W.dtype)], axis=1)
    Dg = W.shape[1]
    Wg = W.reshape(V, Dg // group, group)
    scale = np.abs(Wg).max(-1, keepdims=True) / 7.0 + 1e-12
    q = np.clip(np.round(Wg / scale), -7, 7)
    deq = (q * scale).reshape(V, Dg)
    return deq[:, :D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--x", type=float, default=0.90)
    ap.add_argument("--group", type=int, default=128)
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

    def dist(logits):
        z = logits - logits.max(-1, keepdims=True)
        e = np.exp(z); return e / e.sum(-1, keepdims=True)

    print("[probe] true logits ...")
    p = dist(Hn @ lm_w.T)

    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    q33, q66 = np.quantile(ent, [0.33, 0.66])
    buckets = {"sharp(low-ent)": ent <= q33, "mid": (ent > q33) & (ent <= q66), "flat(high-ent)": ent > q66}

    print(f"[probe] int4 quant (group={args.group}) WITHOUT rotation ...")
    deq_noQ = quant_int4_per_group(lm_w, args.group)
    q_noQ = dist(Hn @ deq_noQ.T)

    print("[probe] QuaRot: Hadamard rotate, int4 quant, recover ...")
    Qh = hadamard(DMODEL)
    lm_rot = lm_w @ Qh
    Hn_rot = Hn @ Qh
    deq_Q = quant_int4_per_group(lm_rot, args.group)
    q_Q = dist(Hn_rot @ deq_Q.T)

    def jsd(p, q):
        m = 0.5 * (p + q)
        return (0.5 * (p * np.log(np.clip(p, 1e-9, 1) / np.clip(m, 1e-9, 1))).sum(1)
                + 0.5 * (q * np.log(np.clip(q, 1e-9, 1) / np.clip(m, 1e-9, 1))).sum(1))

    def capture(dst, mask, x):
        sub = dst[mask]; tt = true_tok[mask]
        order = np.argsort(-sub, axis=1)
        cs = np.cumsum(np.take_along_axis(sub, order, 1), axis=1)
        hit = 0; szs = np.empty(len(sub))
        for i in range(len(sub)):
            cut = int(np.searchsorted(cs[i], x)) + 1
            szs[i] = cut
            if tt[i] in order[i, :cut]:
                hit += 1
        return hit / max(len(sub), 1), szs.mean()

    jp_noQ = jsd(p, q_noQ); jp_Q = jsd(p, q_Q)
    print()
    print(f"{'bucket':>16}{'n':>6}{'JSD_int4':>11}{'JSD_QuaRot':>12}{'cap_true':>10}{'cap_i4':>9}{'cap_QR':>9}{'sz_true':>9}{'sz_QR':>8}")
    for name, mask in buckets.items():
        nb = int(mask.sum())
        ct, st = capture(p, mask, args.x)
        ci, _ = capture(q_noQ, mask, args.x)
        cq, sq = capture(q_Q, mask, args.x)
        print(f"{name:>16}{nb:>6}{jp_noQ[mask].mean():>11.4f}{jp_Q[mask].mean():>12.4f}"
              f"{ct*100:>9.1f}%{ci*100:>8.1f}%{cq*100:>8.1f}%{st:>9.1f}{sq:>8.1f}")
    print()
    print(f"OVERALL: JSD int4={jp_noQ.mean():.4f}  JSD QuaRot={jp_Q.mean():.4f}")
    print("Read: QuaRot JSD should be <= int4 JSD. Check the SHARP bucket -- if QuaRot keeps")
    print("capture~true and set-size~true there, int4-lm_head is faithful enough to deploy (~88MB).")
    print("If sharp positions degrade, need int8.")


if __name__ == "__main__":
    main()
