#!/usr/bin/env python3
"""
hier_nav_probe.py -- can a recursive cluster tree over the vocabulary NAVIGATE to the true
next token, using only small per-node routers, so we never project to the full 152k logits?
Tests the "recursive refinement operator" idea: navigate (search), don't project.

Builds a recursive k-means tree over lm_head rows, then for harvested hiddens measures whether
BEAM navigation reaches the true emitted token, bucketed by entropy (sharp vs flat). Compares
routing signals:
  - centroid (cluster mean) router         : the naive, cheap router (lower bound)
  - cluster-logit-sum (ideal navigation)   : the BEST a router could do (upper bound) --
       at each node, score = logsumexp of the true lm_head logits of tokens under that node.
       If the ideal navigation can't reach the truth, no router can -> idea fails.
       If ideal works but centroid doesn't -> a TRAINED router is needed (still viable).

Usage:
  MIMO_DIR=... DONKEY_DATASET=... python3 hier_nav_probe.py mbpp --n 2000 --branching 32 --leaf 64 --beams 1,4,8,16
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


def kmeans(X, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), min(k, len(X)), replace=False)].copy()
    for _ in range(iters):
        d = -2 * X @ c.T + (c * c).sum(1)[None] + (X * X).sum(1)[:, None]
        a = d.argmin(1)
        newc = np.array([X[a == j].mean(0) if (a == j).any() else c[j] for j in range(len(c))])
        if np.allclose(newc, c):
            break
        c = newc
    return a, c


def build_tree(idx, rows, branching, leaf, depth=0, maxdepth=6):
    node = {"idx": idx, "centroid": rows[idx].mean(0)}
    if len(idx) <= leaf or depth >= maxdepth:
        node["leaf"] = True
        return node
    k = min(branching, len(idx))
    a, c = kmeans(rows[idx], k, seed=depth)
    node["leaf"] = False
    node["children"] = []
    for j in range(len(c)):
        sub = idx[a == j]
        if len(sub):
            node["children"].append(build_tree(sub, rows, branching, leaf, depth + 1, maxdepth))
    return node


def navigate(root, score_fn, beam):
    frontier = [root]
    while any(not n["leaf"] for n in frontier):
        cand = []
        for n in frontier:
            cand.extend(n["children"] if not n["leaf"] else [n])
        scores = np.array([score_fn(c) for c in cand])
        keep = np.argsort(-scores)[:beam]
        frontier = [cand[i] for i in keep]
    toks = []
    for n in frontier:
        toks.extend(n["idx"].tolist())
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--branching", type=int, default=32)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--beams", type=str, default="1,4,8,16")
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

    Hn = Hs * (1.0 / np.sqrt((Hs * Hs).mean(-1, keepdims=True) + eps)) * norm_w

    logits = Hn @ lm_w.T
    z = logits - logits.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    q33, q66 = np.quantile(ent, [0.33, 0.66])
    buckets = {"sharp": ent <= q33, "mid": (ent > q33) & (ent <= q66), "flat": ent > q66}

    print(f"[probe] building recursive tree (branching={args.branching}, leaf={args.leaf}) over {vocab} rows ...")
    t0 = time.time()
    root = build_tree(np.arange(vocab), lm_w, args.branching, args.leaf)
    def stats(n):
        if n["leaf"]:
            return 1, 1, len(n["idx"])
        ch = [stats(c) for c in n["children"]]
        return 1 + sum(c[0] for c in ch), 1 + max(c[1] for c in ch), max(c[2] for c in ch)
    nodes, depth, maxleaf = stats(root)
    print(f"[probe] tree: {nodes} nodes, depth {depth}, max leaf {maxleaf}, built in {time.time()-t0:.1f}s")
    print()

    beams = [int(b) for b in args.beams.split(",")]

    def reach_rate(score_builder, mask, beam):
        sub = np.where(mask)[0]
        hit = 0
        for i in sub:
            h = Hn[i]
            sf = score_builder(h, i)
            toks = navigate(root, sf, beam)
            if true_tok[i] in toks:
                hit += 1
        return hit / max(len(sub), 1), len(sub)

    def centroid_score(h, i):
        return lambda node: float(node["centroid"] @ h)

    def ideal_score(h, i):
        li = logits[i]
        def sf(node):
            lg = li[node["idx"]]
            m = lg.max()
            return float(m + np.log(np.exp(lg - m).sum()))
        return sf

    for routername, builder in [("centroid(naive)", centroid_score), ("ideal(logit-sum)", ideal_score)]:
        print(f"--- Router: {routername} ---")
        print(f"{'beam':>6}" + "".join(f"{b:>12}" for b in buckets))
        for beam in beams:
            row = f"{beam:>6}"
            for name, mask in buckets.items():
                r, nb = reach_rate(builder, mask, beam)
                row += f"{r*100:>11.1f}%"
            print(row)
        print()
    print("Read: 'ideal' = can ANY router navigate to the truth (upper bound). If ideal~100% at")
    print("low beam, the tree STRUCTURE is good. 'centroid' = naive cheap router (lower bound).")
    print("If ideal works but centroid doesn't, a TRAINED router (distilled) is needed -- still viable.")
    print("Beam width = candidate paths = bandwidth cost. Smallest beam reaching truth = the cost.")


if __name__ == "__main__":
    main()
