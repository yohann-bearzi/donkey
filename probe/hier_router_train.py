#!/usr/bin/env python3
"""
hier_router_train.py -- SHALLOW balanced cluster tree over lm_head rows + TRAINED per-node routers
(hierarchical softmax, distilled from lm_head). Navigate (beam) to the true token; lm_head-free.

v2: fixes depth-31 degeneracy. Shallow-by-construction (fixed depth, balanced k-means) so
navigation is only `depth` routing decisions (e.g. 3) -> errors don't compound to death.

Usage:
  MIMO_DIR=... DONKEY_DATASET=... python3 hier_router_train.py mbpp --n 8000 --depth 3 --branching 54 \
      --epochs 60 --beams 1,4,8 [--mlp 256]
"""
import argparse, os, glob, json, time
import numpy as np
from collections import defaultdict

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


def balanced_kmeans(X, k, iters=12, seed=0, slack=1.5):
    r = np.random.default_rng(seed); n = len(X)
    c = X[r.choice(n, min(k, n), replace=False)].copy()
    cap = max(1, int(np.ceil(n / k) * slack))
    assign = np.zeros(n, int)
    for _ in range(iters):
        d = -2 * X @ c.T + (c * c).sum(1)[None] + (X * X).sum(1)[:, None]
        order = np.argsort(d.min(1))
        assign = -np.ones(n, int); counts = np.zeros(k, int)
        for i in order:
            placed = False
            for j in np.argsort(d[i]):
                if counts[j] < cap:
                    assign[i] = j; counts[j] += 1; placed = True; break
            if not placed:
                assign[i] = int(np.argmin(d[i]))
        newc = np.array([X[assign == j].mean(0) if (assign == j).any() else c[j] for j in range(k)])
        if np.allclose(newc, c):
            break
        c = newc
    return assign


def build_shallow(rows, depth, branching, seed=0):
    nodes = []
    def new(idx):
        nid = len(nodes); nodes.append({"id": nid, "idx": idx, "children": None}); return nid
    root = new(np.arange(len(rows)))
    level = [root]
    for lv in range(depth):
        nxt = []
        for nid in level:
            idx = nodes[nid]["idx"]
            if len(idx) <= 1:
                continue
            k = min(branching, len(idx))
            if k < 2:
                continue
            a = balanced_kmeans(rows[idx], k, seed=nid % 100000 + seed)
            kids = []
            for j in range(k):
                sub = idx[a == j]
                if len(sub):
                    kids.append(new(sub))
            if len(kids) > 1:
                nodes[nid]["children"] = kids
                nxt.extend(kids)
        level = nxt
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--branching", type=int, default=54)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--beams", type=str, default="1,4,8")
    ap.add_argument("--mlp", type=int, default=0, help="if >0, hidden width of a per-node MLP router (nonlinear)")
    ap.add_argument("--val-frac", type=float, default=0.2)
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
    hmu = Hn.mean(0, keepdims=True); hsd = Hn.std(0, keepdims=True) + 1e-6
    Hstd = ((Hn - hmu) / hsd).astype(np.float32)

    logits = Hn @ lm_w.T
    z = logits - logits.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    q33, q66 = np.quantile(ent, [0.33, 0.66])

    print(f"[tree] SHALLOW balanced tree (depth={args.depth}, branching={args.branching}) ...")
    t0 = time.time()
    nodes = build_shallow(lm_w, args.depth, args.branching)
    internal = [n for n in nodes if n["children"]]
    leafsizes = np.array([len(n["idx"]) for n in nodes if not n["children"]])
    depthmap = {0: 0}
    for n in nodes:
        if n["children"]:
            for c in n["children"]:
                depthmap[c] = depthmap[n["id"]] + 1
    maxdepth = max(depthmap.values())
    print(f"[tree] {len(nodes)} nodes ({len(internal)} internal), depth {maxdepth}, "
          f"leaf max {leafsizes.max()} mean {leafsizes.mean():.1f}, built {time.time()-t0:.1f}s")

    tok2leaf = {}
    for n in nodes:
        if not n["children"]:
            for t in n["idx"]:
                tok2leaf[int(t)] = n["id"]
    parent = {}; slot = {}
    for n in nodes:
        if n["children"]:
            for s, c in enumerate(n["children"]):
                parent[c] = n["id"]; slot[c] = s
    def path_of(token):
        cur = tok2leaf[int(token)]; steps = []
        while cur in parent:
            par = parent[cur]; steps.append((par, slot[cur])); cur = par
        return steps[::-1]

    use_mlp = args.mlp > 0
    routers = {}
    for n in internal:
        C = len(n["children"])
        if use_mlp:
            routers[n["id"]] = [
                (rng.normal(size=(args.mlp, DMODEL)) * 0.02).astype(np.float32),
                (rng.normal(size=(C, args.mlp)) * 0.02).astype(np.float32),
            ]
        else:
            routers[n["id"]] = (rng.normal(size=(C, DMODEL)) * 0.02).astype(np.float32)

    def fwd(nid, Hb):
        W = routers[nid]
        if use_mlp:
            a = np.maximum(Hb @ W[0].T, 0.0)
            return a @ W[1].T, a
        return Hb @ W.T, None

    nval = int(len(idx) * args.val_frac)
    perm = rng.permutation(len(idx))
    val_i, tr_i = perm[:nval], perm[nval:]
    paths = [path_of(t) for t in true_tok]

    def softmax(x):
        x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

    beams = [int(b) for b in args.beams.split(",")]

    def navigate(h, beam):
        frontier = [(0, 0.0)]
        while any(nodes[nid]["children"] for nid, _ in frontier):
            cand = []
            for nid, sc in frontier:
                n = nodes[nid]
                if n["children"]:
                    s, _ = fwd(nid, h[None])
                    s = s[0]; s = s - s.max()
                    for slot_j, cid in enumerate(n["children"]):
                        cand.append((cid, sc + s[slot_j]))
                else:
                    cand.append((nid, sc))
            cand.sort(key=lambda z: -z[1])
            frontier = cand[:beam]
        toks = []
        for nid, _ in frontier:
            toks.extend(nodes[nid]["idx"].tolist())
        return toks

    def eval_nav(split_i, beam):
        buckets = {"sharp": [], "mid": [], "flat": []}
        for i in split_i:
            e = ent[i]
            b = "sharp" if e <= q33 else ("mid" if e <= q66 else "flat")
            buckets[b].append(true_tok[i] in navigate(Hstd[i], beam))
        return {k: (np.mean(v) if v else 0.0) for k, v in buckets.items()}

    print(f"[train] {'MLP' if use_mlp else 'linear'} routers ({len(routers)} nodes), {args.epochs} epochs ...")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(tr_i)
        node_batch = defaultdict(list)
        for i in order:
            for (nid, sl) in paths[i]:
                node_batch[nid].append((i, sl))
        for nid, items in node_batch.items():
            ex = np.array([it[0] for it in items]); sl = np.array([it[1] for it in items])
            Hb = Hstd[ex]
            sc, a = fwd(nid, Hb)
            pr = softmax(sc)
            g = pr.copy(); g[np.arange(len(ex)), sl] -= 1.0; g /= len(ex)
            if use_mlp:
                W1, W2 = routers[nid]
                gW2 = g.T @ a
                ga = (g @ W2) * (a > 0)
                gW1 = ga.T @ Hb
                routers[nid] = [W1 - args.lr * gW1, W2 - args.lr * gW2]
            else:
                routers[nid] = routers[nid] - args.lr * (g.T @ Hb)
        if ep % 15 == 0 or ep == args.epochs:
            vr = eval_nav(val_i, beams[0])
            print(f"[train ep{ep}] beam{beams[0]} val: sharp {vr['sharp']*100:.1f}% "
                  f"mid {vr['mid']*100:.1f}% flat {vr['flat']*100:.1f}% | {time.time()-t0:.1f}s/ep")

    print()
    print("FINAL trained-router navigation (val), by beam:")
    print(f"{'beam':>6}{'sharp':>10}{'mid':>10}{'flat':>10}{'MB/pred':>10}")
    for beam in beams:
        vr = eval_nav(val_i, beam)
        mb = beam * maxdepth * DMODEL * args.branching * 0.5 / 1e6
        print(f"{beam:>6}{vr['sharp']*100:>9.1f}%{vr['mid']*100:>9.1f}%{vr['flat']*100:>9.1f}%{mb:>10.2f}")
    print()
    print(f"depth {maxdepth} -> {maxdepth} routing decisions/prediction. Compare to ideal (100% sharp).")
    print("If sharp >~85% at small beam, lm_head-free navigation WORKS. Else try --mlp 256 (nonlinear).")


if __name__ == "__main__":
    main()
