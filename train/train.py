#!/usr/bin/env python3
"""
donkey_train_v4.py  --  WAVE 0 (single-step) with the gated-JSD distributional loss.

Multi-corpus, balanced:
  - --val-n N        : FROZEN val pool of N transitions PER corpus (carved once, never resampled).
                       Keeps val balanced (a big corpus can't over-represent it).
  - --per-corpus-n N : EACH EPOCH draw a fresh N transitions PER corpus from its train pool
                       (the remainder not in val). Balanced 50/50 per epoch; over many epochs
                       the donkey sees more of a large corpus than N (fresh draw each epoch).
  - --holdout C      : corpus used ONLY for eval (never trained), pure cross-corpus.

LOSS (per step s, here only s=0):
  L_s = lam_cos*(1-cos(zhat,z_tgt)) + gate(stopgrad cos)*lam_dist*JSD(q,p) + lam_rec*JSD(dec(z_tgt),p) + lam_sig*SIGReg
  q=softmax(lm_head(psi(zhat))), p=harvested nucleus dist (renorm). gate=sigmoid(sharp*(cos-tau)).
  NO argmax in loss; sequence-hit % logged per depth as diagnostic. Action = emitted token.
Note: batched-lane harvest ~0.4%/step intrinsic drift -> cosine ceiling ~0.996.

Usage:
  python3 donkey_train_v4.py mbpp codealpaca --holdout humaneval \
      --per-corpus-n 1000000 --val-n 20000 --d 64 --dp 256 --epochs 300
"""
import argparse, os, sys, glob, json, time, math
import numpy as np

CORPORA = {"humaneval": "humaneval", "mbpp": "mbpp", "codealpaca": "codealpaca_20k"}
DMODEL = 4096
ENC_LAYERS = 2
DEC_LAYERS = 2
PRED_LAYERS = 6
WINDOW = 13
STOP = (151645, 151643)


def load_corpus(base, corpus):
    d = os.path.join(base, "traces", CORPORA[corpus])
    tok = np.fromfile(os.path.join(d, "tokens.bin"), np.int32)
    n = len(tok)
    H = np.memmap(os.path.join(d, "lastHiddenState.bin"), np.float32, "r", shape=(n, DMODEL))
    pidx = np.fromfile(os.path.join(d, "prompt_idx.bin"), np.int32)
    counts = np.fromfile(os.path.join(d, "topp_counts.bin"), np.int32)
    ids = np.fromfile(os.path.join(d, "topp_ids.bin"), np.int32)
    probs = np.fromfile(os.path.join(d, "topp_probs.bin"), np.float16).astype(np.float32)
    offs = np.zeros(n + 1, np.int64)
    offs[1:] = np.cumsum(counts.astype(np.int64))
    return dict(tok=tok, H=H, pidx=pidx, counts=counts, ids=ids, probs=probs, offs=offs, n=n)


def make_transitions(ds, window=WINDOW):
    tok, pidx, n = ds["tok"], ds["pidx"], ds["n"]
    wi, aix, tgt = [], [], []
    start = 0
    for i in range(1, n):
        if pidx[i] != pidx[i - 1]:
            start = i
        if i + 1 < n and pidx[i + 1] == pidx[i] and (i - start + 1) >= window:
            wi.append(i); aix.append(tok[i]); tgt.append(i + 1)
    return np.array(wi, np.int64), np.array(aix, np.int64), np.array(tgt, np.int64)


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
    assert lm_w is not None and norm_w is not None, "lm_head/norm not found in shards"
    assert emb_w is not None, "embed_tokens.weight not found in shards (needed for action embedding)"
    return (mx.array(norm_w.astype(mx.float32)),
            mx.array(lm_w.astype(mx.float32)),
            mx.array(emb_w.astype(mx.float32)),
            eps)


def build_model(mx, nn, d, dp, enc_layers, dec_layers, pred_layers, n_actions, emb_w):
    class Block(nn.Module):
        def __init__(s, w, adaln=False):
            super().__init__()
            s.n1 = nn.RMSNorm(w); s.n2 = nn.RMSNorm(w)
            s.q = nn.Linear(w, w, bias=False); s.k = nn.Linear(w, w, bias=False)
            s.v = nn.Linear(w, w, bias=False); s.o = nn.Linear(w, w, bias=False)
            s.f1 = nn.Linear(w, 4 * w); s.f2 = nn.Linear(4 * w, w)
            s.adaln = adaln
            if adaln:
                s.ada = nn.Linear(w, 4 * w)

        def __call__(s, x, cond=None):
            h = s.n1(x)
            x = x + s.o(s.v(h))
            h2 = s.n2(x)
            if s.adaln and cond is not None:
                g = s.ada(cond)
                shift, scale, gate1, gate2 = mx.split(g, 4, axis=-1)
                if h2.ndim == 3 and shift.ndim == 2:
                    shift = shift[:, None, :]; scale = scale[:, None, :]; gate1 = gate1[:, None, :]
                h2 = h2 * (1 + scale) + shift
                x = x + gate1 * s.f2(nn.gelu(s.f1(h2)))
            else:
                x = x + s.f2(nn.gelu(s.f1(h2)))
            return x

    class WM(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc_in = nn.Linear(DMODEL, dp)
            s.enc_blocks = [Block(dp) for _ in range(enc_layers)]
            s.enc_out = nn.Linear(dp, d)
            s.pred_in = nn.Linear(d, dp)
            s.pred_pos = mx.zeros((WINDOW, dp))
            s.pred_blocks = [Block(dp, adaln=True) for _ in range(pred_layers)]
            s.pred_out = nn.Linear(dp, d)
            s.act_proj = nn.Linear(DMODEL, dp, bias=False)  # projects frozen trunk input-embed -> dp
            s.depth_emb = nn.Embedding(8, dp)
            s.dec_in = nn.Linear(d, dp)
            s.dec_blocks = [Block(dp) for _ in range(dec_layers)]
            s.dec_out = nn.Linear(dp, DMODEL)
            s.rootd = math.sqrt(d)

        def normd(s, z):
            return z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-6) * s.rootd

        def phi(s, h):
            x = s.enc_in(h)
            for b in s.enc_blocks:
                x = b(x)
            return s.normd(s.enc_out(x))

        def predict(s, zwin, action, depth):
            x = s.pred_in(zwin) + s.pred_pos[None]
            cond = s.act_proj(emb_w[action]) + s.depth_emb(depth)
            for b in s.pred_blocks:
                x = b(x, cond)
            x = x.mean(axis=1)
            return s.normd(s.pred_out(x))

        def psi(s, z):
            x = s.dec_in(z)
            for b in s.dec_blocks:
                x = b(x)
            return s.dec_out(x)

    return WM()


def jsd_dense(q, p, eps=1e-9):
    import mlx.core as mx
    m = 0.5 * (p + q)
    qc = mx.clip(q, eps, 1.0); pc = mx.clip(p, eps, 1.0); mc = mx.clip(m, eps, 1.0)
    kl_pm = mx.sum(pc * (mx.log(pc) - mx.log(mc)), axis=-1)
    kl_qm = mx.sum(qc * (mx.log(qc) - mx.log(mc)), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def build_p_dense(mx, ids_batch, probs_batch, vocab):
    B, Kmax = ids_batch.shape
    safe = mx.maximum(ids_batch, 0).astype(mx.int32)
    rows = mx.broadcast_to(mx.arange(B)[:, None], (B, Kmax))
    p = mx.zeros((B, vocab))
    p = p.at[rows.reshape(-1), safe.reshape(-1)].add(probs_batch.reshape(-1))
    s = mx.sum(p, axis=-1, keepdims=True) + 1e-9
    return p / s


def train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    base = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
    model_dir = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
    rng = np.random.default_rng(args.seed)

    dslist = []
    train_pools = []
    va_cid, va_wi, va_aix, va_tgt = [], [], [], []
    for ci, corpus in enumerate(args.corpora):
        print(f"[data] loading train corpus {corpus} ...")
        ds = load_corpus(base, corpus)
        dslist.append(ds)
        wi, aix, tgt = make_transitions(ds)
        perm = rng.permutation(len(wi))
        vn = min(args.val_n, len(wi) // 2) if args.val_n else max(1, int(len(wi) * 0.10))
        vsel, tsel = perm[:vn], perm[vn:]
        train_pools.append((wi[tsel], aix[tsel], tgt[tsel]))
        va_cid.append(np.full(len(vsel), ci, np.int32)); va_wi.append(wi[vsel]); va_aix.append(aix[vsel]); va_tgt.append(tgt[vsel])
        print(f"[data]   {corpus}: {len(wi)} transitions -> train pool {len(tsel)} / FROZEN val {len(vsel)}")
    va_cid = np.concatenate(va_cid); va_wi = np.concatenate(va_wi); va_aix = np.concatenate(va_aix); va_tgt = np.concatenate(va_tgt)
    per_n = args.per_corpus_n if args.per_corpus_n else min(len(tp[0]) for tp in train_pools)
    print(f"[data] FROZEN val {len(va_wi)} ({args.val_n}/corpus) | per-epoch train draw {per_n}/corpus x{len(dslist)}")

    hold = None
    if args.holdout:
        print(f"[data] loading holdout corpus {args.holdout} (NEVER trained) ...")
        dsh = load_corpus(base, args.holdout)
        wih, aixh, tgth = make_transitions(dsh)
        cidh = np.zeros(len(wih), np.int32)
        hold = ([dsh], cidh, wih, aixh, tgth)
        print(f"[holdout] {args.holdout}: {len(wih)} transitions (pure cross-corpus)")

    samp = []
    for ci, (twi, _, _) in enumerate(train_pools):
        if len(twi):
            samp.append(np.array(dslist[ci]["H"][twi[:min(10000, len(twi))]], np.float32))
    samp = np.concatenate(samp, 0)
    mu = samp.mean(0, keepdims=True); sd = samp.std(0, keepdims=True) + 1e-6
    print(f"[norm] per-dim standardize (mu,sd from {samp.shape[0]} pooled hiddens)")

    print("[lm] loading norm + lm_head + embed_tokens from shards ...")
    norm_w, lm_w, emb_w, eps = load_lm_head(model_dir)
    vocab = lm_w.shape[0]
    print(f"[lm] embed_tokens {emb_w.shape} (frozen action embedding)")
    print(f"[lm] lm_head {lm_w.shape}  vocab={vocab}")

    maxtok = max(int(ds["tok"].max()) for ds in dslist)
    if hold is not None:
        maxtok = max(maxtok, int(hold[0][0]["tok"].max()))
    n_actions = maxtok + 1

    model = build_model(mx, nn, args.d, args.dp, args.enc, args.dec, PRED_LAYERS, n_actions, emb_w)
    opt = optim.Adam(learning_rate=args.lr)
    start_ep = 1
    if args.resume:
        from mlx.utils import tree_unflatten
        ck = np.load(args.resume)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        start_ep = int(ck["__epoch__"]) + 1 if "__epoch__" in ck.files else 1
        okeys = [k for k in ck.files if k.startswith("__opt__")]
        if okeys:
            try:
                opt.init(model.trainable_parameters())
                opt.state = tree_unflatten([(k[len("__opt__"):], mx.array(ck[k])) for k in okeys])
                print(f"[resume] restored weights + opt state from {args.resume}, start ep {start_ep}")
            except Exception as e:
                print(f"[resume] weights+epoch restored; opt-state restore failed ({e}); fresh optimizer")
        else:
            print(f"[resume] restored weights (no opt state) from {args.resume}, start ep {start_ep}")
    mu_mx = mx.array(mu); sd_mx = mx.array(sd)

    def standardize(h): return (h - mu_mx) / sd_mx
    def destandardize(h): return h * sd_mx + mu_mx

    def lm_logits(h_pred_std):
        h = destandardize(h_pred_std)
        hn = h * (1.0 / mx.sqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps)) * norm_w
        return hn @ lm_w.T

    def gather_window(dl, cid, end_idx):
        B = len(end_idx)
        out = np.empty((B, WINDOW, DMODEL), np.float32)
        for j in range(B):
            e = int(end_idx[j]); out[j] = dl[int(cid[j])]["H"][e - WINDOW + 1: e + 1]
        return mx.array(out)

    def gather_nucleus(dl, cid, tgt_idx, kmax_cap=64):
        B = len(tgt_idx); rows_ids, rows_probs = [], []; kmax = 1
        for j in range(B):
            t = int(tgt_idx[j]); dsc = dl[int(cid[j])]
            o0, o1 = dsc["offs"][t], dsc["offs"][t + 1]
            ii = dsc["ids"][o0:o1][:kmax_cap]; pp = dsc["probs"][o0:o1][:kmax_cap]
            rows_ids.append(ii); rows_probs.append(pp); kmax = max(kmax, len(ii))
        ib = np.full((B, kmax), -1, np.int32); pb = np.zeros((B, kmax), np.float32)
        for j in range(B):
            k = len(rows_ids[j]); ib[j, :k] = rows_ids[j]; pb[j, :k] = rows_probs[j]
        return mx.array(ib), mx.array(pb)

    def tgt_tokens(dl, cid, tgt_idx):
        return np.array([dl[int(cid[j])]["tok"][int(tgt_idx[j])] for j in range(len(tgt_idx))], np.int32)

    def loss_fn(model, zwin, action, depth, h_tgt_std, p_ids, p_probs):
        z_tgt = model.phi(h_tgt_std)
        zhat = model.predict(zwin, action, depth)
        zn = zhat / (mx.linalg.norm(zhat, axis=-1, keepdims=True) + 1e-6)
        tn = z_tgt / (mx.linalg.norm(z_tgt, axis=-1, keepdims=True) + 1e-6)
        cos = mx.sum(zn * tn, axis=-1); l_cos = 1 - cos
        q = mx.softmax(lm_logits(model.psi(zhat)), axis=-1)
        p = build_p_dense(mx, p_ids, p_probs, vocab)
        jsd_pred = jsd_dense(q, p)
        gate = mx.sigmoid(args.gate_sharp * (mx.stop_gradient(cos) - args.gate_tau))
        l_dist = gate * jsd_pred
        q_rec = mx.softmax(lm_logits(model.psi(z_tgt)), axis=-1)
        l_rec = jsd_dense(q_rec, p)
        l_kd = jsd_dense(q, mx.stop_gradient(q_rec))
        l_sig = mx.mean((mx.var(zhat, axis=0) - 1.0) ** 2)
        l_zl2 = mx.mean((zhat - mx.stop_gradient(z_tgt)) ** 2)
        l_l2 = mx.mean((model.psi(zhat) - h_tgt_std) ** 2)
        _ph = model.psi(zhat)
        _pn = _ph / (mx.linalg.norm(_ph, axis=-1, keepdims=True) + 1e-6)
        _hn = h_tgt_std / (mx.linalg.norm(h_tgt_std, axis=-1, keepdims=True) + 1e-6)
        dcos = mx.mean(mx.sum(_pn * _hn, axis=-1))
        l_pm = mx.mean((_ph - mx.stop_gradient(model.psi(z_tgt))) ** 2)
        S_q = mx.sum(q * q, axis=-1)
        S_p = mx.sum(p * p, axis=-1)
        l_shape = mx.mean(gate * mx.maximum(S_p - S_q, 0.0))
        t_cos = args.lam_cos * mx.mean(l_cos)
        t_dist = args.lam_dist * mx.mean(l_dist)
        t_rec = args.lam_rec * mx.mean(l_rec)
        t_sig = args.lam_sig * l_sig
        t_l2 = args.lam_l2 * l_l2
        t_shape = args.lam_shape * _shape_scale["v"] * l_shape
        t_zl2 = args.lam_zl2 * l_zl2
        t_pm = args.lam_pm * l_pm
        t_kd = args.lam_kd * mx.mean(l_kd)
        L = t_cos + t_dist + t_rec + t_sig + t_l2 + t_shape + t_zl2 + t_pm + t_kd
        return L, (mx.mean(cos), mx.mean(jsd_pred), mx.mean(gate),
                   t_cos, t_dist, t_rec, t_sig, t_l2, t_shape, dcos, t_zl2, t_pm, t_kd)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    _EVAL_IDX = {}
    _eval_rng = np.random.default_rng(12345)
    def _frozen_idx(key, n):
        if key not in _EVAL_IDX:
            _EVAL_IDX[key] = (_eval_rng.choice(n, min(args.eval_n, n), replace=False)
                              if n > args.eval_n else np.arange(n))
        return _EVAL_IDX[key]

    def seq_hit(dl, cid, sub_wi, sub_aix, sub_tgt, nmax=4000, frozen_key=None):
        if frozen_key is not None:
            idx = _frozen_idx(frozen_key, len(sub_wi))
        else:
            idx = np.arange(len(sub_wi))
            if len(idx) > nmax:
                idx = rng.choice(len(idx), nmax, replace=False)
        hit = 0; tot = 0
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            zw = model.phi(standardize(gather_window(dl, cid[sel], sub_wi[sel])))
            action = mx.array(sub_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            zhat = model.predict(zw, action, depth)
            am = mx.argmax(lm_logits(model.psi(zhat)), axis=-1)
            tt = tgt_tokens(dl, cid[sel], sub_tgt[sel])
            hit += int(mx.sum(am == mx.array(tt))); tot += len(sel)
        return hit / max(tot, 1)

    def capture_at(dl, cid, sub_wi, sub_aix, sub_tgt, frozen_key=None, Xs=(0.9, 0.99)):
        if frozen_key is not None:
            idx = _frozen_idx(frozen_key, len(sub_wi))
        else:
            idx = np.arange(len(sub_wi))
            if len(idx) > 4000:
                idx = rng.choice(len(idx), 4000, replace=False)
        tot = 0
        cap = {X: 0 for X in Xs}; dsz = {X: 0.0 for X in Xs}; tsz = {X: 0.0 for X in Xs}
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            zw = model.phi(standardize(gather_window(dl, cid[sel], sub_wi[sel])))
            action = mx.array(sub_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            zhat = model.predict(zw, action, depth)
            q = mx.softmax(lm_logits(model.psi(zhat)), axis=-1)
            qd = -mx.sort(-q, axis=-1)
            cm = mx.cumsum(qd, axis=-1)
            qd_np = np.array(qd); cm_np = np.array(cm)
            tt = tgt_tokens(dl, cid[sel], sub_tgt[sel])
            q_np = np.array(q)
            q_true = q_np[np.arange(len(sel)), tt.astype(np.int64)]
            p_ids, p_probs = gather_nucleus(dl, cid[sel], sub_tgt[sel])
            pp = np.array(p_probs)
            pp = pp / np.clip(pp.sum(1, keepdims=True), 1e-9, None)
            pps = -np.sort(-pp, axis=1); pcm = np.cumsum(pps, axis=1)
            for X in Xs:
                cutoff_np = (cm_np >= X).argmax(axis=1)
                thresh = qd_np[np.arange(len(sel)), cutoff_np]
                cap[X] += int(np.sum(q_true >= thresh - 1e-9))
                dsz[X] += float(np.sum(cutoff_np + 1))
                tsz[X] += float(np.sum((pcm >= X).argmax(axis=1) + 1))
            tot += len(sel)
        out = {}
        for X in Xs:
            out[X] = (cap[X] / max(tot, 1), dsz[X] / max(tot, 1), tsz[X] / max(tot, 1))
        return out


    best = None; besth = None; bad = 0
    if args.probe_failure:
        from mlx.utils import tree_unflatten
        ck = np.load(args.probe_failure)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[probe] loaded {args.probe_failure} ({len(wkeys)} tensors)")

        # precompute action-token frequency per corpus (over the full tok array)
        from collections import Counter
        freq_by_cid = {}
        for ci, ds in enumerate(dslist):
            c = Counter(ds["tok"].tolist())
            freq_by_cid[ci] = c
        if hold is not None:
            hc = Counter(hold[0][0]["tok"].tolist())

        def run_probe(dl, cid, sub_wi, sub_aix, sub_tgt, freq_lookup, tag):
            idx = _frozen_idx(tag, len(sub_wi))
            rows = []
            for b in range(0, len(idx), args.bs):
                sel = idx[b:b + args.bs]
                hwin = gather_window(dl, cid[sel], sub_wi[sel])
                zw = model.phi(standardize(hwin))
                action = mx.array(sub_aix[sel].astype(np.int32))
                depth = mx.zeros(len(sel), dtype=mx.int32)
                zhat = model.predict(zw, action, depth)
                # cos(zhat, z_true): z_true = phi(standardize(true next hidden))
                tnext = np.empty((len(sel), DMODEL), np.float32)
                for jj, s in enumerate(sel):
                    t = int(sub_tgt[s]); tnext[jj] = dl[int(cid[s])]["H"][t]
                z_true = model.phi(standardize(mx.array(tnext)))
                cz = mx.sum(zhat * z_true, axis=-1) / (
                    mx.linalg.norm(zhat, axis=-1) * mx.linalg.norm(z_true, axis=-1) + 1e-6)
                q = mx.softmax(lm_logits(model.psi(zhat)), axis=-1)
                qd = -mx.sort(-q, axis=-1); cm = mx.cumsum(qd, axis=-1)
                q_np = np.array(q); cm_np = np.array(cm); qd_np = np.array(qd)
                cz_np = np.array(cz)
                tt = tgt_tokens(dl, cid[sel], sub_tgt[sel])
                tt_np = np.array(tt).astype(np.int64)
                p_ids, p_probs = gather_nucleus(dl, cid[sel], sub_tgt[sel])
                pp = np.array(p_probs); pp = pp / np.clip(pp.sum(1, keepdims=True), 1e-9, None)
                for jj in range(len(sel)):
                    qi = q_np[jj]
                    rank = int((qi > qi[tt_np[jj]]).sum())
                    cut90 = int((cm_np[jj] >= 0.90).argmax()) + 1
                    cut99 = int((cm_np[jj] >= 0.99).argmax()) + 1
                    ent = float(-(pp[jj][pp[jj] > 0] * np.log(pp[jj][pp[jj] > 0])).sum())
                    af = int(freq_lookup.get(int(tt_np[jj]), 0))
                    rows.append((float(cz_np[jj]), ent, af, rank,
                                 float(qi.max()), int(rank < cut90), int(rank < cut99)))
            return np.array(rows, dtype=np.float64)

        R = run_probe(dslist, va_cid, va_wi, va_aix, va_tgt,
                      freq_by_cid.get(0, {}), "val")
        cols = ["cos", "true_ent", "act_freq", "true_rank", "max_prob", "cap90", "cap99"]
        out = {c: R[:, i] for i, c in enumerate(cols)}
        if hold is not None:
            RH = run_probe(hold[0], hold[1], hold[2], hold[3], hold[4], hc, "hold")
            for i, c in enumerate(cols):
                out["h_" + c] = RH[:, i]
        np.savez("/tmp/donkey_probe.npz", **out)
        print(f"[probe] val n={len(R)} -> saved /tmp/donkey_probe.npz")

        def strat(R, name):
            cos, ent, af, rank, mp, c90, c99 = [R[:, i] for i in range(7)]
            print(f"\n=== {name} (n={len(R)}, overall cap90={100*c90.mean():.1f}%) ===")
            print(" P1 cap90 by TRUE-ENTROPY (sharp->flat):")
            for lo, hi, lab in [(0,.05,"~0 sharp"),(.05,.3,".05-.3"),(.3,.7,".3-.7"),(.7,1.5,".7-1.5"),(1.5,99,"1.5+ flat")]:
                m = (ent>=lo)&(ent<hi)
                if m.sum(): print(f"    {lab:10} n={int(m.sum()):4d}  cap90={100*c90[m].mean():4.0f}%")
            print(" P2 cap90 by COS (low=predictor-fail, high=decoder-fail):")
            for lo, hi in [(0,.9),(.9,.97),(.97,.99),(.99,1.01)]:
                m=(cos>=lo)&(cos<hi)
                if m.sum(): print(f"    cos[{lo:.2f},{hi:.2f}) n={int(m.sum()):4d}  cap90={100*c90[m].mean():4.0f}%")
            miss=c90<.5; near=miss&(c99>.5)
            print(f" P6 of {int(miss.sum())} misses: near(cap99 catches)={int(near.sum())} ({100*near.sum()/max(miss.sum(),1):.0f}%), median miss-rank={int(np.median(rank[miss])) if miss.sum() else 0}")
            print(" P7 calibration (confidence -> cap90):")
            for lo, hi in [(0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]:
                m=(mp>=lo)&(mp<hi)
                if m.sum(): print(f"    conf[{lo:.1f},{hi:.1f}) n={int(m.sum()):4d}  cap90={100*c90[m].mean():4.0f}%")
            print(" P3 cap90 by ACTION-FREQUENCY (rare->common):")
            for lo, hi, lab in [(0,10,"rare<10"),(10,100,"10-100"),(100,1000,"100-1k"),(1000,1e12,"1k+")]:
                m=(af>=lo)&(af<hi)
                if m.sum(): print(f"    {lab:8} n={int(m.sum()):4d}  cap90={100*c90[m].mean():4.0f}%")

        strat(R, "VAL (in-distribution)")
        if hold is not None:
            strat(RH, "HOLDOUT (humaneval)")
        print("\n[probe] done. per-position table at /tmp/donkey_probe.npz (cos,true_ent,act_freq,true_rank,max_prob,cap90,cap99)")
        raise SystemExit(0)

    if args.probe_retrieval:
        from mlx.utils import tree_unflatten
        ck = np.load(args.probe_retrieval)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[retrieval-probe] loaded {args.probe_retrieval}")

        from collections import Counter
        freq0 = Counter(dslist[0]["tok"].tolist())

        # precompute embed_tokens and lm_head row norms for cosine (both [V,4096])
        emb_np = np.array(emb_w)              # [V,4096] frozen trunk input embedding
        lm_np = np.array(lm_w)                # [V,4096] lm_head
        emb_norm = np.linalg.norm(emb_np, axis=1) + 1e-9
        lm_norm = np.linalg.norm(lm_np, axis=1) + 1e-9

        idx = _frozen_idx("val", len(va_wi))
        lm_ranks = []; emb_ranks = []; afs = []
        n_rare_miss = 0
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            zhat = model.predict(zw, action, depth)
            psi = model.psi(zhat)                       # [b,4096]
            # lm_logits applies destandardize+rmsnorm before lm_head; replicate for rank consistency
            h = destandardize(psi)
            hn = h * (1.0 / mx.sqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps)) * norm_w
            psi_np = np.array(hn)                        # the vector that lm_head scores
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            for jj in range(len(sel)):
                af = int(freq0.get(int(tt[jj]), 0))
                if af >= 100:                            # only RARE tokens
                    continue
                q = psi_np[jj]                            # [4096]
                tk = int(tt[jj])
                # lm_head rank of true token (how q is actually scored)
                lm_score = lm_np @ q
                lm_rank = int((lm_score > lm_score[tk]).sum())
                # is it a MISS? (rough: rank outside ~top-2; we care about buried ones)
                # embed_tokens cosine-NN rank of true token
                emb_cos = (emb_np @ q) / (emb_norm * (np.linalg.norm(q) + 1e-9))
                emb_rank = int((emb_cos > emb_cos[tk]).sum())
                lm_ranks.append(lm_rank); emb_ranks.append(emb_rank); afs.append(af)
                n_rare_miss += 1
            if n_rare_miss >= 2000:    # cap for speed (NN over 152k per position is heavy)
                break
        lm_ranks = np.array(lm_ranks); emb_ranks = np.array(emb_ranks)
        np.savez("/tmp/donkey_retrieval.npz", lm_rank=lm_ranks, emb_rank=emb_ranks, act_freq=np.array(afs))
        print(f"[retrieval-probe] measured {len(lm_ranks)} rare-token positions (act_freq<100)")
        print()
        print("  DECISIVE COMPARISON -- true-token rank in lm_head-score vs embed_tokens-NN:")
        print(f"    lm_head rank:      median={np.median(lm_ranks):.0f}  mean={lm_ranks.mean():.0f}")
        print(f"    embed_tokens NN:   median={np.median(emb_ranks):.0f}  mean={emb_ranks.mean():.0f}")
        print()
        print("  Fraction of rare tokens RECOVERABLE by embed_tokens-NN within beam K:")
        for K in [5, 10, 20, 50, 100, 500]:
            lm_in = (lm_ranks < K).mean(); emb_in = (emb_ranks < K).mean()
            print(f"    beam {K:4d}: lm_head {100*lm_in:5.1f}%  embed-NN {100*emb_in:5.1f}%  "
                  f"(embed-NN {'WINS' if emb_in > lm_in + 0.02 else 'ties/loses'})")
        print()
        improved = (emb_ranks < lm_ranks).mean()
        big = (emb_ranks < lm_ranks / 2).mean()
        print(f"  embed-NN ranks true token HIGHER than lm_head for {100*improved:.0f}% of rare tokens")
        print(f"  embed-NN ranks it at <HALF the lm_head rank for {100*big:.0f}%")
        print()
        print("  VERDICT: if embed-NN median << lm_head median AND beam-50 embed-NN >> lm_head,")
        print("  then NN-retrieval in embed_tokens space RECOVERS rare tokens lm_head buries ->")
        print("  the retrieval head would lift the rare-token failure. If ranks are similar in both")
        print("  spaces, the latent doesn't point near the rare token -> retrieval won't help, ceiling real.")
        raise SystemExit(0)

    if args.probe_truehidden:
        from mlx.utils import tree_unflatten
        ck = np.load(args.probe_truehidden)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[truehidden-probe] loaded {args.probe_truehidden}")

        from collections import Counter
        freq0 = Counter(dslist[0]["tok"].tolist())

        def score_rank(h_raw_mx, tt_np):
            # h_raw_mx: [b,4096] RAW pre-norm hidden -> rmsnorm -> lm_head -> rank of true token
            hn = h_raw_mx * (1.0 / mx.sqrt(mx.mean(h_raw_mx * h_raw_mx, axis=-1, keepdims=True) + eps)) * norm_w
            logits = np.array(hn @ lm_w.T)
            ranks = np.empty(len(tt_np), np.int64)
            for j in range(len(tt_np)):
                ranks[j] = int((logits[j] > logits[j, tt_np[j]]).sum())
            return ranks

        idx = _frozen_idx("val", len(va_wi))
        d_ranks = []; t_ranks = []; afs = []
        seen = 0
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            # rare filter on emitted token
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            rare_mask = np.array([freq0.get(int(t), 0) < 100 for t in tt])
            if not rare_mask.any():
                continue
            rsel = sel[rare_mask]; tt_r = tt[rare_mask]
            # DONKEY hidden: psi(zhat) -> destandardize -> raw
            zw = model.phi(standardize(gather_window(dslist, va_cid[rsel], va_wi[rsel])))
            action = mx.array(va_aix[rsel].astype(np.int32))
            depth = mx.zeros(len(rsel), dtype=mx.int32)
            psi = model.psi(model.predict(zw, action, depth))
            donkey_raw = destandardize(psi)                      # raw hidden the donkey predicts
            dr = score_rank(donkey_raw, tt_r)
            # TRUE hidden: H[tgt] raw (NOT standardized, NOT destandardized)
            true_raw = np.empty((len(rsel), DMODEL), np.float32)
            for jj, s in enumerate(rsel):
                t = int(va_tgt[s]); true_raw[jj] = dslist[int(va_cid[s])]["H"][t]
            tr = score_rank(mx.array(true_raw), tt_r)
            d_ranks.extend(dr.tolist()); t_ranks.extend(tr.tolist())
            afs.extend([freq0.get(int(t), 0) for t in tt_r])
            seen += len(rsel)
            if seen >= 2000:
                break
        d = np.array(d_ranks); t = np.array(t_ranks)
        np.savez("/tmp/donkey_truehidden.npz", donkey_rank=d, true_rank=t, act_freq=np.array(afs))
        print(f"[truehidden-probe] {len(d)} rare-token positions (act_freq<100)")
        print()
        print("  RANK of the emitted RARE token under:")
        print(f"    DONKEY hidden:  median={np.median(d):.0f}  mean={d.mean():.0f}")
        print(f"    TRUE hidden:    median={np.median(t):.0f}  mean={t.mean():.0f}   <- the CEILING")
        print()
        print("  TRUE-hidden recovery (is the rare token even IN the true hidden's top-K?):")
        for K in [1, 2, 5, 10, 50, 100]:
            print(f"    top-{K:3d}: true-hidden {100*(t<K).mean():5.1f}%   donkey {100*(d<K).mean():5.1f}%")
        print()
        # The decomposition
        true_good = t < 5      # true hidden ranks it top-5 = the info IS there
        donkey_missed_but_fixable = true_good & (d >= 5)
        irreducible = t >= 50  # even true hidden buries it = temp=1 tail / not in hidden
        print("  DECOMPOSITION of rare-token failures:")
        print(f"    FIXABLE (true-hidden top-5 but donkey missed):  {100*donkey_missed_but_fixable.mean():.0f}%")
        print(f"    IRREDUCIBLE (true-hidden ALSO buries it, rank>=50): {100*irreducible.mean():.0f}%")
        print(f"    true-hidden gets it top-1: {100*(t<1).mean():.0f}%  (the trunk was confident here)")
        print()
        print("  VERDICT:")
        print("   - If TRUE-hidden ranks rare tokens HIGH (top-5 often) but donkey doesn't ->")
        print("     the info IS in the hidden, donkey's latent is the gap -> d=128/tail-loss FIXABLE.")
        print("   - If TRUE-hidden ALSO buries rare tokens (rank hundreds) -> the trunk emitted a")
        print("     temp=1 tail draw the hidden doesn't favor -> IRREDUCIBLE, no predictor recovers it.")
        raise SystemExit(0)

    _shape_scale = {"v": 0.0}
    if args.probe_dump_hidden:
        from mlx.utils import tree_unflatten
        ck = np.load(args.probe_dump_hidden)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[dump-hidden] loaded {args.probe_dump_hidden}")

        from collections import Counter
        freq0 = Counter(dslist[0]["tok"].tolist())

        # scan val frozen set, collect a few RARE-token MISSES and a couple HITS
        idx = _frozen_idx("val", len(va_wi))
        rare_miss = []   # (pos_idx, true_tok, rank)
        hits = []
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            zhat = model.predict(zw, action, depth)
            q = mx.softmax(lm_logits(model.psi(zhat)), axis=-1)
            qd = -mx.sort(-q, axis=-1); cm = mx.cumsum(qd, axis=-1)
            q_np = np.array(q); cm_np = np.array(cm); qd_np = np.array(qd)
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            for jj in range(len(sel)):
                qi = q_np[jj]; tk = int(tt[jj])
                rank = int((qi > qi[tk]).sum())
                cut90 = int((cm_np[jj] >= 0.90).argmax()) + 1
                hit90 = rank < cut90
                af = int(freq0.get(tk, 0))
                gpos = int(sel[jj])
                if (not hit90) and af < 100 and len(rare_miss) < 3:
                    rare_miss.append((gpos, tk, rank, af))
                elif hit90 and af < 1000 and len(hits) < 2:
                    hits.append((gpos, tk, rank, af))
            if len(rare_miss) >= 3 and len(hits) >= 2:
                break

        chosen = [("MISS", *r) for r in rare_miss] + [("HIT", *h) for h in hits]
        print(f"[dump-hidden] chosen {len(chosen)} positions (3 rare-miss + 2 hit)")
        print("FORMAT: each position prints PSI line then TRUE line, %.4f comma-sep, 4096 vals.\n")

        # collect arrays for all chosen positions
        rows = []
        for tag, gpos, tk, rank, af in chosen:
            zw = model.phi(standardize(gather_window(dslist, va_cid[gpos:gpos+1], va_wi[gpos:gpos+1])))
            action = mx.array(va_aix[gpos:gpos+1].astype(np.int32))
            depth = mx.zeros(1, dtype=mx.int32)
            zhat = model.predict(zw, action, depth)
            psi = np.array(model.psi(zhat))[0]
            t = int(va_tgt[gpos]); craw = int(va_cid[gpos])
            h_raw = dslist[craw]["H"][t]
            h_std = np.array(standardize(mx.array(h_raw[None])))[0]
            cosv = float(np.dot(psi, h_std) / (np.linalg.norm(psi) * np.linalg.norm(h_std) + 1e-9))
            rows.append((tag, gpos, tk, rank, af, cosv, psi.astype(np.float32), h_std.astype(np.float32)))

        # save raw arrays (small-ish .npz) regardless
        np.savez("/tmp/donkey_hidden_dump.npz",
                 **{f"{i}_psi": r[6] for i, r in enumerate(rows)},
                 **{f"{i}_true": r[7] for i, r in enumerate(rows)},
                 meta=np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=np.float64))
        print("[dump-hidden] saved raw arrays to /tmp/donkey_hidden_dump.npz")

        # try to render PNGs directly
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(rows)
            fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
            if n == 1:
                axes = axes[None, :]
            for i, (tag, gpos, tk, rank, af, cosv, psi, h_std) in enumerate(rows):
                P = psi.reshape(64, 64); T = h_std.reshape(64, 64); D = P - T
                vlim = max(abs(P).max(), abs(T).max())
                axes[i, 0].imshow(T, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
                axes[i, 0].set_title(f"{tag} TRUE  tok={tk} rk={rank} f={af}", fontsize=8)
                axes[i, 1].imshow(P, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
                axes[i, 1].set_title(f"PSI (pred)  cos={cosv:.4f}", fontsize=8)
                dl = abs(D).max()
                axes[i, 2].imshow(D, cmap="RdBu_r", vmin=-dl, vmax=dl)
                axes[i, 2].set_title(f"DIFF (psi-true) max|d|={dl:.2f}", fontsize=8)
                for a in axes[i]:
                    a.set_xticks([]); a.set_yticks([])
            plt.tight_layout()
            plt.savefig("/tmp/donkey_hidden_viz.png", dpi=110, bbox_inches="tight")
            print("[dump-hidden] SAVED IMAGE: /tmp/donkey_hidden_viz.png  <-- upload this to chat")
        except Exception as e:
            print(f"[dump-hidden] matplotlib unavailable ({e}). Falling back to 16x16 downsample print:")
            for tag, gpos, tk, rank, af, cosv, psi, h_std in rows:
                # downsample 64x64 -> 16x16 by 4x4 block mean
                def ds(v):
                    return v.reshape(64, 64).reshape(16, 4, 16, 4).mean(axis=(1, 3)).reshape(-1)
                print(f"=== {tag} pos={gpos} tok={tk} rank={rank} freq={af} cos={cosv:.4f} ===")
                print("PSI16:" + ",".join(f"{v:.3f}" for v in ds(psi)))
                print("TRUE16:" + ",".join(f"{v:.3f}" for v in ds(h_std)))
            print("[dump-hidden] paste the === 16x16 blocks (256 vals each).")
        raise SystemExit(0)

    if args.probe_roundtrip:
        from mlx.utils import tree_unflatten
        from collections import Counter
        ck = np.load(args.probe_roundtrip)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[roundtrip] loaded {args.probe_roundtrip}")
        freq0 = Counter(dslist[0]["tok"].tolist())
        idx = _frozen_idx("val", len(va_wi))
        acc = {k: [] for k in ["c_rec","c_pred","c_pp","mse_rec","mse_pred",
                               "k_true","k_rec","k_pred","rank_rec","rank_pred","freq",
                               "sz_rec","cap_rec","sz_pred","cap_pred"]}
        def _kurt(a):
            m = a.mean(axis=-1, keepdims=True); s = a.std(axis=-1, keepdims=True) + 1e-9
            return (((a - m) / s) ** 4).mean(axis=-1)
        def _cos(a, b):
            an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
            bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
            return (an * bn).sum(-1)
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            H = np.stack([dslist[int(va_cid[g])]["H"][int(va_tgt[g])] for g in sel])
            h_std = standardize(mx.array(H))
            rec = model.psi(model.phi(h_std))
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            pred = model.psi(model.predict(zw, action, depth))
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            for nm, dec in (("rank_rec", rec), ("rank_pred", pred)):
                lg = lm_logits(dec)
                tok_lg = mx.take_along_axis(lg, mx.array(tt)[:, None], axis=-1)
                rank = np.array(mx.sum(lg > tok_lg, axis=-1))
                acc[nm].append(rank)
                q = mx.softmax(lg, axis=-1)
                cm = np.array(mx.cumsum(-mx.sort(-q, axis=-1), axis=-1))
                cut = (cm >= 0.90).argmax(axis=-1) + 1
                tag = "rec" if nm == "rank_rec" else "pred"
                acc["sz_" + tag].append(cut)
                acc["cap_" + tag].append((rank < cut).astype(np.float64))
            h_np = np.array(h_std); r_np = np.array(rec); p_np = np.array(pred)
            acc["c_rec"].append(_cos(r_np, h_np)); acc["c_pred"].append(_cos(p_np, h_np))
            acc["c_pp"].append(_cos(p_np, r_np))
            acc["mse_rec"].append(((r_np - h_np) ** 2).mean(-1))
            acc["mse_pred"].append(((p_np - h_np) ** 2).mean(-1))
            acc["k_true"].append(_kurt(h_np)); acc["k_rec"].append(_kurt(r_np)); acc["k_pred"].append(_kurt(p_np))
            acc["freq"].append(np.array([freq0.get(int(t), 0) for t in tt]))
        A = {k: np.concatenate(v) for k, v in acc.items()}
        rare = A["freq"] < 100
        print(f"\n[roundtrip] N={len(A['c_rec'])}  (rare<100: {int(rare.sum())})")
        print("HIDDEN-SPACE RECOVERY (cos to true h_std):")
        print(f"  roundtrip psi(phi(h))   : {A['c_rec'].mean():.3f} +/- {A['c_rec'].std():.3f}")
        print(f"  prediction psi(zhat)    : {A['c_pred'].mean():.3f} +/- {A['c_pred'].std():.3f}")
        print(f"  pred vs roundtrip       : {A['c_pp'].mean():.3f}   (high => predictor adds little decode error)")
        print(f"  MSE: roundtrip {A['mse_rec'].mean():.4f}  prediction {A['mse_pred'].mean():.4f}")
        print("SPIKES (per-sample kurtosis over 4096 dims):")
        print(f"  true {A['k_true'].mean():.1f}   roundtrip {A['k_rec'].mean():.1f}   prediction {A['k_pred'].mean():.1f}")
        print("TOKEN READOUT (rank of emitted token; median [overall / rare<100 / common]):")
        for nm in ("rank_rec", "rank_pred"):
            r = A[nm]
            print(f"  {nm}: {np.median(r):.0f} / {np.median(r[rare]):.0f} / {np.median(r[~rare]):.0f}")
        print("READOUT WIDTH (90%-mass set size / cap90):")
        print(f"  roundtrip : sz {A['sz_rec'].mean():.2f}  cap90 {A['cap_rec'].mean():.1%}   rare: sz {A['sz_rec'][rare].mean():.2f} cap {A['cap_rec'][rare].mean():.1%}")
        print(f"  prediction: sz {A['sz_pred'].mean():.2f}  cap90 {A['cap_pred'].mean():.1%}   rare: sz {A['sz_pred'][rare].mean():.2f} cap {A['cap_pred'][rare].mean():.1%}")
        np.savez("/tmp/donkey_roundtrip.npz", **A)
        print("[roundtrip] saved /tmp/donkey_roundtrip.npz")
        raise SystemExit(0)

    if args.probe_recency:
        from mlx.utils import tree_unflatten
        from collections import Counter
        ck = np.load(args.probe_recency)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[recency] loaded {args.probe_recency}")
        freq0 = Counter(dslist[0]["tok"].tolist())
        idx = _frozen_idx("val", len(va_wi))
        D = {k: [] for k in ["dist", "freq", "rank_rec", "rank_pred", "cap_rec", "cap_pred"]}
        STOPS = (151645, 151643)
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            H = np.stack([dslist[int(va_cid[g])]["H"][int(va_tgt[g])] for g in sel])
            h_std = standardize(mx.array(H))
            rec = model.psi(model.phi(h_std))
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            pred = model.psi(model.predict(zw, action, depth))
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            for nm, dec in (("rec", rec), ("pred", pred)):
                lg = lm_logits(dec)
                tok_lg = mx.take_along_axis(lg, mx.array(tt)[:, None], axis=-1)
                rank = np.array(mx.sum(lg > tok_lg, axis=-1))
                q = mx.softmax(lg, axis=-1)
                cm = np.array(mx.cumsum(-mx.sort(-q, axis=-1), axis=-1))
                cut = (cm >= 0.90).argmax(axis=-1) + 1
                D["rank_" + nm].append(rank)
                D["cap_" + nm].append((rank < cut).astype(np.float64))
            for jj in range(len(sel)):
                g = int(sel[jj]); craw = int(va_cid[g]); t = int(va_tgt[g]); tk = int(tt[jj])
                s0 = max(0, t - 4096)
                seg = np.asarray(dslist[craw]["tok"][s0:t])
                st = np.where((seg == STOPS[0]) | (seg == STOPS[1]))[0]
                lo = int(st[-1]) + 1 if len(st) else 0
                m = np.where(seg[lo:] == tk)[0]
                D["dist"].append(len(seg) - (lo + int(m[-1])) if len(m) else -1)
                D["freq"].append(freq0.get(tk, 0))
        D = {k: np.asarray(np.concatenate(v) if isinstance(v[0], np.ndarray) else v) for k, v in D.items()}
        edges = [(1, 4, "1-4"), (5, 13, "5-13 ~win"), (14, 64, "14-64"), (65, 256, "65-256"),
                 (257, 1024, "257-1024"), (1025, 10**9, ">1024"), (-1, -1, "none/prompt")]
        for title, mask in (("RARE (freq<100)", D["freq"] < 100), ("ALL", np.ones(len(D["dist"]), bool))):
            print(f"\n=== {title}  (n={int(mask.sum())}) ===")
            hdr_d = "last-occurrence"; hdr_n = "n"
            print(f"{hdr_d:<16}{hdr_n:>6}{'pred cap90':>12}{'rec cap90':>11}{'pred mdrk':>11}{'rec mdrk':>10}")
            for a, b2, lab in edges:
                sel2 = mask & ((D["dist"] == -1) if a == -1 else ((D["dist"] >= a) & (D["dist"] <= b2)))
                n = int(sel2.sum())
                if n == 0:
                    print(f"{lab:<16}{0:>6}")
                    continue
                print(f"{lab:<16}{n:>6}{D['cap_pred'][sel2].mean():>11.1%}{D['cap_rec'][sel2].mean():>10.1%}"
                      f"{np.median(D['rank_pred'][sel2]):>11.0f}{np.median(D['rank_rec'][sel2]):>10.0f}")
        np.savez("/tmp/donkey_recency.npz", **D)
        print("\n[recency] saved /tmp/donkey_recency.npz")
        raise SystemExit(0)

    if args.probe_interp:
        from mlx.utils import tree_unflatten
        from collections import Counter
        ck = np.load(args.probe_interp)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[interp] loaded {args.probe_interp}")
        freq0 = Counter(dslist[0]["tok"].tolist())
        idx = _frozen_idx("val", len(va_wi))[:2400]
        alphas = np.linspace(0.0, 1.0, 11)
        NA = len(alphas)
        R_l, I_l, P_l, S_l, F_l = [], [], [], [], []
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            H = np.stack([dslist[int(va_cid[g])]["H"][int(va_tgt[g])] for g in sel])
            h_std = standardize(mx.array(H))
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            pred = model.psi(model.predict(zw, action, depth))
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            tcol = mx.array(tt)[:, None]
            rB = np.zeros((len(sel), NA), dtype=np.int32)
            iB = np.zeros((len(sel), NA), dtype=np.float32)
            pB = np.zeros((len(sel), NA), dtype=np.float32)
            sB = np.zeros((len(sel), NA), dtype=np.float32)
            for ai in range(NA):
                a = float(alphas[ai])
                x = (1.0 - a) * pred + a * h_std
                lg = lm_logits(x)
                tok_lg = mx.take_along_axis(lg, tcol, axis=-1)
                rank = np.array(mx.sum(lg > tok_lg, axis=-1))
                q = mx.softmax(lg, axis=-1)
                cm = np.array(mx.cumsum(-mx.sort(-q, axis=-1), axis=-1))
                cut = (cm >= 0.90).argmax(axis=-1) + 1
                rB[:, ai] = rank
                iB[:, ai] = (rank < cut).astype(np.float32)
                pB[:, ai] = np.array(mx.take_along_axis(q, tcol, axis=-1))[:, 0]
                sB[:, ai] = cut
            R_l.append(rB); I_l.append(iB); P_l.append(pB); S_l.append(sB)
            F_l.append(np.array([freq0.get(int(t), 0) for t in tt]))
        R = np.concatenate(R_l); I = np.concatenate(I_l); P = np.concatenate(P_l)
        S = np.concatenate(S_l); F = np.concatenate(F_l)
        rare = F < 100
        miss0 = I[:, 0] < 0.5
        rmiss = rare & miss0
        print(f"\n[interp] N={len(F)}  rare={int(rare.sum())}  rare-miss@a0={int(rmiss.sum())}")
        print(f"{'alpha':>6}{'ALL cap90':>11}{'RARE cap90':>12}{'Rmiss cap90':>13}{'Rmiss mdrk':>12}{'Rmiss pTrue':>13}{'sz':>7}")
        for ai in range(NA):
            print(f"{alphas[ai]:>6.1f}{I[:, ai].mean():>10.1%}{I[rare, ai].mean():>11.1%}"
                  f"{I[rmiss, ai].mean():>12.1%}{np.median(R[rmiss, ai]):>12.0f}"
                  f"{P[rmiss, ai].mean():>13.4f}{S[:, ai].mean():>7.2f}")
        astar = np.full(len(F), np.nan)
        for j in range(len(F)):
            w = np.where(I[j] > 0.5)[0]
            if len(w):
                astar[j] = alphas[w[0]]
        ok = rmiss & ~np.isnan(astar)
        if ok.sum():
            qs = np.percentile(astar[ok], [10, 25, 50, 75, 90])
            print(f"\nflip-point a* (rare-miss, n={int(ok.sum())}; {int((rmiss & np.isnan(astar)).sum())} never flip):")
            print("  p10/p25/p50/p75/p90: " + " ".join(f"{v:.2f}" for v in qs))
        np.savez("/tmp/donkey_interp.npz", alphas=alphas, ranks=R, inset=I, ptrue=P, sz=S, freq=F)
        print("[interp] saved /tmp/donkey_interp.npz")
        raise SystemExit(0)

    if args.probe_latent:
        from mlx.utils import tree_unflatten
        from collections import Counter
        ck = np.load(args.probe_latent)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[latent] loaded {args.probe_latent}")
        freq0 = Counter(dslist[0]["tok"].tolist())
        idx = _frozen_idx("val", len(va_wi))[:2400]
        C_l, M_l, RN_l, I_l, F_l = [], [], [], [], []
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            H = np.stack([dslist[int(va_cid[g])]["H"][int(va_tgt[g])] for g in sel])
            h_std = standardize(mx.array(H))
            z_tgt = np.array(model.phi(h_std))
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            zhat_mx = model.predict(zw, action, depth)
            zhat = np.array(zhat_mx)
            r = zhat - z_tgt
            zn = zhat / (np.linalg.norm(zhat, axis=-1, keepdims=True) + 1e-9)
            tn = z_tgt / (np.linalg.norm(z_tgt, axis=-1, keepdims=True) + 1e-9)
            C_l.append((zn * tn).sum(-1))
            M_l.append((r ** 2).mean(-1))
            RN_l.append(np.linalg.norm(r, axis=-1) / (np.linalg.norm(z_tgt, axis=-1) + 1e-9))
            lg = lm_logits(model.psi(zhat_mx))
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            tok_lg = mx.take_along_axis(lg, mx.array(tt)[:, None], axis=-1)
            rank = np.array(mx.sum(lg > tok_lg, axis=-1))
            q = mx.softmax(lg, axis=-1)
            cm = np.array(mx.cumsum(-mx.sort(-q, axis=-1), axis=-1))
            cut = (cm >= 0.90).argmax(axis=-1) + 1
            I_l.append((rank < cut).astype(np.float32))
            F_l.append(np.array([freq0.get(int(t), 0) for t in tt]))
        C = np.concatenate(C_l); M = np.concatenate(M_l); RN = np.concatenate(RN_l)
        I = np.concatenate(I_l); F = np.concatenate(F_l)
        rare = F < 100
        miss = I < 0.5
        print(f"\n[latent] N={len(F)}  rare={int(rare.sum())}  miss={int(miss.sum())}")
        print(f"{'stratum':<14}{'n':>6}{'cosL mean':>11}{'cosL p10':>10}{'mse/dim':>9}{'|r|/|z| p50':>13}{'p90':>7}")
        strata = [("ALL", np.ones(len(F), bool)), ("hit", ~miss), ("miss", miss),
                  ("RARE hit", rare & ~miss), ("RARE miss", rare & miss),
                  ("COMM hit", ~rare & ~miss), ("COMM miss", ~rare & miss)]
        for lab, mk in strata:
            n = int(mk.sum())
            if n == 0:
                print(f"{lab:<14}{0:>6}")
                continue
            print(f"{lab:<14}{n:>6}{C[mk].mean():>11.4f}{np.percentile(C[mk],10):>10.4f}"
                  f"{M[mk].mean():>9.4f}{np.percentile(RN[mk],50):>13.3f}{np.percentile(RN[mk],90):>7.3f}")
        np.savez("/tmp/donkey_latent.npz", cosL=C, mse=M, relnorm=RN, inset=I, freq=F)
        print("[latent] saved /tmp/donkey_latent.npz  (same 2400 positions as interp -> cross-tab ok)")
        raise SystemExit(0)

    if args.probe_accept:
        from mlx.utils import tree_unflatten
        from collections import Counter
        ck = np.load(args.probe_accept)
        wkeys = [k for k in ck.files if not k.startswith("__")]
        model.update(tree_unflatten([(k, mx.array(ck[k])) for k in wkeys]))
        mx.eval(model.parameters())
        print(f"[accept] loaded {args.probe_accept}")
        freq0 = Counter(dslist[0]["tok"].tolist())
        idx = _frozen_idx("val", len(va_wi))
        AL_l, AU_l, GR_l, MT_l, CV_l, F_l, RC_l = [], [], [], [], [], [], []
        for b in range(0, len(idx), args.bs):
            sel = idx[b:b + args.bs]
            zw = model.phi(standardize(gather_window(dslist, va_cid[sel], va_wi[sel])))
            action = mx.array(va_aix[sel].astype(np.int32))
            depth = mx.zeros(len(sel), dtype=mx.int32)
            q = mx.softmax(lm_logits(model.psi(model.predict(zw, action, depth))), axis=-1)
            H = np.stack([dslist[int(va_cid[g])]["H"][int(va_tgt[g])] for g in sel])
            q_rec = mx.softmax(lm_logits(model.psi(model.phi(standardize(mx.array(H))))), axis=-1)
            P_ids, P_pr = gather_nucleus(dslist, va_cid[sel], va_tgt[sel])
            P_ids = np.array(P_ids).astype(np.int64)
            P_pr = np.array(P_pr).astype(np.float32)
            pmask = (P_pr > 0)
            q_at = np.array(mx.take_along_axis(q, mx.array(P_ids), axis=-1)) * pmask
            rec_at = np.array(mx.take_along_axis(q_rec, mx.array(P_ids), axis=-1)) * pmask
            cover = P_pr.sum(-1)
            acc_low = np.minimum(P_pr, q_at).sum(-1)
            acc_rec = np.minimum(P_pr, rec_at).sum(-1)
            q_out = np.clip(1.0 - q_at.sum(-1), 0.0, 1.0)
            acc_up = acc_low + np.minimum(np.clip(1.0 - cover, 0.0, 1.0), q_out)
            xhat = np.array(mx.argmax(q, axis=-1))
            hitK = (P_ids == xhat[:, None]) & pmask
            greedy = (P_pr * hitK).sum(-1)
            tp = P_pr.argmax(1)
            match = (P_ids[np.arange(len(sel)), tp] == xhat).astype(np.float32)
            tt = np.array(tgt_tokens(dslist, va_cid[sel], va_tgt[sel])).astype(np.int64)
            AL_l.append(acc_low); AU_l.append(acc_up); GR_l.append(greedy); MT_l.append(match); CV_l.append(cover); RC_l.append(acc_rec)
            F_l.append(np.array([freq0.get(int(t), 0) for t in tt]))
        AL = np.concatenate(AL_l); AU = np.concatenate(AU_l); GR = np.concatenate(GR_l); RC = np.concatenate(RC_l)
        MT = np.concatenate(MT_l); CV = np.concatenate(CV_l); F = np.concatenate(F_l)
        rare = F < 100
        print(f"\n[accept] N={len(F)}  rare={int(rare.sum())}  p top-K coverage mean {CV.mean():.4f} (min {CV.min():.3f})")
        print(f"{'stratum':<8}{'fullq acc (lo..up)':>22}{'greedy acc':>12}{'argmax match':>14}{'TEACHER acc':>13}")
        for lab, mk in (("ALL", np.ones(len(F), bool)), ("RARE", rare), ("COMMON", ~rare)):
            print(f"{lab:<8}{AL[mk].mean():>11.3f} ..{AU[mk].mean():>7.3f}{GR[mk].mean():>12.3f}{MT[mk].mean():>14.3f}{RC[mk].mean():>13.3f}")
        a = AL.mean()
        print(f"\nexpected accepted-run length at temp 1 (full-q, lower bound): {a/(1-a):.2f} tokens")
        r = RC.mean()
        print(f"TEACHER ceiling: acceptance {r:.3f} -> run {r/(1-r):.2f} tokens (distill headroom: {r-a:+.3f})")
        g = GR.mean()
        print(f"expected accepted-run length at temp 1 (greedy draft):         {g/(1-g):.2f} tokens")
        np.savez("/tmp/donkey_accept.npz", acc_low=AL, acc_up=AU, greedy=GR, match=MT, cover=CV, freq=F, acc_rec=RC)
        print("[accept] saved /tmp/donkey_accept.npz")
        raise SystemExit(0)

    for ep in range(start_ep, args.epochs + 1):
        # ramp lam_shape: 0 during warmup, linear to 1.0 over shape_warmup epochs after
        if args.shape_warmup > 0:
            _shape_scale["v"] = max(0.0, min(1.0, (ep - start_ep) / float(args.shape_warmup)))
        else:
            _shape_scale["v"] = 1.0
        t0 = time.time()
        ec, ew, ea, et = [], [], [], []
        for ci, (twi, taix, ttgt) in enumerate(train_pools):
            take = min(per_n, len(twi))
            dsel = rng.choice(len(twi), take, replace=False)
            ec.append(np.full(take, ci, np.int32)); ew.append(twi[dsel]); ea.append(taix[dsel]); et.append(ttgt[dsel])
        ec = np.concatenate(ec); ew = np.concatenate(ew); ea = np.concatenate(ea); et = np.concatenate(et)
        order = rng.permutation(len(ew))
        ec, ew, ea, et = ec[order], ew[order], ea[order], et[order]

        tot_loss = 0.0; nb = 0
        for b in range(0, len(ew), args.bs):
            sel = slice(b, b + args.bs)
            zw = model.phi(standardize(gather_window(dslist, ec[sel], ew[sel])))
            action = mx.array(ea[sel].astype(np.int32))
            depth = mx.zeros(len(ew[sel]), dtype=mx.int32)
            h_tgt = standardize(gather_window(dslist, ec[sel], et[sel])[:, -1, :])
            p_ids, p_probs = gather_nucleus(dslist, ec[sel], et[sel])
            (L, aux), grads = loss_and_grad(model, zw, action, depth, h_tgt, p_ids, p_probs)
            grads = optim.clip_grad_norm(grads, 1.0)[0]
            opt.update(model, grads); mx.eval(model.parameters(), opt.state)
            tot_loss += float(L); nb += 1

        vh = seq_hit(dslist, va_cid, va_wi, va_aix, va_tgt, frozen_key="val")
        th = seq_hit(dslist, ec, ew, ea, et, nmax=4000)
        hh = seq_hit(hold[0], hold[1], hold[2], hold[3], hold[4], frozen_key="hold") if hold is not None else None
        vc = capture_at(dslist, va_cid, va_wi, va_aix, va_tgt, frozen_key="val")
        vcap, vdsz, vtsz = vc[0.9]; vcap99, vdsz99, vtsz99 = vc[0.99]
        hc = capture_at(hold[0], hold[1], hold[2], hold[3], hold[4], frozen_key="hold") if hold is not None else None
        if hc is not None:
            hcap, hdsz, htsz = hc[0.9]; hcap99, hdsz99, htsz99 = hc[0.99]
        else:
            hcap = hcap99 = None
        gap = th - vh; star = ""
        if best is None or vh > best:
            best = vh; bad = 0; star = " *"
            if args.out:
                import os as _os
                flat = _flatten(dict(model.parameters()))
                payload = {k: np.array(v) for k, v in flat.items()}
                if args.save_opt:
                    optflat = _flatten(dict(opt.state))
                    for k, v in optflat.items():
                        payload["__opt__" + k] = np.array(v)
                    payload["__epoch__"] = np.array(ep)
                _stem, _ext = _os.path.splitext(args.out)
                np.savez(f"{_stem}_ep{ep:03d}_val{vh*100:05.2f}{_ext}", **payload)
                np.savez(args.out, **payload)
        else:
            bad += 1
        if args.out and hh is not None and (besth is None or hh > besth):
            besth = hh
            import os as _os
            flat = _flatten(dict(model.parameters()))
            payload = {k: np.array(v) for k, v in flat.items()}
            _stem, _ext = _os.path.splitext(args.out)
            np.savez(f"{_stem}_besthold_ep{ep:03d}_hold{hh*100:05.2f}{_ext}", **payload)
        cos_m, jsd_m, gate_m, Lcos, Ldist, Lrec, Lsig, Ll2, Lshape, dcos_m, Lzl2, Lpm, Lkd = aux
        hstr = f" | holdout({args.holdout}) {hh*100:.2f}%" if hh is not None else ""
        print(f"[v4 ep{ep}/{args.epochs}] loss {tot_loss/nb:.4f} | "
              f"train {th*100:.2f}% | val(in) {vh*100:.2f}%{star} | gap {gap*100:+.1f}{hstr} | "
              f"cos {float(cos_m):.3f} dcos {float(dcos_m):.3f} jsd {float(jsd_m):.3f} gate {float(gate_m):.2f} | "
              f"L[cos {float(Lcos):.3f} dist {float(Ldist):.3f} rec {float(Lrec):.3f} sig {float(Lsig):.3f} l2 {float(Ll2):.3f} shape {float(Lshape):.3f} zl2 {float(Lzl2):.3f} pm {float(Lpm):.3f} kd {float(Lkd):.3f}] | "
              f"cap90(v) {vcap*100:.1f}%/{vcap99*100:.1f}% sz {vdsz:.1f}/{vtsz:.1f}" +
              (f" cap90(h) {hcap*100:.1f}%/{hcap99*100:.1f}% sz {hdsz:.1f}/{htsz:.1f}" if hcap is not None else "") +
              f" | bad {bad} | {time.time()-t0:.1f}s")
        if bad >= args.patience:
            print(f"[v4] early stop (patience {args.patience}); best val {best*100:.2f}%")
            break


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                if hasattr(vv, "items"):
                    out.update(_flatten(dict(vv), f"{key}.{i}."))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", choices=sorted(CORPORA))
    ap.add_argument("--holdout", type=str, default=None)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dp", type=int, default=256)
    ap.add_argument("--enc", type=int, default=ENC_LAYERS)
    ap.add_argument("--dec", type=int, default=DEC_LAYERS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-corpus-n", type=int, default=0, help="fresh draw of N PER corpus EACH epoch (0=min train pool)")
    ap.add_argument("--val-n", type=int, default=0, help="FROZEN val pool size PER corpus (0=10pct)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--eval-n", type=int, default=8000, help="frozen eval subset size for val+holdout")
    ap.add_argument("--save-opt", action="store_true", help="also save optimizer state for resume")
    ap.add_argument("--resume", type=str, default=None, help="resume from a checkpoint .npz")
    ap.add_argument("--probe-failure", type=str, default=None, help="load ckpt, run failure-mode probe, exit")
    ap.add_argument("--probe-retrieval", type=str, default=None, help="load ckpt, test rare-token NN-retrieval feasibility, exit")
    ap.add_argument("--probe-truehidden", type=str, default=None, help="load ckpt, compare donkey vs TRUE-hidden rare-token rank (fixable vs irreducible), exit")
    ap.add_argument("--probe-dump-hidden", type=str, default=None, help="load ckpt, dump psi(zhat) vs true hidden for a few miss/hit positions (for 64x64 viz), exit")
    ap.add_argument("--probe-roundtrip", type=str, default=None, help="load ckpt, measure psi(phi(h)) vs h recovery (AE roundtrip vs prediction decomposition), exit")
    ap.add_argument("--probe-recency", type=str, default=None, help="load ckpt, stratify pred-vs-roundtrip capture by distance to target token last occurrence, exit")
    ap.add_argument("--probe-interp", type=str, default=None, help="load ckpt, interpolate hidden space pred->true, track readout vs alpha, exit")
    ap.add_argument("--probe-latent", type=str, default=None, help="load ckpt, measure latent distance zhat vs phi(h) stratified by hit/miss/rare, exit")
    ap.add_argument("--probe-accept", type=str, default=None, help="load ckpt, compute temp-1 speculative acceptance sum min(p,q) and greedy-draft acceptance on frozen val, exit")
    ap.add_argument("--lam-cos", type=float, default=0.3)
    ap.add_argument("--lam-dist", type=float, default=1.0)
    ap.add_argument("--lam-rec", type=float, default=0.5)
    ap.add_argument("--lam-sig", type=float, default=0.1)
    ap.add_argument("--lam-l2", type=float, default=0.0, help="L2 on psi(zhat) vs h_tgt_std (residual fidelity)")
    ap.add_argument("--lam-shape", type=float, default=0.0, help="penalize q flatter than p (one-sided, cos-gated set-size control)")
    ap.add_argument("--shape-warmup", type=int, default=0, help="epochs to ramp lam_shape from 0 to target (0=no ramp)")
    ap.add_argument("--lam-zl2", type=float, default=0.0, help="latent L2: mean((zhat-z_tgt)^2), prices the latent residual carrying rare-token signal")
    ap.add_argument("--lam-pm", type=float, default=0.0, help="psi-match: ||psi(zhat)-sg(psi(z_tgt))||^2, Jacobian-weighted reachable decode target")
    ap.add_argument("--lam-kd", type=float, default=0.0, help="logit distill: JSD(q, sg(q_rec)) toward the teacher readout")
    ap.add_argument("--gate-tau", type=float, default=0.5)
    ap.add_argument("--gate-sharp", type=float, default=8.0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--coverage", type=float, default=0.90)
    args = ap.parse_args()
    assert args.K == 0, "this build is WAVE 0 only; K>0 not yet implemented"
    if args.holdout and args.holdout in args.corpora:
        raise SystemExit(f"holdout {args.holdout} is also a train corpus -- remove it")
    train(args)


if __name__ == "__main__":
    main()
