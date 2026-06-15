#!/usr/bin/env python3
"""donkey_train.py - LeWM-faithful latent world model over MiMo-V2.5 trunk hiddens (MLX)."""
import argparse, json, os, time, glob
import numpy as np

BASE   = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
MODEL  = os.environ.get("MIMO_DIR", "/Volumes/TB5/llm/MiMo-V2.5-MLX")
CKPTS  = os.environ.get("DONKEY_CKPTS", os.path.expanduser("~/projects/donkey/ckpts"))
CORPORA = {"humaneval":"humaneval","mbpp":"mbpp","codealpaca":"codealpaca_20k","codealpaca_20k":"codealpaca_20k"}
STOP = (151645, 151643)
DMODEL = 4096
SEED = 0
SIGREG_LAMBDA = 0.1
SIGREG_M      = 1024
WINDOW        = 13
PRED_LAYERS   = 6
ENC_LAYERS    = 2
DEC_LAYERS    = 2

def load_trace(corpus):
    d = os.path.join(BASE, "traces", CORPORA[corpus])
    h = np.fromfile(os.path.join(d, "lastHiddenState.bin"), dtype=np.float32).reshape(-1, DMODEL)
    tok = np.fromfile(os.path.join(d, "tokens.bin"), dtype=np.int32)
    pid = np.fromfile(os.path.join(d, "prompt_idx.bin"), dtype=np.int32)
    assert h.shape[0] == tok.shape[0] == pid.shape[0], (h.shape, tok.shape, pid.shape)
    # Standardize per-dim: raw hiddens have std ~65 (per-dim 21-132), which makes MSE huge
    # and drowns SIGReg. Normalize so MSE/SIGReg/latent-scale all live at O(1) and cooperate.
    mu = h.mean(0, keepdims=True); sd = h.std(0, keepdims=True) + 1e-6
    h = ((h - mu) / sd).astype(np.float32)
    print(f"[data] standardized hiddens (was std {float(sd.mean()):.1f}); now ~unit per-dim")
    return h, tok, pid

def load_ipr_weights(corpus):
    """Per-position IPR weight w = sum(p_i^2) over the renormalized 99%-nucleus probs.
    Dirac -> 1 (next state well-determined, full gradient); flat -> 1/k (ambiguous, ~0 gradient)."""
    d = os.path.join(BASE, "traces", CORPORA[corpus])
    counts = np.fromfile(os.path.join(d, "topp_counts.bin"), np.int32)
    probs = np.fromfile(os.path.join(d, "topp_probs.bin"), np.float16).astype(np.float64)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    w = np.ones(len(counts), np.float64)
    for i in range(len(counts)):
        p = probs[off[i]:off[i+1]]
        s = p.sum()
        if s > 0:
            p = p / s
            w[i] = float((p * p).sum())
    return w.astype(np.float32)

def build_transitions(h, tok, pid, W, keep_only_terminated=True):
    """Skip prompts whose final token is not a STOP (cap-hit runaways): their late
    trajectory is non-terminating drift, off-distribution for the drafter."""
    N = h.shape[0]
    boundaries = np.where(np.diff(pid) != 0)[0] + 1
    spans = np.split(np.arange(N), boundaries)
    win_idx, act, tgt, srcpos = [], [], [], []
    n_skip = 0
    for span in spans:
        if span.size < 2:
            continue
        if keep_only_terminated and int(tok[span[-1]]) not in STOP:
            n_skip += 1
            continue
        s0 = span[0]
        for j in range(span.size - 1):
            idxs = [span[max(0, j - (W - 1) + k)] if (j - (W - 1) + k) >= 0 else s0 for k in range(W)]
            win_idx.append(idxs); act.append(tok[span[j]]); tgt.append(span[j + 1]); srcpos.append(span[j])
    if keep_only_terminated and n_skip:
        print(f"[data] skipped {n_skip} cap-hit/non-terminated prompts")
    return (np.asarray(win_idx, np.int64), np.asarray(act, np.int32),
            np.asarray(tgt, np.int64), np.asarray(srcpos, np.int64))

def make_sigreg(mx, mode, M=SIGREG_M):
    knots = mx.linspace(-5.0, 5.0, 33)
    target = mx.exp(-(knots ** 2) / 2.0)[None, :]
    weight = mx.exp(-(knots ** 2) / 2.0)[None, :]
    def sigreg(Z):
        Zf = Z.reshape(-1, Z.shape[-1]); d = Zf.shape[-1]
        u = mx.random.normal((d, M)); u = u / mx.sqrt(mx.sum(u * u, axis=0, keepdims=True) + 1e-9)
        H = Zf @ u
        if mode == "shape":
            mu = mx.mean(H, axis=0, keepdims=True)
            sd = mx.sqrt(mx.var(H, axis=0, keepdims=True) + 1e-6)
            H = (H - mu) / sd
        arg = H[:, :, None] * knots[None, None, :]
        re = mx.mean(mx.cos(arg), axis=0)
        im = mx.mean(mx.sin(arg), axis=0)
        disc = ((re - target) ** 2 + im ** 2) * weight
        return mx.mean(mx.sum(disc, axis=-1))
    return sigreg

def make_models(mx, nn, d, dp, n_enc, n_dec, n_actions, option):
    class Block(nn.Module):
        def __init__(self, w, adaln=False, heads=8):
            super().__init__()
            self.n1 = nn.RMSNorm(w); self.n2 = nn.RMSNorm(w)
            self.q = nn.Linear(w, w, bias=False); self.k = nn.Linear(w, w, bias=False)
            self.v = nn.Linear(w, w, bias=False); self.o = nn.Linear(w, w, bias=False)
            self.f1 = nn.Linear(w, 4 * w); self.f2 = nn.Linear(4 * w, w)
            self.heads = heads; self.w = w; self.adaln = adaln
            if adaln:
                self.ada = nn.Linear(w, 4 * w)
                self.ada.weight = mx.zeros_like(self.ada.weight)
                self.ada.bias = mx.zeros_like(self.ada.bias)
        def __call__(self, x, a_emb=None, mask=None):
            B, T, W = x.shape
            if self.adaln and a_emb is not None:
                s1, b1, s2, b2 = mx.split(self.ada(a_emb)[:, None, :], 4, axis=-1)
            else:
                s1 = b1 = s2 = b2 = 0.0
            y = self.n1(x) * (1 + s1) + b1
            hd = self.w // self.heads
            q = self.q(y).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
            k = self.k(y).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
            v = self.v(y).reshape(B, T, self.heads, hd).transpose(0, 2, 1, 3)
            att = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
            if mask is not None: att = att + mask
            att = mx.softmax(att, axis=-1)
            o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, W)
            x = x + self.o(o)
            y = self.n2(x) * (1 + s2) + b2
            return x + self.f2(nn.gelu(self.f1(y)))
    class Encoder(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Linear(DMODEL, dp)
            s.blocks = [Block(dp) for _ in range(n_enc)]
            s.head = nn.Linear(dp, d)
        def __call__(s, h):
            x = s.inp(h)
            for b in s.blocks: x = b(x)
            return s.head(x)
    class Predictor(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Linear(d, dp)
            s.pos = mx.random.normal((WINDOW, dp)) * 0.02
            s.blocks = [Block(dp, adaln=True) for _ in range(PRED_LAYERS)]
            s.head = nn.Linear(dp, d)
        def __call__(s, z, a_emb, mask):
            x = s.inp(z) + s.pos[None, : z.shape[1], :]
            for b in s.blocks: x = b(x, a_emb=a_emb, mask=mask)
            return s.head(x[:, -1, :])
    class Decoder(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Linear(d, dp)
            s.blocks = [Block(dp) for _ in range(n_dec)]
            s.head = nn.Linear(dp, DMODEL)
        def __call__(s, z):
            x = s.inp(z)[:, None, :]
            for b in s.blocks: x = b(x)
            return s.head(x[:, 0, :])
    class Ident(nn.Module):
        def __call__(s, x): return x
    aemb = nn.Embedding(n_actions, dp)
    if option == "a":
        return Ident(), Predictor(), Ident(), aemb
    return Encoder(), Predictor(), Decoder(), aemb

def causal_mask(mx, W):
    return mx.triu(mx.full((W, W), -1e9), k=1)[None, None]

def n_params(mx, module):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(module.parameters()))

def sram_line(p):
    fp16 = p * 2 / 1024**2; q4 = p * 0.5 / 1024**2; tr = p * 4 / 1024**2
    return (f"{p/1e6:.1f}M params | fp16 {fp16:.0f}MB | 4bit {q4:.0f}MB | "
            f"train~{tr:.0f}MB+act | {'FITS' if tr < 128 else 'TRAIN>128 (infer/4bit)'} @128MB")

def load_lm_head(mx):
    cfg = json.load(open(os.path.join(MODEL, "config.json"))); cfg["model_type"] = "mimo_v2_block_fp8"
    norm_w = lm_w = emb_w = None
    for s in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
        w = mx.load(s)
        for k, v in w.items():
            if k.endswith("model.norm.weight") or k == "norm.weight": norm_w = v
            if k.endswith("lm_head.weight"): lm_w = v
            if k.endswith("embed_tokens.weight"): emb_w = v
    if lm_w is None: lm_w = emb_w
    assert lm_w is not None and norm_w is not None, "could not load lm_head/norm"
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    def project(h):
        x = h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps) * norm_w
        return x @ lm_w.T
    return project

def save_ckpt(mx, path, wm, meta):
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(wm.parameters()))
    mx.savez(path, **flat)
    json.dump(meta, open(path + ".meta.json", "w"))

def load_ckpt(mx, wm, path):
    from mlx.utils import tree_unflatten
    wm.update(tree_unflatten(list(mx.load(path).items())))
    return json.load(open(path + ".meta.json"))

def phase1(args, mx, nn, optim):
    mx.random.seed(SEED); rng = np.random.default_rng(SEED)
    h, tok, pid = load_trace(args.corpus)
    wi, act, tgt, srcpos = build_transitions(h, tok, pid, WINDOW)
    ipr_w = load_ipr_weights(args.corpus)              # per-position IPR weight
    tw = ipr_w[srcpos]                                 # weight per transition (source position)
    print(f"[data] IPR weights: mean {tw.mean():.3f} median {np.median(tw):.3f} "
          f"(<0.2: {100*np.mean(tw<0.2):.0f}% near-flat, >0.8: {100*np.mean(tw>0.8):.0f}% near-dirac)")
    uniq = np.unique(act); remap = {int(t): i for i, t in enumerate(uniq)}
    act_ix = np.array([remap[int(t)] for t in act], np.int32); n_actions = len(uniq)
    d = DMODEL if args.option == "a" else args.d
    phi, pred, psi, aemb = make_models(mx, nn, d, args.dp, args.enc, DEC_LAYERS, n_actions, args.option)
    mask = causal_mask(mx, WINDOW)
    sigreg = make_sigreg(mx, args.sigreg)
    # lm_head for the HONEST metric: does the predicted latent decode to the right token?
    # decode path needs psi too, but in Phase 1 psi is untrained -> we measure token-agreement
    # of the TARGET-encoded latent's decode only as a sanity ref; real token metric is Phase 2.
    # Here we instead report CENTERED cosine (honest per-step prediction, immune to shared mean).
    class WM(nn.Module):
        def __init__(s): super().__init__(); s.phi = phi; s.pred = pred; s.aemb = aemb
    wm = WM()
    print(f"[phase1] corpus={args.corpus} d={d} dp={args.dp} enc={args.enc} pred={PRED_LAYERS} "
          f"actions={n_actions} sigreg={args.sigreg}")
    print(f"[phase1] world-model: {sram_line(n_params(mx, wm))}")
    print(f"[data] {wi.shape[0]} transitions over {h.shape[0]} positions, W={WINDOW}")
    def loss_fn(hw, a, ht, w):
        z = wm.phi(hw) if args.option == "b" else hw
        zt = wm.phi(ht[:, None, :])[:, 0, :] if args.option == "b" else ht
        if args.detach_target: zt = mx.stop_gradient(zt)
        zhat = wm.pred(z, wm.aemb(a), mask)
        se = mx.sum((zhat - zt) ** 2, axis=-1)                  # [B] per-sample sq error
        Lpred = mx.sum(w * se) / (mx.sum(w) + 1e-6)             # IPR-weighted, scale-stable
        Lsig = sigreg(z) if args.option == "b" else mx.array(0.0)
        return Lpred + SIGREG_LAMBDA * Lsig
    lg = nn.value_and_grad(wm, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)
    T = wi.shape[0]; bs = args.bs
    for ep in range(args.epochs):
        t0 = time.time(); idx = rng.permutation(T); tot = nb = 0
        for i in range(0, T - bs + 1, bs):
            b = idx[i:i + bs]
            hw = mx.array(h[wi[b]]); a = mx.array(act_ix[b]); ht = mx.array(h[tgt[b]]); w = mx.array(tw[b])
            L, g = lg(hw, a, ht, w); opt.update(wm, g); mx.eval(wm.parameters(), opt.state)
            tot += float(L); nb += 1
        b = rng.integers(0, T, size=min(4096, T))
        hw = mx.array(h[wi[b]]); a = mx.array(act_ix[b]); ht = mx.array(h[tgt[b]])
        z = wm.phi(hw) if args.option == "b" else hw
        zt = wm.phi(ht[:, None, :])[:, 0, :] if args.option == "b" else ht
        zhat = wm.pred(z, wm.aemb(a), mask)
        mse = float(mx.mean(mx.sum((zhat - zt) ** 2, axis=-1)))
        cos = float(mx.mean(mx.sum(zhat * zt, axis=-1) /
                            (mx.linalg.norm(zhat, axis=-1) * mx.linalg.norm(zt, axis=-1) + 1e-6)))
        # CENTERED cosine: remove batch mean first -> immune to shared-direction cheat.
        zh_c = zhat - mx.mean(zhat, axis=0); zt_c = zt - mx.mean(zt, axis=0)
        ccos = float(mx.mean(mx.sum(zh_c * zt_c, axis=-1) /
                             (mx.linalg.norm(zh_c, axis=-1) * mx.linalg.norm(zt_c, axis=-1) + 1e-6)))
        # baseline: how well does "predict the batch-mean latent" score on RAW cos? (the cheat floor)
        zmean = mx.mean(zt, axis=0, keepdims=True)
        cos_mean = float(mx.mean(mx.sum(zmean * zt, axis=-1) /
                                 (mx.linalg.norm(zmean, axis=-1) * mx.linalg.norm(zt, axis=-1) + 1e-6)))
        zstd = float(mx.mean(mx.std(z.reshape(-1, z.shape[-1]), axis=0)))
        # latent-rho: does the LEARNED latent reveal dynamics the raw hidden hides?
        # consecutive within-window latents z[:, -2] vs z[:, -1] (last two positions)
        if args.option == "b" and z.shape[1] >= 2:
            za = z[:, -2, :]; zb = z[:, -1, :]
            zac = za - mx.mean(za, axis=0); zbc = zb - mx.mean(zb, axis=0)
            rho = mx.mean(mx.sum(zac*zbc, axis=0) /
                          (mx.sqrt(mx.sum(zac*zac, axis=0)*mx.sum(zbc*zbc, axis=0)) + 1e-6))
            rho_s = f" | latent_rho {float(rho):+.3f} (>>0 = world found)"
        else:
            rho_s = ""
        print(f"[ep {ep+1}/{args.epochs}] loss {tot/nb:.4f} | predMSE {mse:.4f} | "
              f"cos {cos:.3f} (mean-cheat floor {cos_mean:.3f}) | CENTERED-cos {ccos:+.3f} (HONEST) "
              f"| latent_std {zstd:.3f}{rho_s} | {time.time()-t0:.1f}s")
    os.makedirs(CKPTS, exist_ok=True)
    ck = os.path.join(CKPTS, f"p1_{args.corpus}_d{d}_dp{args.dp}_{args.option}_{args.sigreg}.npz")
    save_ckpt(mx, ck, wm, {"d": d, "dp": args.dp, "enc": args.enc, "option": args.option,
                           "sigreg": args.sigreg, "n_actions": n_actions, "remap": remap})
    print(f"[saved] {ck}")

def phase2(args, mx, nn, optim):
    mx.random.seed(SEED); rng = np.random.default_rng(SEED)
    meta = json.load(open(args.ckpt + ".meta.json"))
    d, dp, enc, option = meta["d"], meta["dp"], meta["enc"], meta["option"]
    remap = {int(k): v for k, v in meta["remap"].items()}
    h, tok, pid = load_trace(args.corpus)
    wi, act, tgt, srcpos = build_transitions(h, tok, pid, WINDOW)
    act_ix = np.array([remap.get(int(t), 0) for t in act], np.int32)
    phi, pred, psi, aemb = make_models(mx, nn, d, dp, enc, DEC_LAYERS, meta["n_actions"], option)
    class WM(nn.Module):
        def __init__(s): super().__init__(); s.phi = phi; s.pred = pred; s.aemb = aemb
    wm = WM(); load_ckpt(mx, wm, args.ckpt)
    mask = causal_mask(mx, WINDOW)
    lm = load_lm_head(mx)
    class DEC(nn.Module):
        def __init__(s): super().__init__(); s.psi = psi
    dwm = DEC()
    print(f"[phase2] decoder psi: {sram_line(n_params(mx, dwm))} (load-bearing)")
    def dec_loss(hb):
        # hb: [B,4096]. phi expects [B,T,W]; encode as length-1 sequence then squeeze.
        z = wm.phi(hb[:, None, :])[:, 0, :] if option == "b" else hb
        hrec = dwm.psi(z) if option == "b" else hb
        logits = lm(hrec)
        tgt_tok = mx.argmax(lm(hb), axis=-1)
        return mx.mean(nn.losses.cross_entropy(logits, tgt_tok))
    lg = nn.value_and_grad(dwm, dec_loss)
    opt = optim.AdamW(learning_rate=args.lr)
    pos = np.unique(tgt); T = pos.shape[0]; bs = args.bs
    for ep in range(args.epochs):
        t0 = time.time(); idx = rng.permutation(T); tot = nb = 0
        for i in range(0, T - bs + 1, bs):
            hb = mx.array(h[pos[idx[i:i+bs]]])
            L, g = lg(hb); opt.update(dwm, g); mx.eval(dwm.parameters(), opt.state)
            tot += float(L); nb += 1
        print(f"[ep {ep+1}/{args.epochs}] dec CE {tot/nb:.4f} | {time.time()-t0:.1f}s")
    b = rng.integers(0, wi.shape[0], size=min(4096, wi.shape[0]))
    hw = mx.array(h[wi[b]]); a = mx.array(act_ix[b]); ht = mx.array(h[tgt[b]])
    trunk_tok = np.array(mx.argmax(lm(ht), axis=-1))
    z_real = wm.phi(ht[:, None, :])[:, 0, :] if option == "b" else ht
    h_rec = psi(z_real) if option == "b" else ht
    rec_tok = np.array(mx.argmax(lm(h_rec), axis=-1))
    ceiling = float((rec_tok == trunk_tok).mean())
    z = wm.phi(hw) if option == "b" else hw
    zt = wm.phi(ht[:, None, :])[:, 0, :] if option == "b" else ht
    zhat = wm.pred(z, wm.aemb(a), mask)
    dyn_mse = float(mx.mean(mx.sum((zhat - zt) ** 2, axis=-1)))
    dyn_cos = float(mx.mean(mx.sum(zhat * zt, axis=-1) /
                            (mx.linalg.norm(zhat, axis=-1) * mx.linalg.norm(zt, axis=-1) + 1e-6)))
    h_pred = psi(zhat) if option == "b" else zhat
    e2e_tok = np.array(mx.argmax(lm(h_pred), axis=-1))
    accept = float((e2e_tok == trunk_tok).mean())
    print("\n==================== DIAGNOSTICS ====================")
    print(f"(1) recon CEILING (latent retains token?) : {ceiling*100:5.1f}%")
    print(f"(2) dynamics  predMSE {dyn_mse:.4f}  cos {dyn_cos:.3f}")
    print(f"(3) end-to-end ACCEPT (depth-1)           : {accept*100:5.1f}%")
    print("-----------------------------------------------------")
    if accept < ceiling - 0.05:
        print("  read: accept << ceiling => PREDICTOR bottleneck. deepen/widen predictor.")
    elif ceiling < 0.9:
        print(f"  read: ceiling low => LATENT too lossy at d={d}. raise d or add recon term.")
    else:
        print("  read: ceiling high AND accept~ceiling => Option B VALIDATED at this (d,dp).")
    print("=====================================================")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dp", type=int, default=384)
    ap.add_argument("--enc", type=int, default=ENC_LAYERS)
    ap.add_argument("--option", choices=["a", "b"], default="b")
    ap.add_argument("--sigreg", choices=["full", "shape"], default="full")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--detach-target", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    if args.sweep:
        for d in (64, 128, 256, 512):
            for dp in (256, 384, 512):
                print(f"\n########## sweep d={d} dp={dp} sigreg={args.sigreg} ##########")
                a2 = argparse.Namespace(**vars(args)); a2.d = d; a2.dp = dp; a2.sweep = False
                phase1(a2, mx, nn, optim)
        return
    if args.phase == 1:
        phase1(args, mx, nn, optim)
    else:
        assert args.ckpt, "--phase 2 needs --ckpt <phase1.npz>"
        phase2(args, mx, nn, optim)

if __name__ == "__main__":
    main()
