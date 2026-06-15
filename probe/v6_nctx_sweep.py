#!/usr/bin/env python3
"""Sweep V6 across n_ctx = [4, 8, 16, 32, 64] on MiMo bf16.
Measures the diminishing-returns curve and finds the knee.
Run with --extended to also test n_ctx=[128, 256] (adds ~12 hours)."""
from __future__ import annotations
import sys, os, json, time, argparse
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("pip install safetensors"); sys.exit(1)

MODEL_DIR = "/Volumes/TB5/llm/MiMo-V2-Flash"
BLOCK_SIZE = 16
E4M3_MAX = 448.0
N_EVAL_TENSORS = 15

# ============ E4M3 quantization helper ============
def _e4m3():
    n = 1 << 7; bias = 7; m = np.zeros(n)
    for c in range(n):
        e, mm = c >> 3, c & 7
        m[c] = ((mm/8)*(2.0**(1-bias))) if e == 0 else ((1+mm/8)*(2.0**(e-bias)))
    m[-1] = np.nan
    return np.sort(m[~np.isnan(m)])
E4M3 = _e4m3()

def q_e4m3(x):
    av = np.abs(x).astype(np.float32); sg = np.sign(x).astype(np.float32)
    mids = ((E4M3[:-1] + E4M3[1:]) * 0.5).astype(np.float32)
    return sg * E4M3.astype(np.float32)[np.searchsorted(mids, av, side="right")]

# ============ Tensor loading ============
def load_bf16(path, name):
    with open(path, "rb") as fp:
        hs = int.from_bytes(fp.read(8), "little")
        hdr = json.loads(fp.read(hs).decode())
        e = hdr[name]; off = e["data_offsets"]; shape = e["shape"]
        fp.seek(8 + hs + off[0]); raw = fp.read(off[1] - off[0])
    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)

def is_bf16(path, name):
    try:
        with safe_open(path, framework="numpy") as f:
            return "BF16" in str(f.get_slice(name).get_dtype()).upper()
    except: return False

def list_bf16_2d(d):
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    out = []
    for name, fname in idx["weight_map"].items():
        if not name.endswith(".weight") or "norm" in name.lower(): continue
        p = os.path.join(d, fname)
        if not is_bf16(p, name): continue
        with safe_open(p, framework="numpy") as f:
            shape = tuple(f.get_slice(name).get_shape())
        if len(shape) != 2 or shape[0] < 64 or shape[1] < 64: continue
        out.append((name, p, shape))
    return out

def col_importance(W):
    cn = np.linalg.norm(W, axis=0)
    return (cn / (cn.mean() + 1e-8)).astype(np.float32)

def lloyd_max(samples, n_levels, n_iter=50):
    qs = np.linspace(0.5/n_levels, 1 - 0.5/n_levels, n_levels)
    g = np.quantile(samples, qs)
    for _ in range(n_iter):
        g = np.sort(g); mids = (g[:-1] + g[1:]) / 2.0
        bins = np.searchsorted(mids, samples); new = np.zeros(n_levels)
        for i in range(n_levels):
            m = bins == i; new[i] = samples[m].mean() if m.any() else g[i]
        g = new
    return np.sort(g)

def nearest(values, grid):
    mids = ((grid[:-1] + grid[1:]) * 0.5).astype(np.float32)
    idx = np.clip(np.searchsorted(mids, values.astype(np.float32), side="right"),
                  0, len(grid) - 1)
    return grid.astype(np.float32)[idx]

# ============ V6 codec ============
def variant6(W, n_ctx=4, n_codes=16, n_iter=20, verbose=False):
    rows, cols = W.shape
    bpr = cols // BLOCK_SIZE
    Wf = W[:, :bpr*BLOCK_SIZE].reshape(-1, BLOCK_SIZE).astype(np.float32)
    bm = np.max(np.abs(Wf), axis=1); ma = float(np.max(np.abs(Wf))) or 1.0
    gs = ma / (6.0 * E4M3_MAX); bs_q = q_e4m3((bm / 6.0) / gs)
    eff = bs_q * gs; safe = np.where(eff > 0, eff, 1.0).astype(np.float32)
    Wn = Wf / safe[:, None]
    flat = np.abs(Wn.reshape(-1))
    if len(flat) > 200_000:
        rng_s = np.random.default_rng(42)
        flat = flat[rng_s.choice(len(flat), size=200_000, replace=False)]
    base = lloyd_max(flat, n_levels=max(2, n_codes // 2))
    rng = np.random.default_rng(0)
    codebook = np.zeros((n_ctx, n_codes))
    for c in range(n_ctx):
        sf = 1.0 + 0.3 * (c - n_ctx / 2) / max(1, n_ctx)
        mg = np.sort(np.abs(base) * sf)
        signed = np.unique(np.sort(np.concatenate([-mg[mg > 0], [0.0], mg[mg > 0]])))
        if len(signed) > n_codes: signed = signed[:n_codes]
        elif len(signed) < n_codes:
            extra = rng.uniform(-0.5, 0.5, n_codes - len(signed))
            signed = np.sort(np.concatenate([signed, extra]))
        codebook[c] = signed
    contexts = np.zeros(Wf.shape[0], dtype=np.int32)
    for it in range(n_iter):
        err_per_ctx = np.zeros((Wf.shape[0], n_ctx))
        for c in range(n_ctx):
            g = np.sort(codebook[c])
            Wq = nearest(Wn.reshape(-1), g).reshape(Wn.shape)
            err_per_ctx[:, c] = np.sum((Wq - Wn) ** 2, axis=1)
        new_ctx = np.argmin(err_per_ctx, axis=1)
        new_cb = codebook.copy()
        for c in range(n_ctx):
            mask = new_ctx == c
            if not mask.any(): continue
            blocks_c = Wn[mask].reshape(-1)
            g_c = np.sort(codebook[c])
            mids = (g_c[:-1] + g_c[1:]) / 2.0
            bins = np.searchsorted(mids, blocks_c)
            for k in range(n_codes):
                bm2 = bins == k
                if bm2.any(): new_cb[c, k] = blocks_c[bm2].mean()
        for c in range(n_ctx): new_cb[c] = np.sort(new_cb[c])
        converged = np.array_equal(new_ctx, contexts) and it > 0
        codebook = new_cb; contexts = new_ctx
        if verbose and (it < 3 or it % 5 == 0):
            err = np.mean(err_per_ctx[np.arange(len(contexts)), contexts])
            print(f"      iter {it}: err={err:.5f}", flush=True)
        if converged: break
    W_hat_norm = np.zeros_like(Wn)
    for c in range(n_ctx):
        mask = contexts == c
        if mask.any():
            g = np.sort(codebook[c])
            W_hat_norm[mask] = nearest(Wn[mask].reshape(-1), g).reshape((mask.sum(), BLOCK_SIZE))
    return (W_hat_norm * eff[:, None]).reshape(rows, bpr*BLOCK_SIZE).astype(np.float32)

# ============ Metrics ============
def w_re(W, Wh):
    return float(np.linalg.norm(W - Wh) / (np.linalg.norm(W) + 1e-30))

def op_re(W, Wh, ci, n_samples=64, seed=123):
    rng = np.random.default_rng(seed)
    cols = W.shape[-1]
    x = rng.standard_normal((cols, n_samples)).astype(np.float32) * np.sqrt(ci[:, None])
    y_true = W @ x; y_hat = Wh @ x
    return float(np.linalg.norm(y_true - y_hat) / (np.linalg.norm(y_true) + 1e-30))

# ============ Main sweep ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true",
                    help="Also test n_ctx=128 and 256 (adds ~12 hours)")
    ap.add_argument("--n-tensors", type=int, default=N_EVAL_TENSORS)
    ap.add_argument("--max-tensor-size", type=int, default=8_000_000,
                    help="Crop tensors larger than this for speed")
    args = ap.parse_args()
    
    nctx_values = [4, 8, 16, 32, 64]
    if args.extended:
        nctx_values += [128, 256]
    
    print("=== V6 n_ctx sweep on MiMo bf16 ===")
    print(f"n_ctx values: {nctx_values}")
    print(f"Tensors: {args.n_tensors}\n")
    
    specs = list_bf16_2d(MODEL_DIR)
    print(f"Found {len(specs)} bf16 2D tensors")
    
    rng = np.random.default_rng(42)
    sel = [specs[i] for i in rng.choice(len(specs), size=min(args.n_tensors, len(specs)), replace=False)]
    
    results = {n: {"w": [], "op": [], "time": []} for n in nctx_values}
    
    # Header
    hdr = f"  {'tensor':<55}"
    for n in nctx_values: hdr += f"  n={n:<5}"
    print(hdr)
    print("  " + "-" * len(hdr))
    
    t_global = time.time()
    nvfp4_baseline_w = []
    nvfp4_baseline_op = []
    
    for tensor_idx, (name, path, shape) in enumerate(sel):
        try:
            W = load_bf16(path, name)
            if W.size > args.max_tensor_size:
                # Crop to keep runtime reasonable
                W = W[:W.shape[0] // 2, :W.shape[1]]
            ci = col_importance(W)
        except Exception as e:
            print(f"  skip {name}: {e}"); continue
        
        # NVFP4 baseline for this tensor (for gain computation)
        # Use V3 inline as proxy baseline (or compute NVFP4)
        # For sweep purposes we just track absolute rel_err
        
        sn = name if len(name) <= 55 else (name[:25] + "..." + name[-27:])
        print(f"  {sn:<55}", end="", flush=True)
        
        for n_ctx in nctx_values:
            t0 = time.time()
            try:
                Wh = variant6(W, n_ctx=n_ctx, n_iter=20)
                elapsed = time.time() - t0
                w_e = w_re(W[:, :Wh.shape[1]], Wh)
                o_e = op_re(W[:, :Wh.shape[1]], Wh, ci[:Wh.shape[1]])
                results[n_ctx]["w"].append(w_e)
                results[n_ctx]["op"].append(o_e)
                results[n_ctx]["time"].append(elapsed)
                print(f"  {w_e:.4f}", end="", flush=True)
            except Exception as e:
                print(f"  FAIL  ", end="", flush=True)
        print()
        
        # Progress estimate
        if tensor_idx < len(sel) - 1:
            done = tensor_idx + 1
            remaining = len(sel) - done
            avg_per_tensor = (time.time() - t_global) / done
            eta_s = avg_per_tensor * remaining
            if eta_s > 60:
                print(f"  [progress {done}/{len(sel)}, ETA {eta_s/60:.1f} min]", flush=True)
    
    print(f"\n  Total runtime: {(time.time()-t_global)/60:.1f} min\n")
    
    print(f"  {'n_ctx':<8} {'b/elem':<10} {'mean w_rel':<12} {'mean op_rel':<12} "
          f"{'calib min':<10}")
    print("  " + "-" * 60)
    
    NVFP4_BASELINE_REL_ERR = 0.0944
    for n in nctx_values:
        if not results[n]["w"]:
            print(f"  {n:<8} no data")
            continue
        b_per_elem = 4.5 + np.log2(n) / 16.0
        mw = np.mean(results[n]["w"])
        mo = np.mean(results[n]["op"])
        gain = (NVFP4_BASELINE_REL_ERR - mw) / NVFP4_BASELINE_REL_ERR * 100
        mt = np.mean(results[n]["time"]) / 60
        print(f"  {n:<8} {b_per_elem:<10.4f} {mw:<12.5f} {mo:<12.5f} "
              f"{mt:<10.2f}  ({gain:+.1f}% over NVFP4)")
    
    print("\n  Diminishing returns curve:")
    prev_w = None
    for n in nctx_values:
        if not results[n]["w"]: continue
        mw = np.mean(results[n]["w"])
        if prev_w is not None:
            marg = (prev_w - mw) / prev_w * 100
            print(f"    n_ctx={n}: {mw:.5f}  (marginal gain {marg:+.2f}% over n_ctx={prev_n})")
        else:
            print(f"    n_ctx={n}: {mw:.5f}  (start)")
        prev_w = mw
        prev_n = n

if __name__ == "__main__":
    main()
