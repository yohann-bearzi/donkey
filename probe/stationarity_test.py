#!/usr/bin/env python3
"""Test whether harvested trunk hiddens satisfy LeJEPA's stationarity / OU assumptions.
Reads bins safely while harvest is still running (only up to last flushed prompt).
No model load. Usage: python3 stationarity_test.py [humaneval|mbpp|codealpaca_20k]"""
import sys, os, numpy as np

BASE = os.environ.get("DONKEY_DATASET", "/Volumes/TB5/donkey/dataset")
DMODEL = 4096
STOP = (151645, 151643)
corpus = sys.argv[1] if len(sys.argv) > 1 else "humaneval"
T = os.path.join(BASE, "traces", corpus)

# --- load safely: truncate all streams to the min common length (harvest may be mid-write) ---
hbytes = os.path.getsize(os.path.join(T, "lastHiddenState.bin"))
n_hid = hbytes // (4 * DMODEL)
tok = np.fromfile(os.path.join(T, "tokens.bin"), np.int32)
pid = np.fromfile(os.path.join(T, "prompt_idx.bin"), np.int32)
n = min(n_hid, len(tok), len(pid))
tok, pid = tok[:n], pid[:n]
# memory-map hiddens (4096-wide, could be many GB) and read the first n rows
H = np.memmap(os.path.join(T, "lastHiddenState.bin"), dtype=np.float32, mode="r",
              shape=(n_hid, DMODEL))[:n]
# drop the last prompt (may be partially written) and any non-terminated cap-hit prompts
last_pid = pid[-1]
keep_mask = pid < last_pid
print(f"[{corpus}] {n} positions, {len(np.unique(pid))} prompts (dropping last as possibly-mid-write)")

# helper: per-dim stats on a row subset, using float64 accumulation
def stats(rows):
    x = np.asarray(rows, np.float64)
    return x.mean(0), x.std(0)

# ============ TEST 1: global per-dim mean/std + overall scale ============
gm, gs = stats(H[keep_mask])
print(f"\n[1] global latent geometry: |mean| {np.abs(gm).mean():.3f}  avg std {gs.mean():.3f}  "
      f"(anisotropy: std range {gs.min():.2f}-{gs.max():.2f})")

# ============ TEST 2: drift across ABSOLUTE position blocks (whole-stream stationarity) ============
print("\n[2] drift across stream blocks (absolute time):")
idx = np.where(keep_mask)[0]
blocks = np.array_split(idx, 8)
bms, bss = [], []
for i, b in enumerate(blocks):
    m, s = stats(H[b]); bms.append(m.mean()); bss.append(s.mean())
    print(f"    block {i}: mean {m.mean():+.3f}  std {s.mean():.3f}  n={len(b)}")
print(f"    => mean drift {max(bms)-min(bms):.3f}, std drift {max(bss)-min(bss):.3f}  (large=non-stationary)")

# ============ TEST 3: WITHIN-PROMPT fractional position (think vs answer) — THE KEY ONE ============
print("\n[3] within-prompt fractional position (early=think -> late=answer):")
nbins = 5
bins = [[] for _ in range(nbins)]
for p in np.unique(pid[keep_mask]):
    pidx = np.where(pid == p)[0]
    L = len(pidx)
    if L < nbins: continue
    for r, i in enumerate(pidx):
        bins[min(nbins-1, int(nbins*r/L))].append(i)
fb_means, fb_stds = [], []
for i, b in enumerate(bins):
    if not b: continue
    m, s = stats(H[np.array(b)]); fb_means.append(m.mean()); fb_stds.append(s.mean())
    print(f"    bin {i} ({i*100//nbins}-{(i+1)*100//nbins}%): mean {m.mean():+.3f}  std {s.mean():.3f}  n={len(b)}")
print(f"    => mean swing early->late {max(fb_means)-min(fb_means):.3f}  "
      f"(large => think != answer => NON-STATIONARY within prompt)")

# ============ TEST 4: lag-1 autocorrelation (OU predicts single clean rho) ============
print("\n[4] lag-1 autocorrelation (OU: single rho, Cov(z,z')=rho*I):")
# consecutive WITHIN-prompt pairs only
pairs0, pairs1 = [], []
for p in np.unique(pid[keep_mask]):
    pidx = np.where(pid == p)[0]
    if len(pidx) < 2: continue
    pairs0.append(pidx[:-1]); pairs1.append(pidx[1:])
i0 = np.concatenate(pairs0); i1 = np.concatenate(pairs1)
samp = np.random.default_rng(0).choice(len(i0), size=min(50000, len(i0)), replace=False)
z0 = np.asarray(H[i0[samp]], np.float64); z1 = np.asarray(H[i1[samp]], np.float64)
z0c = z0 - z0.mean(0); z1c = z1 - z1.mean(0)
rho_perdim = (z0c*z1c).sum(0) / (np.sqrt((z0c**2).sum(0)*(z1c**2).sum(0)) + 1e-9)
print(f"    lag-1 rho: mean {rho_perdim.mean():.3f}  spread {rho_perdim.std():.3f} "
      f"(range {rho_perdim.min():.2f}-{rho_perdim.max():.2f})")
print(f"    OU wants a single rho (low spread). high spread => not a single-rho OU process.")

print("\n================= READING =================")
print("Test 3 is the one that matters most for the donkey:")
print("  small swing  => trunk dynamics ~stationary within prompt => LeJEPA identifiability ~holds")
print("  large swing  => think/answer phases differ => approximate-identifiability regime;")
print("                  consider separate dynamics per phase, or phase as an action input.")
