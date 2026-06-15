"""Precompute 1M low-discrepancy ε samples for stochastic donkey.

Uses scrambled Sobol sequences mapped through inverse standard-normal CDF to
get Gaussian samples that fill ε-space more uniformly than i.i.d. Better
Wasserstein convergence at small N (per QMC theory).

Output: ~/projects/donkey/donkey-trainer/weights/blue_noise_eps64_1M.pt
        torch.float16 tensor [N, dim]
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import qmc, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parents[1] / "weights/blue_noise_eps64_1M.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.monotonic()
    # Sobol needs N = 2^m for unbiased generation
    log2N = int(np.ceil(np.log2(args.N)))
    N_pow2 = 2 ** log2N
    print(f"[bn] generating Sobol sequence: N={N_pow2:,} (target {args.N:,}), dim={args.dim}")
    sobol = qmc.Sobol(d=args.dim, scramble=True, seed=args.seed)
    u = sobol.random_base2(m=log2N)  # [N, d] uniform in [0,1)
    print(f"[bn] Sobol done in {time.monotonic()-t0:.1f}s")

    # Avoid inverse-CDF blowup at boundary
    u = np.clip(u, 1e-7, 1 - 1e-7)
    eps = norm.ppf(u).astype(np.float32)
    eps = eps[:args.N]
    print(f"[bn] inverse-CDF done. shape={eps.shape}  "
          f"mean={eps.mean():.4f}  std={eps.std():.4f}  "
          f"min={eps.min():.3f}  max={eps.max():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t = torch.from_numpy(eps).to(torch.float16)
    torch.save({"eps": t, "N": args.N, "dim": args.dim, "seed": args.seed,
                "source": "sobol_scrambled+inv_cdf"}, args.out)
    sz_mb = args.out.stat().st_size / 1e6
    print(f"[bn] saved {tuple(t.shape)} {t.dtype} -> {args.out}")
    print(f"[bn] size: {sz_mb:.1f} MB  total wall: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()
