"""Probe MiMo's next-token distribution spread using existing trace data.

We already saved hidden states and the dequantized lm_head. The full
softmax distribution at each position is just softmax(lm_head @ h).
No new decode needed.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path,
                    help="Directory with lastHiddenState.bin + meta.json")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--lm-head", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_lm_head_fp16.pt")
    ap.add_argument("--num-positions", type=int, default=2000,
                    help="Random subset of positions to probe (avoid full 140K)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    print(f"[probe] device: {device}")
    print(f"[probe] trace: {args.trace_dir}")
    print(f"[probe] out:   {args.out_dir}")

    meta = json.loads((args.trace_dir / "meta.json").read_text())
    T_total = meta["total_positions"]
    H = meta["hidden_dim"]
    print(f"[probe] trace: {T_total} positions, hidden_dim={H}")

    hidden = np.memmap(args.trace_dir / "lastHiddenState.bin",
                       dtype=np.float32, mode="r", shape=(T_total, H))

    print(f"[probe] loading lm_head from {args.lm_head}")
    lm_data = torch.load(args.lm_head, map_location="cpu", weights_only=False)
    lm_head = lm_data["weight"].to(device).to(torch.float32)
    V = lm_head.shape[0]
    print(f"[probe] lm_head: {lm_head.shape}  vocab={V}")
    lm_head_t = lm_head.t().contiguous()  # [H, V]

    rng = np.random.default_rng(args.seed)
    N = min(args.num_positions, T_total)
    sel = sorted(rng.choice(T_total, size=N, replace=False))
    print(f"[probe] sampling {N} of {T_total} positions (seed={args.seed})")

    MAX_K_PROBE = 1024
    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    MASS_THRESHOLDS = [0.50, 0.75, 0.90, 0.95, 0.99]

    print(f"[probe] computing distributions...")
    t_start = time.monotonic()
    all_top_probs_sorted = np.zeros((N, MAX_K_PROBE), dtype=np.float32)
    batch = 64

    for i in range(0, N, batch):
        j = min(i + batch, N)
        idxs = sel[i:j]
        h_batch = torch.from_numpy(np.ascontiguousarray(hidden[idxs])).to(device).float()
        with torch.no_grad():
            logits = h_batch @ lm_head_t                          # [b, V]
            probs = torch.softmax(logits, dim=-1)                 # [b, V]
            top_vals, _ = torch.topk(probs, MAX_K_PROBE, dim=-1)  # [b, K_MAX] sorted desc
        all_top_probs_sorted[i:j] = top_vals.cpu().numpy()
        if (i + batch) % 256 == 0 or j == N:
            print(f"[probe]   {j}/{N}  ({100*j/N:.0f}%)")
    print(f"[probe] done in {time.monotonic()-t_start:.1f}s")

    cum_mass = np.cumsum(all_top_probs_sorted, axis=1)

    print("\n=== Mass capture by K ===")
    results_per_K = {}
    for K in K_VALUES:
        if K > MAX_K_PROBE: continue
        m = cum_mass[:, K - 1]
        results_per_K[K] = {
            "mean": float(m.mean()), "median": float(np.median(m)),
            "p10": float(np.percentile(m, 10)), "p25": float(np.percentile(m, 25)),
            "p75": float(np.percentile(m, 75)), "p90": float(np.percentile(m, 90)),
            "p99": float(np.percentile(m, 99)),
            "min": float(m.min()), "max": float(m.max()),
        }
        r = results_per_K[K]
        print(f"  K={K:<5} mean={r['mean']:.3f}  median={r['median']:.3f}  "
              f"p10={r['p10']:.3f}  p90={r['p90']:.3f}")

    print("\n=== Required K for mass threshold ===")
    results_per_threshold = {}
    for threshold in MASS_THRESHOLDS:
        ge = cum_mass >= threshold
        K_needed = np.where(ge.any(axis=1), ge.argmax(axis=1) + 1, MAX_K_PROBE)
        r = {
            "median_K": int(np.median(K_needed)),
            "p25_K": int(np.percentile(K_needed, 25)),
            "p75_K": int(np.percentile(K_needed, 75)),
            "p90_K": int(np.percentile(K_needed, 90)),
            "p99_K": int(np.percentile(K_needed, 99)),
            "max_K": int(K_needed.max()),
            "fraction_K_le_8":   float((K_needed <= 8).mean()),
            "fraction_K_le_16":  float((K_needed <= 16).mean()),
            "fraction_K_le_32":  float((K_needed <= 32).mean()),
            "fraction_K_le_64":  float((K_needed <= 64).mean()),
            "fraction_K_le_128": float((K_needed <= 128).mean()),
        }
        results_per_threshold[threshold] = r
        print(f"  thr={threshold:.2f}: median K={r['median_K']}  p90 K={r['p90_K']}  "
              f"p99 K={r['p99_K']}  max K={r['max_K']}")
        print(f"     frac K<=8: {r['fraction_K_le_8']:.1%}  K<=16: {r['fraction_K_le_16']:.1%}  "
              f"K<=64: {r['fraction_K_le_64']:.1%}")

    eps = 1e-12
    full_entropy = -(all_top_probs_sorted * np.log(all_top_probs_sorted.clip(eps))).sum(axis=1)
    entropy_bits = full_entropy / np.log(2)
    print(f"\n=== Distribution entropy (bits) ===")
    print(f"  mean: {float(entropy_bits.mean()):.2f}  median: {float(np.median(entropy_bits)):.2f}")
    print(f"  p10: {float(np.percentile(entropy_bits, 10)):.2f}  "
          f"p90: {float(np.percentile(entropy_bits, 90)):.2f}  "
          f"max: {float(entropy_bits.max()):.2f}")

    summary = {
        "config": {"trace_dir": str(args.trace_dir), "lm_head": str(args.lm_head),
                   "n_positions_sampled": int(N), "T_total": int(T_total),
                   "MAX_K_PROBE": MAX_K_PROBE, "seed": args.seed},
        "per_K": results_per_K, "per_threshold": results_per_threshold,
        "entropy_bits": {
            "mean": float(entropy_bits.mean()), "median": float(np.median(entropy_bits)),
            "p10": float(np.percentile(entropy_bits, 10)),
            "p90": float(np.percentile(entropy_bits, 90)),
            "min": float(entropy_bits.min()), "max": float(entropy_bits.max()),
        },
    }
    (args.out_dir / "probe_results.json").write_text(json.dumps(summary, indent=2))

    md = [f"# Trunk Distribution Probe — {args.trace_dir.name}\n\n",
          f"_{N} positions sampled from {T_total} total._\n\n## Recommendation\n\n"]
    if results_per_threshold[0.99]["p90_K"] <= 32:
        rec_K = max(16, int(results_per_threshold[0.99]["p90_K"]))
        md.append(f"**K_trunk = {rec_K}** — captures 99% mass on 90%+ of positions.\n\n")
    elif results_per_threshold[0.95]["p90_K"] <= 64:
        md.append(f"**K_trunk = {max(32, int(results_per_threshold[0.95]['p90_K']))}** "
                  f"— captures 95% mass on 90%+ of positions.\n\n")
    else:
        md.append("**K_trunk = 64+** — distribution unusually flat. "
                  "Consider energy-aware loss.\n\n")
    md.append("## Mass capture by K\n\n| K | mean | median | p10 | p90 |\n|---:|---:|---:|---:|---:|\n")
    for K in K_VALUES:
        if K not in results_per_K: continue
        r = results_per_K[K]
        md.append(f"| {K} | {r['mean']:.3f} | {r['median']:.3f} | "
                  f"{r['p10']:.3f} | {r['p90']:.3f} |\n")
    md.append("\n## Required K per mass threshold\n\n"
              "| Threshold | median | p75 | p90 | p99 | max | frac K≤16 | frac K≤64 |\n"
              "|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for threshold in MASS_THRESHOLDS:
        r = results_per_threshold[threshold]
        md.append(f"| {threshold:.2f} | {r['median_K']} | {r['p75_K']} | {r['p90_K']} | "
                  f"{r['p99_K']} | {r['max_K']} | {r['fraction_K_le_16']:.1%} | "
                  f"{r['fraction_K_le_64']:.1%} |\n")
    (args.out_dir / "probe_summary.md").write_text("".join(md))
    print(f"\n[probe] wrote {args.out_dir / 'probe_summary.md'}")
    print(f"[probe] wrote {args.out_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
