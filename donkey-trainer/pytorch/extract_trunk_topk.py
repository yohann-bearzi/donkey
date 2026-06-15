"""Extract top-K trunk distribution at fixed K=256, with streaming write.

Per position: top-256 token IDs and probabilities renormalized to sum to 1.0
within the kept entries. Fixed shape, dense storage.

Output files per trace dir (overwrites any previous extraction):
    trunk_topk_ids.bin    int32   [T, 256]   row-major
    trunk_topk_probs.bin  float32 [T, 256]   row-major, each row sums to 1.0

meta.json updated:
    topk_format: "fixed_K256"
    topk_K: 256
    topk_mean_kept_mass: fraction of original mass captured by top-256

Usage:
    extract_trunk_topk.py <trace_dir> [--batch 256]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


K = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lm-head", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_lm_head_fp16.pt")
    ap.add_argument("--trunk-shard", type=Path,
                    default=Path("/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M/model-00144-of-00144.safetensors"))
    args = ap.parse_args()

    meta_path = args.trace_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    T, H = meta["total_positions"], meta["hidden_dim"]
    print(f"[extract] {args.trace_dir.name}: {T} positions, K={K}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    hidden = np.memmap(args.trace_dir / "lastHiddenState.bin", dtype=np.float32,
                       mode="r", shape=(T, H))
    with safe_open(str(args.trunk_shard), framework="pt") as f:
        gamma = f.get_tensor("model.norm.weight").to(torch.float32).to(device)
    lm = torch.load(args.lm_head, map_location="cpu", weights_only=False)["weight"].to(device).float()
    lm_t = lm.t().contiguous()
    eps = 1e-5
    V = lm.shape[0]

    # Pre-allocate output memmaps (fixed shape — streaming write per batch)
    ids_path = args.trace_dir / "trunk_topk_ids.bin"
    probs_path = args.trace_dir / "trunk_topk_probs.bin"
    # Remove any old files first (might be ragged format with different size)
    for p in [ids_path, probs_path, args.trace_dir / "trunk_topk_offsets.bin"]:
        if p.exists():
            p.unlink()
            print(f"[extract] removed old {p.name}")

    ids_mm = np.memmap(ids_path, dtype=np.int32, mode="w+", shape=(T, K))
    probs_mm = np.memmap(probs_path, dtype=np.float32, mode="w+", shape=(T, K))

    t0 = time.monotonic()
    kept_mass_sum = 0.0  # for diagnostic: avg fraction of mass captured by top-K

    for i in range(0, T, args.batch):
        j = min(i + args.batch, T)
        h_b = torch.from_numpy(np.ascontiguousarray(hidden[i:j])).to(device).float()
        with torch.no_grad():
            rms = h_b.pow(2).mean(dim=-1, keepdim=True).sqrt()
            h_normed = h_b * gamma / (rms + eps)
            logits = h_normed @ lm_t                            # [b, V]
            probs = torch.softmax(logits, dim=-1)              # [b, V]
            top_p, top_i = probs.topk(K, dim=-1)               # [b, K]
            # Diagnostic: sum of top-K mass before renormalization
            kept = top_p.sum(dim=-1)                            # [b]
            kept_mass_sum += kept.sum().item()
            # Renormalize per row so each row sums to 1.0
            top_p_renorm = top_p / kept.unsqueeze(-1).clamp(min=1e-12)

        # Streaming write directly to memmap
        ids_mm[i:j] = top_i.cpu().numpy().astype(np.int32)
        probs_mm[i:j] = top_p_renorm.cpu().numpy().astype(np.float32)

        if (i // args.batch) % 100 == 0 or j == T:
            elapsed = time.monotonic() - t0
            rate = j / elapsed if elapsed > 0 else 0
            eta = (T - j) / rate if rate > 0 else 0
            avg_mass = kept_mass_sum / j
            print(f"[extract]   {j}/{T} ({100*j/T:.1f}%)  "
                  f"{rate:.0f} pos/s  ETA {eta:.0f}s  "
                  f"avg top-{K} mass so far: {avg_mass:.3f}")

    # Flush memmaps
    ids_mm.flush()
    probs_mm.flush()
    del ids_mm
    del probs_mm

    # Sanity: top-1 vs tokens
    tokens = np.fromfile(args.trace_dir / "tokens.bin", dtype=np.int32)
    ids_check = np.memmap(ids_path, dtype=np.int32, mode="r", shape=(T, K))
    N_check = min(2000, T)
    top1 = ids_check[:N_check, 0]
    matches = int((top1 == tokens[:N_check]).sum())
    pct = 100.0 * matches / N_check
    print(f"[extract] sanity: top-1 vs tokens.bin = {matches}/{N_check} ({pct:.1f}%)")

    avg_mass = kept_mass_sum / T
    meta["topk_format"] = "fixed_K256"
    meta["topk_K"] = K
    meta["topk_mean_kept_mass"] = avg_mass
    meta["topk_top1_pct_match"] = pct
    # Clear old fields if present
    for legacy in ["topk_extracted", "topk_extraction_pct_top1_match",
                   "topk_total_entries", "topk_K_99_stats", "topk_mass_threshold"]:
        meta.pop(legacy, None)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    total_size = ids_path.stat().st_size + probs_path.stat().st_size
    print(f"[extract] DONE in {time.monotonic()-t0:.1f}s. "
          f"avg top-{K} captures {avg_mass:.3f} of original mass. "
          f"+{total_size / 1e6:.1f} MB on disk.")


if __name__ == "__main__":
    main()
