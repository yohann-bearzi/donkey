"""Measure per-K accept rate on a held-out trace using a trained donkey checkpoint.

usage: measure_accept_rate.py <checkpoint.safetensors> <trace_dir> [--batch N]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from donkey_config import DonkeyConfig
from donkey_world import DonkeyWorldRef
from trace_dataset import DonkeyTraceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--spec", type=Path,
                    default=Path(__file__).parents[2] / "spec/donkey_v2_default.json")
    ap.add_argument("--lm-head", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_lm_head_fp16.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="cap how many windows to evaluate; 0 = all")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to write per-example parquet")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = DonkeyConfig.from_json(args.spec)

    print(f"[eval] loading donkey from {args.checkpoint}")
    model = DonkeyWorldRef(cfg).to(device).eval()
    sd = load_file(str(args.checkpoint))
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] donkey {n_params/1e6:.2f}M params on {device}")

    print(f"[eval] lm_head <- {args.lm_head}")
    lm_data = torch.load(args.lm_head, map_location="cpu", weights_only=False)
    lm_head = lm_data["weight"].to(device).to(torch.float32)
    lm_head.requires_grad_(False)

    print(f"[eval] dataset <- {args.trace_dir}")
    ds = DonkeyTraceDataset(args.trace_dir, cfg.sequence.window_size,
                            cfg.sequence.draft_size, trunk_hidden=cfg.trunk.hidden_dim)
    N = len(ds)
    if args.limit_windows > 0:
        N = min(N, args.limit_windows)
    print(f"[eval] {N} windows to evaluate (of {len(ds)} available)")

    correct = np.zeros((N, cfg.sequence.draft_size), dtype=np.int8)
    cos = np.zeros((N, cfg.sequence.draft_size), dtype=np.float32)
    records = [] if args.out else None
    K = cfg.sequence.draft_size

    with torch.no_grad():
        i = 0
        while i < N:
            j = min(i + args.batch, N)
            histories = torch.stack([ds[k]["history"]       for k in range(i, j)]).to(device)
            targets   = torch.stack([ds[k]["targets"]       for k in range(i, j)]).to(device)
            tokens    = torch.stack([ds[k]["target_tokens"] for k in range(i, j)]).to(device)

            out = model.forward_for_training(histories)
            z_pred = out["z_pred"]            # [B, D, K]
            pred_decoded = out["pred_decoded"] # [B, H, K]
            z_target = model.input_proj(targets)

            # cos in latent
            zp = z_pred.permute(0, 2, 1)
            zt = z_target.permute(0, 2, 1)
            c = torch.nn.functional.cosine_similarity(zp, zt, dim=-1)  # [B, K]

            # argmax through lm_head
            pf = pred_decoded.permute(0, 2, 1).float()
            logits = pf @ lm_head.float().t()
            argmax = logits.argmax(dim=-1)
            corr = (argmax == tokens).int()  # [B, K]

            correct[i:j] = corr.cpu().numpy().astype(np.int8)
            cos[i:j]     = c.cpu().numpy().astype(np.float32)

            if records is not None:
                for bi in range(j - i):
                    for k in range(K):
                        records.append({
                            "window_idx": i + bi,
                            "k_slot": k,
                            "target_token": int(tokens[bi, k].item()),
                            "argmax_token": int(argmax[bi, k].item()),
                            "argmax_correct": bool(corr[bi, k].item()),
                            "cos_pred_target": float(c[bi, k].item()),
                        })
            i = j
            if i % (args.batch * 4) == 0 or i == N:
                print(f"[eval]   {i}/{N} ({100.0*i/N:.0f}%)  running K0={100*correct[:i,0].mean():.1f}% "
                      f"K1={100*correct[:i,1].mean():.1f}% K2={100*correct[:i,2].mean():.1f}%")

    print()
    print(f"=== HELD-OUT ACCEPT RATE (N={N} windows) ===")
    overall = 100.0 * correct.mean()
    print(f"  aggregate (all K, all windows): {overall:.2f}%")
    for k in range(K):
        acc = 100.0 * correct[:, k].mean()
        mc  = float(cos[:, k].mean())
        print(f"  K={k}  accept={acc:6.2f}%   mean_cos={mc:.3f}")
    print(f"  any-K accept (at least one correct per window): "
          f"{100.0 * (correct.any(axis=1)).mean():.2f}%")
    print(f"  all-K accept (all K correct simultaneously):    "
          f"{100.0 * (correct.all(axis=1)).mean():.2f}%")
    print()
    print(f"  cos range: {cos.min():.3f} .. {cos.max():.3f}  mean: {cos.mean():.3f}")

    if records is not None:
        try:
            import pandas as pd
            pd.DataFrame(records).to_parquet(args.out)
            print(f"[eval] per-example dump -> {args.out}")
        except ImportError:
            print("[eval] pandas missing, skipping parquet dump")


if __name__ == "__main__":
    main()
