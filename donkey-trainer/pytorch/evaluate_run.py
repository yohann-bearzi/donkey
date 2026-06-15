"""Evaluate a completed donkey training run on its sealed verify split.

Loads <run_dir>/checkpoints/best.safetensors (or --ckpt label if specified),
reconstructs the verify split using <run_dir>/split.json, runs forward over
all verify windows, reports K-slot accept rate, writes verify_results.json.

usage: evaluate_run.py <run_dir> [--ckpt best|final|step_NNNNNN] [--batch N]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from donkey_config import DonkeyConfig
from donkey_world import DonkeyWorldRef
from trace_dataset import split_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ckpt", default="best",
                    help="checkpoint label: best, final, or step_NNNNNN")
    ap.add_argument("--spec", type=Path,
                    default=Path(__file__).parents[2] / "spec/donkey_v2_default.json")
    ap.add_argument("--lm-head", type=Path,
                    default=Path(__file__).parents[1] / "weights/mimo_lm_head_fp16.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = ap.parse_args()

    # Load run metadata
    run_config = json.loads((args.run_dir / "config.json").read_text())
    split_meta = json.loads((args.run_dir / "split.json").read_text())
    print(f"[eval] run: {args.run_dir.name}")
    print(f"[eval] datasets: {run_config['datasets']}")
    print(f"[eval] split seed: {split_meta['split_seed']}  fractions: {split_meta['fractions']}")
    print(f"[eval] expected verify size: {split_meta['verify_size']}")

    # Resolve checkpoint
    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"{args.ckpt}.safetensors"
    if not ckpt_path.exists():
        print(f"[eval] FAIL: {ckpt_path} not found")
        print(f"[eval] available: {sorted(p.name for p in ckpt_dir.glob('*.safetensors'))}")
        sys.exit(2)
    ckpt_meta_path = ckpt_dir / f"{args.ckpt}.json"
    if ckpt_meta_path.exists():
        print(f"[eval] ckpt meta: {json.loads(ckpt_meta_path.read_text())}")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
    cfg = DonkeyConfig.from_json(args.spec)

    print(f"[eval] loading donkey from {ckpt_path.name}")
    model = DonkeyWorldRef(cfg).to(device).eval()
    sd = load_file(str(ckpt_path))
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})

    print(f"[eval] lm_head <- {args.lm_head}")
    lm_data = torch.load(args.lm_head, map_location="cpu", weights_only=False)
    lm_head = lm_data["weight"].to(device).to(torch.float32)
    lm_head.requires_grad_(False)

    # Rebuild splits with the exact same seed + fractions
    datasets_root = Path(run_config["datasets_root"])
    trace_dirs = [datasets_root / "traces" / d for d in run_config["datasets"]]
    _, _, verify_ds = split_datasets(
        trace_dirs, cfg.sequence.window_size, cfg.sequence.draft_size,
        trunk_hidden=cfg.trunk.hidden_dim,
        fractions=tuple(split_meta["fractions"]),
        seed=split_meta["split_seed"])
    N = len(verify_ds)
    print(f"[eval] verify split: {N} windows")
    assert N == split_meta["verify_size"], \
        f"verify size mismatch: {N} != {split_meta['verify_size']} (re-split unstable?)"

    K = cfg.sequence.draft_size
    correct = np.zeros((N, K), dtype=np.int8)
    cos = np.zeros((N, K), dtype=np.float32)
    per_trace_correct = {i: np.zeros(K, dtype=np.int64) for i in range(len(trace_dirs))}
    per_trace_count   = {i: np.zeros(K, dtype=np.int64) for i in range(len(trace_dirs))}

    with torch.no_grad():
        i = 0
        while i < N:
            j = min(i + args.batch, N)
            histories = torch.stack([verify_ds[k]["history"]       for k in range(i, j)]).to(device)
            targets   = torch.stack([verify_ds[k]["targets"]       for k in range(i, j)]).to(device)
            tokens    = torch.stack([verify_ds[k]["target_tokens"] for k in range(i, j)]).to(device)
            trace_ids = [verify_ds[k]["trace_id"]                  for k in range(i, j)]

            out = model.forward_for_training(histories)
            z_pred = out["z_pred"]
            pred_decoded = out["pred_decoded"]
            z_target = model.input_proj(targets)

            zp = z_pred.permute(0, 2, 1)
            zt = z_target.permute(0, 2, 1)
            c = torch.nn.functional.cosine_similarity(zp, zt, dim=-1)
            pf = pred_decoded.permute(0, 2, 1).float()
            logits = pf @ lm_head.float().t()
            argmax = logits.argmax(dim=-1)
            corr = (argmax == tokens).int()

            correct[i:j] = corr.cpu().numpy().astype(np.int8)
            cos[i:j]     = c.cpu().numpy().astype(np.float32)
            corr_cpu = corr.cpu().numpy()
            for bi, tid in enumerate(trace_ids):
                per_trace_correct[tid] += corr_cpu[bi]
                per_trace_count[tid]   += 1
            i = j
            if i % (args.batch * 20) == 0 or i == N:
                print(f"[eval]   {i}/{N} ({100.0*i/N:.0f}%) running "
                      f"K0={100*correct[:i,0].mean():.1f}% "
                      f"K1={100*correct[:i,1].mean():.1f}% "
                      f"K2={100*correct[:i,2].mean():.1f}%")

    # Aggregate + per-trace
    print()
    print(f"=== VERIFY RESULTS (run={args.run_dir.name}, ckpt={args.ckpt}, N={N}) ===")
    results = {
        "run_name": args.run_dir.name,
        "ckpt": args.ckpt,
        "n_windows": int(N),
        "aggregate_accept_pct": float(100.0 * correct.mean()),
        "any_K_accept_pct":     float(100.0 * (correct.any(axis=1)).mean()),
        "all_K_accept_pct":     float(100.0 * (correct.all(axis=1)).mean()),
        "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
        "per_K": {},
        "per_trace": {},
    }
    print(f"  aggregate (all K, all windows): {results['aggregate_accept_pct']:.2f}%")
    for k in range(K):
        acc = 100.0 * correct[:, k].mean()
        mc  = float(cos[:, k].mean())
        results["per_K"][f"K{k}"] = {"accept_pct": float(acc), "mean_cos": mc}
        print(f"  K={k}  accept={acc:6.2f}%   mean_cos={mc:.3f}")
    print(f"  any-K accept: {results['any_K_accept_pct']:.2f}%")
    print(f"  all-K accept: {results['all_K_accept_pct']:.2f}%")
    print()
    for tid, name in enumerate(run_config["datasets"]):
        n_total = int(per_trace_count[tid][0])  # same for all K
        if n_total == 0: continue
        per_k = {}
        for k in range(K):
            acc = 100.0 * per_trace_correct[tid][k] / n_total
            per_k[f"K{k}"] = float(acc)
        avg = 100.0 * per_trace_correct[tid].sum() / (n_total * K)
        results["per_trace"][name] = {"n_windows": n_total, "aggregate_pct": float(avg), "per_K": per_k}
        print(f"  {name:30s} N={n_total:>6}  avg={avg:5.2f}%  "
              f"K0={per_k['K0']:.1f}% K1={per_k['K1']:.1f}% K2={per_k['K2']:.1f}%")

    (args.run_dir / "verify_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[eval] -> {args.run_dir / 'verify_results.json'}")


if __name__ == "__main__":
    main()
