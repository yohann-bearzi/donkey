"""Extract MiMo's embed_tokens, dequantizing JANG-quantized weights via MLX.

Same recipe as dump_lm_head.py but for the input embedding matrix. Needed
as the ground metric for token-space Wasserstein in stochastic donkey.

Output: torch checkpoint with weight [vocab, hidden]=152576x4096 fp16, ~1.25 GB

Usage: dump_embed_tokens.py <trunk_dir> <out_path>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trunk_dir", type=Path)
    ap.add_argument("out_path", type=Path)
    args = ap.parse_args()

    index = json.loads((args.trunk_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((args.trunk_dir / "config.json").read_text())
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])

    keys = sorted(k for k in weight_map if k.startswith("model.embed_tokens"))
    print(f"[dump-embed] embed_tokens keys: {keys}")
    if not keys:
        print(f"FAIL: no embed_tokens.* keys in safetensors index", file=sys.stderr)
        sys.exit(2)

    tensors = {}
    for k in keys:
        with safe_open(args.trunk_dir / weight_map[k], framework="pt") as f:
            t = f.get_tensor(k)
            tensors[k] = t
            print(f"[dump-embed]   {k}: shape={tuple(t.shape)} dtype={t.dtype}")

    weight_key = next((k for k in tensors if k.endswith(".weight")), None)
    scales_key = next((k for k in tensors if k.endswith(".scales")), None)
    biases_key = next((k for k in tensors if k.endswith(".biases")), None)
    if not (weight_key and scales_key and biases_key):
        print(f"FAIL: missing .weight/.scales/.biases keys", file=sys.stderr)
        sys.exit(3)

    w_packed = tensors[weight_key]
    scales = tensors[scales_key]
    biases = tensors[biases_key]
    packed_cols = w_packed.shape[1]
    bits = (32 * packed_cols) // hidden
    num_groups = scales.shape[1]
    group_size = hidden // num_groups
    print(f"[dump-embed] derived: bits={bits} group_size={group_size}")

    import mlx.core as mx
    w_np = w_packed.numpy()
    if w_np.dtype != np.uint32:
        w_np = w_np.view(np.uint32)

    w_mx = mx.array(w_np)
    s_mx = mx.array(scales.to(torch.float16).numpy())
    b_mx = mx.array(biases.to(torch.float16).numpy())
    print("[dump-embed] mx.dequantize...")
    w_fp = mx.dequantize(w_mx, scales=s_mx, biases=b_mx,
                         group_size=group_size, bits=bits)
    mx.eval(w_fp)
    w_arr = np.array(w_fp).astype(np.float32)
    if w_arr.shape != (vocab, hidden):
        print(f"FAIL: shape {w_arr.shape} != ({vocab},{hidden})", file=sys.stderr)
        sys.exit(7)

    print(f"[dump-embed] stats: min={w_arr.min():.4f}  max={w_arr.max():.4f}  "
          f"mean={w_arr.mean():.6f}  std={w_arr.std():.4f}")

    weight = torch.from_numpy(w_arr).to(torch.float16).contiguous()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "source_key": weight_key,
                "vocab_size": vocab, "hidden_size": hidden,
                "dequantized": True, "bits": bits, "group_size": group_size},
               args.out_path)
    sz_gb = args.out_path.stat().st_size / 1e9
    print(f"[dump-embed] saved {tuple(weight.shape)} -> {args.out_path}")
    print(f"[dump-embed] size: {sz_gb:.2f} GB")


if __name__ == "__main__":
    main()
