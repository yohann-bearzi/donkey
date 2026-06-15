"""
Extract MiMo's lm_head, dequantizing JANG-quantized weights via MLX.

Output: torch checkpoint at <out_path> with:
    weight       fp16 [vocab, hidden]
    source_key, source_kind, vocab_size, hidden_size, dequantized, bits, group_size

JANG_4M stores lm_head as affine-quantized triplet (.weight uint32-packed,
.scales fp16 per group, .biases fp16 per group). We use mx.dequantize
which is the same path MLX uses at trunk inference time — donkey trains
against precisely the head it will face at deploy.

Usage: dump_lm_head.py <trunk_dir> <out_path>
"""
import argparse
import json
import sys
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
    tie = bool(config.get("tie_word_embeddings", False))
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])

    lm_keys = sorted(k for k in weight_map if k.startswith("lm_head"))
    embed_keys = sorted(k for k in weight_map if k.startswith("model.embed_tokens"))
    print(f"[dump] vocab={vocab} hidden={hidden} tie_word_embeddings={tie}")
    print(f"[dump] lm_head keys: {lm_keys}")
    print(f"[dump] embed_tokens keys: {embed_keys}")

    if lm_keys:
        source_keys, source_kind = lm_keys, "lm_head"
    elif embed_keys and tie:
        source_keys, source_kind = embed_keys, "tied_embed_tokens"
    elif embed_keys:
        source_keys, source_kind = embed_keys, "embed_tokens (untied fallback)"
    else:
        print("FAIL: no lm_head or embed_tokens", file=sys.stderr); sys.exit(2)
    print(f"[dump] source: {source_kind}")

    tensors = {}
    for k in source_keys:
        with safe_open(args.trunk_dir / weight_map[k], framework="pt") as f:
            t = f.get_tensor(k)
            tensors[k] = t
            print(f"[dump]   {k}: shape={tuple(t.shape)} dtype={t.dtype}")

    # Unquantized case
    if len(tensors) == 1:
        k, t = next(iter(tensors.items()))
        if tuple(t.shape) == (vocab, hidden):
            save(args.out_path, t.to(torch.float16).contiguous(), k,
                 source_kind, vocab, hidden, dequant=False)
            return

    weight_key  = next((k for k in tensors if k.endswith(".weight")), None)
    scales_key  = next((k for k in tensors if k.endswith(".scales")), None)
    biases_key  = next((k for k in tensors if k.endswith(".biases")), None)
    if not (weight_key and scales_key and biases_key):
        print(f"FAIL: missing .weight/.scales/.biases", file=sys.stderr); sys.exit(3)

    w_packed = tensors[weight_key]
    scales   = tensors[scales_key]
    biases   = tensors[biases_key]

    packed_cols = w_packed.shape[1]
    bits        = (32 * packed_cols) // hidden
    num_groups  = scales.shape[1]
    group_size  = hidden // num_groups
    print(f"[dump] derived: bits={bits}  group_size={group_size}  packed_cols={packed_cols}")
    if bits * hidden != 32 * packed_cols:
        print("FAIL: bits/hidden/packed inconsistent", file=sys.stderr); sys.exit(4)
    if num_groups * group_size != hidden:
        print("FAIL: group_size doesn't tile hidden", file=sys.stderr); sys.exit(5)

    try:
        import mlx.core as mx
    except ImportError:
        print("FAIL: mlx not installed; pip install mlx", file=sys.stderr); sys.exit(6)

    w_np = w_packed.numpy()
    if w_np.dtype != np.uint32:
        w_np = w_np.view(np.uint32)
    s_np = scales.to(torch.float16).numpy()
    b_np = biases.to(torch.float16).numpy()

    print("[dump] mx.dequantize...")
    w_mx = mx.array(w_np)
    s_mx = mx.array(s_np)
    b_mx = mx.array(b_np)
    w_fp = mx.dequantize(w_mx, scales=s_mx, biases=b_mx,
                         group_size=group_size, bits=bits)
    mx.eval(w_fp)
    w_arr = np.array(w_fp).astype(np.float32)

    if w_arr.shape != (vocab, hidden):
        print(f"FAIL: dequant shape {w_arr.shape} != ({vocab},{hidden})", file=sys.stderr)
        sys.exit(7)

    print(f"[dump] dequant stats: min={w_arr.min():.4f}  max={w_arr.max():.4f}  "
          f"mean={w_arr.mean():.6f}  std={w_arr.std():.4f}")

    weight = torch.from_numpy(w_arr).to(torch.float16).contiguous()
    save(args.out_path, weight, weight_key, source_kind, vocab, hidden,
         dequant=True, bits=bits, group_size=group_size)


def save(out_path, weight, source_key, source_kind, vocab, hidden, dequant,
         bits=None, group_size=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weight": weight,
        "source_key": source_key,
        "source_kind": source_kind,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "dtype": str(weight.dtype),
        "dequantized": dequant,
    }
    if dequant:
        payload["bits"] = bits
        payload["group_size"] = group_size
    torch.save(payload, out_path)
    print(f"[dump] saved {tuple(weight.shape)} {weight.dtype} -> {out_path}")
    print(f"[dump] size: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
