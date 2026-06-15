"""
Tokenize HumanEval prompts with MiMo's tokenizer; write flat bins
that the Swift batch selfgen can read without re-implementing BPE.

Output layout (under <out_dir>):
    prompts.bin    int32 stream, all token IDs concatenated
    offsets.bin    int32, length = N+1; prompt i is prompts[offsets[i]:offsets[i+1]]
    prompts_meta.json  { count, total_tokens, max_prompt_len, source, tokenizer_path }

Usage:
    python prepare_humaneval.py <out_dir> [--limit N] [--trunk-dir PATH]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    print("FAIL: 'datasets' not installed; pip install datasets", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--trunk-dir", type=Path,
                    default=Path("/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"))
    ap.add_argument("--max-prompt-len", type=int, default=512,
                    help="truncate prompts longer than this many tokens")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep] loading tokenizer from {args.trunk_dir}")
    tok = AutoTokenizer.from_pretrained(str(args.trunk_dir), trust_remote_code=True)

    print(f"[prep] loading HumanEval")
    ds = load_dataset("openai_humaneval", split="test")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[prep] {len(ds)} problems")

    flat_ids = []
    offsets = [0]
    max_len = 0
    truncated = 0
    for row in ds:
        ids_enc = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,  # MiMo generates its own <think>...</think>
        )
        # apply_chat_template returns either a list of ints or an Encoding;
        # normalize to a plain list of ints.
        # apply_chat_template return type varies by transformers version:
        # - BatchEncoding (dict-like): {"input_ids": [...], "attention_mask": [...]}
        # - tokenizers.Encoding: has .ids attribute
        # - plain list of ints
        if isinstance(ids_enc, dict) or hasattr(ids_enc, "data"):
            ids = ids_enc["input_ids"]
        elif hasattr(ids_enc, "ids"):
            ids = ids_enc.ids
        elif isinstance(ids_enc, list) and ids_enc and isinstance(ids_enc[0], int):
            ids = ids_enc
        else:
            ids = list(ids_enc)
        if len(ids) > args.max_prompt_len:
            ids = ids[: args.max_prompt_len]
            truncated += 1
        if len(ids) == 0:
            continue
        flat_ids.extend(ids)
        offsets.append(len(flat_ids))
        max_len = max(max_len, len(ids))

    flat = np.asarray(flat_ids, dtype=np.int32)
    offs = np.asarray(offsets, dtype=np.int32)
    flat.tofile(args.out_dir / "prompts.bin")
    offs.tofile(args.out_dir / "offsets.bin")

    meta = {
        "count": len(offs) - 1,
        "total_tokens": int(flat.size),
        "max_prompt_len": int(max_len),
        "truncated_count": int(truncated),
        "max_prompt_len_cap": args.max_prompt_len,
        "source": "openai_humaneval/test",
        "tokenizer_path": str(args.trunk_dir),
    }
    (args.out_dir / "prompts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep] wrote {meta['count']} prompts, {meta['total_tokens']} tokens total, max={max_len}")
    if truncated:
        print(f"[prep] truncated {truncated} prompts to {args.max_prompt_len}")


if __name__ == "__main__":
    main()
