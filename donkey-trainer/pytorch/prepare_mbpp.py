"""Tokenize MBPP prompts with MiMo's tokenizer; write same flat bin layout as
prepare_humaneval.py. MBPP problems are short prompt-style code tasks similar
in shape to HumanEval but ~6x more of them.

Usage: prepare_mbpp.py <out_dir> [--limit N] [--trunk-dir PATH]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    print("FAIL: pip install datasets", file=sys.stderr); sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trunk-dir", type=Path,
                    default=Path("/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M"))
    ap.add_argument("--max-prompt-len", type=int, default=512)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prep-mbpp] loading tokenizer from {args.trunk_dir}")
    tok = AutoTokenizer.from_pretrained(str(args.trunk_dir), trust_remote_code=True)
    print(f"[prep-mbpp] loading mbpp/full (all splits)")
    from datasets import concatenate_datasets
    parts = [load_dataset("mbpp", "full", split=sp)
             for sp in ("train", "test", "validation", "prompt")]
    ds = concatenate_datasets(parts)
    print(f"[prep-mbpp] {len(ds)} problems (across train+test+validation+prompt)")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    flat, offsets, max_len, truncated = [], [0], 0, 0
    for row in ds:
        # MBPP "sanitized": fields are 'prompt' (text task) + 'code' + 'test_list'
        # The prompt-style for the LM is the natural-language description.
        prompt_text = row.get("prompt") or row.get("text") or ""
        ids_enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
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
            ids = ids[:args.max_prompt_len]; truncated += 1
        if not ids:
            continue
        flat.extend(ids); offsets.append(len(flat))
        max_len = max(max_len, len(ids))

    np.asarray(flat, dtype=np.int32).tofile(args.out_dir / "prompts.bin")
    np.asarray(offsets, dtype=np.int32).tofile(args.out_dir / "offsets.bin")
    meta = {
        "count": len(offsets) - 1,
        "total_tokens": len(flat),
        "max_prompt_len": max_len,
        "truncated_count": truncated,
        "max_prompt_len_cap": args.max_prompt_len,
        "source": "mbpp/full (train+test+validation+prompt)",
        "tokenizer_path": str(args.trunk_dir),
    }
    (args.out_dir / "prompts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep-mbpp] wrote {meta['count']} prompts, {meta['total_tokens']} tokens, max={max_len}")
    if truncated:
        print(f"[prep-mbpp] truncated {truncated}")


if __name__ == "__main__":
    main()
