"""Tokenize a prompt corpus with MiMo's tokenizer; write flat int32 bins.

Registry-driven so adding a new dataset = adding a CORPORA entry. Each
entry specifies how to load and what field is the prompt text.

Output layout (under <out_dir>):
    prompts.bin       int32 stream
    offsets.bin       int32, [N+1]
    prompts_meta.json {count, total_tokens, max_prompt_len, source, ...}

Usage:
    prepare_prompts.py <corpus_name> [<out_dir>] [--limit N]
        if out_dir omitted -> /Volumes/TB5/donkey/dataset/prompts/<corpus_name>/
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

DEFAULT_TRUNK = Path("/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M")
DEFAULT_ROOT  = Path("/Volumes/TB5/donkey/dataset/prompts")


CORPORA = {
    "humaneval": {
        "loader":  ("openai_humaneval", None, ["test"]),
        "fields":  ["prompt"],
    },
    "mbpp": {
        "loader":  ("mbpp", "full", ["train", "test", "validation", "prompt"]),
        "fields":  ["text", "prompt"],
    },
    "codealpaca_20k": {
        "loader":  ("sahil2801/CodeAlpaca-20k", None, ["train"]),
        "fields":  ["instruction"],
        "extra_fields": ["input"],  # appended after instruction if non-empty
    },
    "evol_instruct_code_80k": {
        "loader":  ("nickrosh/Evol-Instruct-Code-80k-v1", None, ["train"]),
        "fields":  ["instruction"],
    },
    "conala_mined": {
        "loader":  ("neulab/conala", "mined", ["train"]),
        "fields":  ["intent", "snippet"],
    },
    "conala_curated": {
        "loader":  ("neulab/conala", "curated", ["train", "test"]),
        "fields":  ["intent", "rewritten_intent"],
    },
    "python_codes_25k": {
        "loader":  ("flytech/python-codes-25k", None, ["train"]),
        "fields":  ["instruction"],
    },
    "code_search_net_python": {
        "loader":  ("code_search_net", "python", ["train"]),
        "fields":  ["func_documentation_string", "func_code_string"],
    },
}


def extract_text(row, fields, extra_fields=None):
    base = None
    for f in fields:
        if f in row and row[f]:
            v = row[f]
            if isinstance(v, str) and v.strip():
                base = v
                break
    if base is None: return None
    if extra_fields:
        for ef in extra_fields:
            if ef in row and isinstance(row[ef], str) and row[ef].strip():
                base = base + "\n\n" + row[ef]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(CORPORA.keys()))
    ap.add_argument("out_dir", type=Path, nargs="?", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trunk-dir", type=Path, default=DEFAULT_TRUNK)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = DEFAULT_ROOT / args.corpus

    spec = CORPORA[args.corpus]
    print(f"[prep] corpus={args.corpus}  out={args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep] tokenizer from {args.trunk_dir}")
    tok = AutoTokenizer.from_pretrained(str(args.trunk_dir), trust_remote_code=True)

    from datasets import load_dataset, concatenate_datasets
    name, config, splits = spec["loader"]
    parts = []
    for sp in splits:
        try:
            parts.append(load_dataset(name, config, split=sp))
        except Exception as e:
            print(f"[prep]   split {sp!r}: skipping ({type(e).__name__})")
    if not parts:
        print("FAIL: no splits loaded", file=sys.stderr); sys.exit(2)
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[prep] {len(ds)} rows from {name}/{config}/{splits}")

    flat, offsets = [], [0]
    max_len = 0
    truncated = 0
    empty = 0
    fields = spec["fields"]
    extra_fields = spec.get("extra_fields")
    for row in ds:
        text = extract_text(row, fields, extra_fields)
        if text is None:
            empty += 1; continue
        ids_enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
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
            ids = ids[: args.max_prompt_len]; truncated += 1
        if not ids:
            empty += 1; continue
        flat.extend(ids); offsets.append(len(flat))
        max_len = max(max_len, len(ids))

    np.asarray(flat, dtype=np.int32).tofile(args.out_dir / "prompts.bin")
    np.asarray(offsets, dtype=np.int32).tofile(args.out_dir / "offsets.bin")
    meta = {
        "corpus": args.corpus,
        "source": f"{name}/{config or 'default'}/{','.join(splits)}",
        "fields_tried": fields,
        "count": len(offsets) - 1,
        "total_tokens": len(flat),
        "max_prompt_len": max_len,
        "max_prompt_len_cap": args.max_prompt_len,
        "truncated_count": truncated,
        "empty_skipped": empty,
        "tokenizer_path": str(args.trunk_dir),
    }
    (args.out_dir / "prompts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prep] count={meta['count']} total_tokens={meta['total_tokens']} max_len={max_len}  "
          f"truncated={truncated} empty={empty}")


if __name__ == "__main__":
    main()
