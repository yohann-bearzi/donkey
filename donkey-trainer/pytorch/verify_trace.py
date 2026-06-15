"""
Phase 5 verifier: read back trace bins, validate shapes + sane norms.

Usage:
    python verify_trace.py /tmp/donkey_trace
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("usage: verify_trace.py <trace_dir>", file=sys.stderr)
        sys.exit(2)

    trace_dir = Path(sys.argv[1])
    meta_path = trace_dir / "meta.json"
    hidden_path = trace_dir / "lastHiddenState.bin"
    tokens_path = trace_dir / "tokens.bin"

    for p in (meta_path, hidden_path, tokens_path):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            sys.exit(3)

    meta = json.loads(meta_path.read_text())
    print(f"[meta] {meta}")

    T = meta["total_positions"]
    H = meta["hidden_dim"]
    assert meta["hidden_dtype"] == "float32", meta
    assert meta["token_dtype"] == "int32", meta

    hidden = np.fromfile(hidden_path, dtype=np.float32)
    if hidden.size != T * H:
        print(f"FAIL: hidden size {hidden.size} != T*H = {T*H}", file=sys.stderr)
        sys.exit(4)
    hidden = hidden.reshape(T, H)

    tokens = np.fromfile(tokens_path, dtype=np.int32)
    if tokens.size != T:
        print(f"FAIL: tokens size {tokens.size} != T = {T}", file=sys.stderr)
        sys.exit(5)

    norms = np.linalg.norm(hidden, axis=-1)
    p50 = float(np.percentile(norms, 50))
    p90 = float(np.percentile(norms, 90))
    p99 = float(np.percentile(norms, 99))
    n_massive = int((norms > 10 * p50).sum())
    print(f"[hidden] shape={hidden.shape} dtype={hidden.dtype}")
    print(f"[hidden] norm  p50={p50:.1f}  p90={p90:.1f}  p99={p99:.1f}  max={norms.max():.1f}")
    print(f"[hidden] massive-activations (>10x p50): {n_massive}/{T} = {100*n_massive/T:.1f}%")
    print(f"[hidden] mean  abs={np.abs(hidden).mean():.4f}")
    print(f"[hidden] any-nan={np.isnan(hidden).any()} any-inf={np.isinf(hidden).any()}")
    print(f"[tokens] shape={tokens.shape} dtype={tokens.dtype}")
    print(f"[tokens] range min={tokens.min()} max={tokens.max()}")
    print(f"[tokens] first 8: {tokens[:8].tolist()}")

    # Real failure modes only: NaN/Inf and full-vector collapse.
    # Massive activations (Sun et al 2024) are expected in modern LLMs.
    # Whitespace/paren tokens routinely develop residual norms 100-1000x
    # typical; this is attention-sink behavior, not corruption. Phase 6
    # loss must handle this (Huber on L2, or standardization).
    if np.isnan(hidden).any() or np.isinf(hidden).any():
        print("FAIL: NaN or Inf in hidden states", file=sys.stderr); sys.exit(6)
    if p50 < 1e-3:
        print(f"FAIL: hidden norm collapsed (p50 {p50})", file=sys.stderr); sys.exit(7)
    if p50 > 1e4:
        print(f"FAIL: even p50 too high ({p50}) — distribution-wide blowup", file=sys.stderr); sys.exit(8)

    print("OK")


if __name__ == "__main__":
    main()
