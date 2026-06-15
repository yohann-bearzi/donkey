"""validate_ops.py — Phase 2b of donkey v2 test harness.

Reads the bin files written by ops-tests, computes the naive reference for
each op in numpy, asserts Swift output agrees within tolerance.

usage: python validate_ops.py <test_dir>
"""
import sys
import json
from pathlib import Path
import numpy as np


def load(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs(a, b):
    return float(np.abs(a.ravel() - b.ravel()).max())


def check(label, swift, expected, tol_cos, tol_abs):
    c = cos(swift, expected)
    m = max_abs(swift, expected)
    ok = (c >= tol_cos) and (m <= tol_abs)
    print(f"  {label}: cos={c:.7f}  max_abs={m:.4e}  ", end="")
    if ok:
        print("OK")
    else:
        print(f"FAIL (need cos>={tol_cos} max_abs<={tol_abs})")
    return ok


# Canonical shapes — must mirror OpsTests.swift.
CH, SP = 1024, 16
FFN_N = 4096 * 16


def naive_rmsnorm(x, gamma, eps=1e-6):
    """x: [CH, SP] row-major. gamma: [CH]. Returns [CH, SP]."""
    x2 = x.reshape(CH, SP)
    ms = (x2 * x2).mean(axis=0, keepdims=True)            # [1, SP]
    rrms = 1.0 / np.sqrt(ms + eps)                         # [1, SP]
    return (x2 * rrms * gamma.reshape(CH, 1)).reshape(-1)


def naive_silu(x):
    # x * sigmoid(x), in fp32. For large |x|, clamp to avoid overflow in expf.
    z = np.clip(-x, -50.0, 50.0)
    return (x / (1.0 + np.exp(z))).astype(np.float32)


def naive_sigmoid(x):
    z = np.clip(-x, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(z))).astype(np.float32)


def main():
    if len(sys.argv) != 2:
        print("usage: validate_ops.py <test_dir>", file=sys.stderr)
        sys.exit(1)
    d = Path(sys.argv[1])
    all_ok = True

    # === 1. RMSNorm — main case ===
    print("[2.1] rmsnorm:")
    x     = load(d / "rmsnorm_input.bin")
    gamma = load(d / "rmsnorm_gamma.bin")
    swift = load(d / "rmsnorm_swift.bin")
    expect = naive_rmsnorm(x, gamma)
    all_ok &= check("main", swift, expect, tol_cos=0.99999, tol_abs=1e-4)

    # Edge: row with all-zero values at one spatial position must produce zeros
    # (and not NaN) at that position.
    xE    = load(d / "rmsnorm_edge_input.bin")
    swE   = load(d / "rmsnorm_edge_swift.bin")
    expE  = naive_rmsnorm(xE, gamma)
    all_ok &= check("edge (zero row)", swE, expE, tol_cos=0.99999, tol_abs=1e-4)
    # Explicit no-NaN check on Swift output.
    if np.isnan(swE).any() or np.isinf(swE).any():
        print("  FAIL: NaN/Inf in rmsnorm edge output"); all_ok = False
    else:
        print("  edge no-NaN check: OK")

    # === 2. SiLU — main case ===
    print("[2.2] silu:")
    xs    = load(d / "silu_input.bin")
    swift = load(d / "silu_swift.bin")
    expect = naive_silu(xs)
    all_ok &= check("main", swift, expect, tol_cos=0.99999, tol_abs=1e-4)

    # Edge: saturation. Swift must produce sane values at ±20.
    xE  = load(d / "silu_edge_input.bin")
    swE = load(d / "silu_edge_swift.bin")
    expE = naive_silu(xE)
    all_ok &= check("edge (sat)", swE, expE, tol_cos=0.999, tol_abs=1e-3)

    # === 3. Residual add ===
    print("[2.3] residual_add:")
    a = load(d / "add_a.bin")
    b = load(d / "add_b.bin")
    swift = load(d / "add_swift.bin")
    expect = (a + b).astype(np.float32)
    # vDSP_vadd is pure SIMD add; must be bit-exact (max_abs=0, cos=1).
    all_ok &= check("main", swift, expect, tol_cos=0.999999, tol_abs=1e-7)

    # Edge: in-place add (a and out same buffer). Result must still equal a+b.
    swift_ip = load(d / "add_inplace_swift.bin")
    all_ok &= check("in-place", swift_ip, expect, tol_cos=0.999999, tol_abs=1e-7)

    # === 4. Sigmoid ===
    print("[2.4] sigmoid:")
    xs = load(d / "sigmoid_input.bin")
    swift = load(d / "sigmoid_swift.bin")
    expect = naive_sigmoid(xs)
    # Saturated cases at ±1000 must be exactly 0 and 1, not NaN.
    if np.isnan(swift).any() or np.isinf(swift).any():
        print("  FAIL: NaN/Inf in sigmoid output"); all_ok = False
    else:
        print("  no-NaN check: OK")
    all_ok &= check("all values", swift, expect, tol_cos=0.99999, tol_abs=1e-4)

    # === 5. Performance regression ===
    print("[2.5] perf:")
    with open(d / "perf.json") as f:
        perf = json.load(f)
    for key, label in [("rmsnorm", "rmsnorm"), ("silu", "silu")]:
        measured = perf[f"{key}_ms"]
        threshold = perf[f"{key}_threshold_ms"]
        if measured <= threshold:
            print(f"  {label}: {measured:.4f}ms <= {threshold:.4f}ms  OK")
        else:
            print(f"  {label}: {measured:.4f}ms > {threshold:.4f}ms  FAIL"); all_ok = False

    print()
    if all_ok:
        print("[ops validation] ALL PASS")
        sys.exit(0)
    else:
        print("[ops validation] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
