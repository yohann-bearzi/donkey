#!/usr/bin/env bash
# tests/run_all.sh — run every donkey test in sequence. Grows as phases land.
#
# Each test is fast (<5s), deterministic, idempotent. Adding a new test means
# appending a `run_one ...` line below.
set -e
cd "$(dirname "$0")/.."

VENV=$HOME/.venvs/donkey-coreml
PYTHON=$VENV/bin/python

PASS=0; FAIL=0
run_one() {
    local label="$1"; shift
    printf '%-50s ' "$label"
    if "$@" > /tmp/test_out.$$ 2>&1; then
        echo "OK"
        PASS=$((PASS+1))
    else
        echo "FAIL"
        cat /tmp/test_out.$$
        FAIL=$((FAIL+1))
    fi
    rm -f /tmp/test_out.$$
}

echo "=== donkey v2 test harness ==="

# --- Phase 1: DonkeyConfig ---
run_one "[1.1] Swift  DonkeyConfig load + invariants" \
    .build/arm64-apple-macosx/release/config-smoke

run_one "[1.2] Python DonkeyConfig load + invariants" \
    "$PYTHON" donkey-trainer/pytorch/test_donkey_config.py

# --- Phase 2: DonkeyOps (correctness + edge cases + perf) ---
# Two-step: Swift writes inputs+outputs, Python computes naive reference + asserts.
OPS_DIR=/tmp/donkey_ops_test_run
rm -rf "$OPS_DIR"
run_one "[2.0] Swift  OpsTests writes bins" \
    .build/arm64-apple-macosx/release/ops-tests "$OPS_DIR"

run_one "[2.1-2.5] Python ops validation (4 ops + perf)" \
    "$PYTHON" donkey-trainer/pytorch/validate_ops.py "$OPS_DIR"

# --- Phase 3: PyTorch DonkeyWorldRef (forward + determinism + golden) ---
run_one "[3.1-3.3] Python DonkeyWorldRef forward + golden" \
    "$PYTHON" donkey-trainer/pytorch/test_donkey_world.py

# --- Phase 4: Swift DonkeyWorldForward orchestrator ---
run_one "[4.1] Swift  DonkeyWorldForward kernel compile" \
    .build/arm64-apple-macosx/release/world-compile-smoke

run_one "[4.2] Swift  DonkeyWorldForward end-to-end forward" \
    .build/arm64-apple-macosx/release/world-forward-smoke

# --- Phase 4.5: Swift orchestrator vs PyTorch reference cosine ---
# Two-step: Swift dumps weights+output, Python loads same weights, runs ref, compares.
DUMP_DIR=/tmp/donkey_world_xval
rm -rf "$DUMP_DIR"
run_one "[4.5a] Swift  WorldDumpSmoke writes weights + output" \
    .build/arm64-apple-macosx/release/world-dump-smoke "$DUMP_DIR"

# Python: load Swift weights, run reference, write outputs (no separate test entry;
# just the side-effect of the cosine comparator below pulling these in).
run_one "[4.5b] Python load Swift weights + run reference" \
    "$PYTHON" donkey-trainer/pytorch/run_world_reference.py "$DUMP_DIR"

run_one "[4.5c] Swift vs PyTorch cosine (>= 0.99)" \
    "$PYTHON" donkey-trainer/pytorch/test_world_cosine.py "$DUMP_DIR"

# --- Phase 5: trace collection (MiMo trunk required; ~30s load) ---
# Gated behind DONKEY_FULL=1 so the fast inner loop stays <10s.
if [ "${DONKEY_FULL:-0}" = "1" ]; then
    TRACE_DIR=/tmp/donkey_trace_test
    rm -rf "$TRACE_DIR"
    run_one "[5.1a] Swift  trace-collect-smoke writes bins" \
        .build/arm64-apple-macosx/release/trace-collect-smoke "$TRACE_DIR"

    run_one "[5.1b] Python verify trace (shapes + sane norms)" \
        "$PYTHON" donkey-trainer/pytorch/verify_trace.py "$TRACE_DIR"

    SELFGEN_DIR=/tmp/donkey_selfgen_test
    rm -rf "$SELFGEN_DIR"
    run_one "[5.2a] Swift  trace-selfgen-smoke (KV-cached, 16 tok)" \
        .build/arm64-apple-macosx/release/trace-selfgen-smoke "$SELFGEN_DIR" 16 42

    run_one "[5.2b] Python verify selfgen trace" \
        "$PYTHON" donkey-trainer/pytorch/verify_trace.py "$SELFGEN_DIR"
else
    echo "[5.1] skipped (set DONKEY_FULL=1 to run trace-collect; ~30s)"
fi


echo ""
echo "=== results: $PASS passed, $FAIL failed ==="
[ "$FAIL" = "0" ]
