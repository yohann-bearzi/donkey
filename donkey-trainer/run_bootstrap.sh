#!/usr/bin/env bash
# Donkey bootstrap pipeline.
#
# Pipeline:
#   1. Prepare prompts: tokenize each --prepare corpus into /Volumes/TB5/donkey/dataset/prompts/<name>/
#   2. Decode traces:   run trace-selfgen-batch for each --decode corpus into traces/<name>_v1/
#   3. Train:           train.py with --datasets <train args> + patience
#   4. Evaluate:        evaluate_run.py on best.safetensors over verify split
#
# Usage:
#   run_bootstrap.sh \
#       --prepare codealpaca_20k python_codes_25k \
#       --decode codealpaca_20k \
#       --train humaneval_v1 mbpp_v1 codealpaca_20k_v1 \
#       --max-steps 50000 --batch 64 --patience 10 --val-every 200
#
# Phases can be skipped: if --decode list is empty, jumps to train.
# If you want to skip train entirely, omit --train.
set -euo pipefail

PROJECT=~/projects/donkey
VENV=~/.venvs/donkey-coreml/bin/python
DATASETS_ROOT=/Volumes/TB5/donkey/dataset
TRACE_BINARY="$PROJECT/.build/arm64-apple-macosx/release/trace-selfgen-batch"

PREPARE=()
DECODE=()
TRAIN=()
MAX_NEW=128
TRAIN_ARGS=()
RUN_NAME=""

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do PREPARE+=("$1"); shift; done
            ;;
        --decode)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do DECODE+=("$1"); shift; done
            ;;
        --train)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TRAIN+=("$1"); shift; done
            ;;
        --max-new) MAX_NEW="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        # anything else is passed through to train.py
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] [bootstrap] $*"; }

# --- Phase 1: prepare prompts ---
if [[ ${#PREPARE[@]} -gt 0 ]]; then
    log "=== PHASE 1: prepare prompts (${#PREPARE[@]} corpora) ==="
    for corpus in "${PREPARE[@]}"; do
        out_dir="$DATASETS_ROOT/prompts/$corpus"
        if [[ -f "$out_dir/prompts.bin" ]]; then
            log "  $corpus -- already prepared, skipping"
        else
            log "  preparing $corpus -> $out_dir"
            $VENV "$PROJECT/donkey-trainer/pytorch/prepare_prompts.py" "$corpus" "$out_dir"
        fi
    done
else
    log "(skipping prepare phase)"
fi

# --- Phase 2: decode traces ---
if [[ ${#DECODE[@]} -gt 0 ]]; then
    log "=== PHASE 2: decode traces (${#DECODE[@]} corpora, maxNew=$MAX_NEW) ==="
    for corpus in "${DECODE[@]}"; do
        prompts_dir="$DATASETS_ROOT/prompts/$corpus"
        trace_dir="$DATASETS_ROOT/traces/${corpus}_v1"
        if [[ -d "$trace_dir" && -f "$trace_dir/meta.json" ]]; then
            log "  ${corpus}_v1 -- trace already exists, skipping"
            continue
        fi
        if [[ ! -f "$prompts_dir/prompts.bin" ]]; then
            log "  ERR $corpus prompts not found at $prompts_dir; run with --prepare $corpus first"
            exit 2
        fi
        log "  decoding $corpus -> $trace_dir"
        cd "$PROJECT"
        "$TRACE_BINARY" "$prompts_dir" "$trace_dir" "$MAX_NEW"
        # Auto-write COLLECTED.md
        cat > "$trace_dir/COLLECTED.md" <<EOF
# ${corpus}_v1 — collected $(date +%Y-%m-%d)

Source: prompts at $prompts_dir
Tokenizer: MiMo-V2-Flash-JANG_4M
Decoder: trace-selfgen-batch, KV-cached, greedy
Decode: maxNew=$MAX_NEW
EOF
        log "  verifying $trace_dir"
        $VENV "$PROJECT/donkey-trainer/pytorch/verify_trace.py" "$trace_dir" || {
            log "  ERR verification failed for $trace_dir"; exit 3;
        }
    done
else
    log "(skipping decode phase)"
fi

# --- Phase 3: train ---
if [[ ${#TRAIN[@]} -gt 0 ]]; then
    log "=== PHASE 3: train on ${TRAIN[*]} ==="
    cd "$PROJECT"
    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME="$(date +%Y-%m-%dT%H-%M)_bootstrap_$(IFS=_; echo "${TRAIN[*]}")"
    fi
    log "  run name: $RUN_NAME"
    $VENV "$PROJECT/donkey-trainer/pytorch/train.py" \
        --datasets "${TRAIN[@]}" \
        --run-name "$RUN_NAME" \
        "${TRAIN_ARGS[@]}"
    log "  train phase done."

    # --- Phase 4: evaluate ---
    log "=== PHASE 4: evaluate best.safetensors on verify split ==="
    $VENV "$PROJECT/donkey-trainer/pytorch/evaluate_run.py" \
        "$DATASETS_ROOT/runs/$RUN_NAME" --ckpt best
else
    log "(skipping train phase)"
fi

log "=== ALL PHASES COMPLETE ==="
