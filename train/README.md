# Stage 2 -- Train

Train the JEPA drafter against harvested hiddens.

  DONKEY_DATASET=/path/to/dataset MIMO_DIR=/path/to/trunk \
  python train.py <corpus> [<corpus> ...] --holdout <corpus> \
      --d 64 --dp 256 [--lam-* ...] [--resume <ckpt>] --out <ckpt-stem> --save-opt

- train.py : the trainer (encoder/predictor/decoder + the loss in ARCHITECTURE.md). Self-contained.
- sweep.py : hyperparameter sweep helper
- DONKEY_TRAINING_DESIGN.md : design notes

Resumable (weights + optimizer + epoch); writes best-on-holdout and per-validation-record checkpoints. The --probe-* flags load a checkpoint and run a diagnostic (see probe/).

Note: --out must be set or checkpointing is silently skipped. No save-on-interrupt handler yet -- kills lose the current epoch. Known rough edges of the WIP state.
