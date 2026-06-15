# Donkey artifact layout

Donkey is a continually-learning world model trained against a specific
trunk's representation space. Its weights are meaningful only for the
trunk they were trained on.

Donkey lives in a `donkey/` subdirectory of the trunk it serves.

Layout:

    <trunk_dir>/                          # trunk owner -- read-only
      config.json
      tokenizer.json
      model-00001-of-00144.safetensors
      ...                                 # trunk files, never touched
      donkey/                             # ours
        donkey.json                       # arch + trunk binding
        weights.bin                       # live training copy
        weights_slow.bin                  # EWMA -- used at inference
        adam_m.bin                        # Adam first moments
        adam_v.bin                        # Adam second moments
        calibration.bin                   # rolling (score, accept) buffer
        stats.json                        # acceptance, ECE, step count
        trunk_binding.json                # hash of trunk it was trained on

## donkey.json schema (JSON)

    {
      "version": 1,
      "model_name": "Donkey-v1",
      "arch": {
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "head_dim": 64,
        "num_layers": 2,
        "max_draft_len": 8,
        "trunk_hidden_dim": 4096,
        "tap_layers": [2, 16, 28]
      },
      "trunk_binding": {
        "trunk_path": "/Volumes/TB5/llm/MiMo-V2-Flash-JANG_4M",
        "trunk_hash": "sha256:...",
        "trunk_model_type": "mimo_v2_flash"
      },
      "training": {
        "step": 12453,
        "total_observed_positions": 198848,
        "current_lr": 1.0e-5
      }
    }

## Behavior on load

1. Check `<trunk_dir>/donkey/donkey.json` exists.
2. Compute trunk hash, compare to `trunk_binding.trunk_hash`.
3. If both match: load weights, enter WARMUP, start training loop.
4. If JSON missing: cold-start WARMUP from fresh weights.
5. If hash mismatch: refuse to use; donkey would be predicting against
   a model it never saw.

## Why a subdirectory and not a sibling

- Co-location makes the trunk -> donkey bond visible on disk.
- Copying or backing up the trunk dir drags donkey along.
- Wiping `donkey/` resets training without touching the trunk.
- Discovery is a single stat() -- no naming convention to search for.
