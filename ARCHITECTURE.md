# Architecture

## Model

A JEPA-style latent predictor with three learned modules around a frozen trunk readout.

- Encoder phi: maps each trunk pre-norm hidden to a 64-dim latent on a fixed-radius sphere. A window of recent hiddens becomes a latent window.
- Predictor: conditions on the latent window, the emitted-token embedding (the "action"), and a depth embedding (for future multi-step rollout). Outputs the predicted next latent z_hat, on the same sphere.
- Decoder psi: maps a latent back to a standardized hidden state.
- Readout (frozen): the trunk's own final norm + output projection, applied to the decoded hidden to produce a token distribution. Never trained.

At deployment only the prediction path runs: encode window, predict z_hat, decode, read out a token set. The encode-the-true-target path exists only as a training teacher.

## Loss

Combines (tuned weights): a cosine term aligning predicted/target latents; a Jensen-Shannon term between the decoded-prediction token distribution and the trunk's, gated by latent agreement; a teacher term on the decoded true latent; a latent-variance regularizer (prevents hedging collapse); an L2 hidden-fidelity term; and a "psi-match" term distilling the prediction's decode toward the teacher's decode -- the empirically most effective lever. Optional terms (logit distillation, latent-L2, width shaping) are flag-gated and used in controlled experiments.

## Data format

Harvested traces are memmapped binary, one directory per corpus under traces/<corpus>/:
- lastHiddenState.bin : pre-norm hidden states (float32)
- tokens.bin : emitted token ids (int32)
- prompt_idx.bin : per-position sequence delimiters
- topp_{counts,ids,probs}.bin : the trunk's top-p distribution per position (ragged/CSR), the distributional target

The trainer reads corpora by name and joins them onto DONKEY_DATASET; trunk readout weights load from MIMO_DIR.

## Training & checkpoints

Resumable from any checkpoint (weights + optimizer + epoch). Best-on-holdout and per-validation-record checkpoints are written during training. A suite of --probe-* modes load a checkpoint, run a diagnostic on a frozen eval subset, and exit.

## Deployment

The Swift modules (donkey-cli, donkey-runtime, donkey-bridge) target Apple Silicon / ANE inference, where the small drafter's footprint is the design constraint that motivated compressed conditioning.
