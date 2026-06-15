# Donkey

A latent world-model drafter for speculative decoding. Donkey is a ~12M-parameter JEPA-style model that predicts the next hidden state of a large LLM trunk in a learned 64-dim latent space, then reads tokens out through the trunk's own frozen output head. The goal: cheap multi-token draft proposals, small enough to run on-device (Apple Neural Engine), with the trunk verifying.

Status: research WIP. Single-step drafting is trained and extensively characterized; multi-step rollout is in design. This repo is an honest snapshot of an in-progress investigation. The value is the measured decomposition of where a compressed-conditioning drafter succeeds and fails.

## The idea

Most drafters (Medusa, EAGLE, MTP heads) condition on the trunk's full hidden state. Donkey conditions on a compressed window of recent hiddens, trading conditioning richness for a tiny rollout footprint that fits the ANE. The question this repo answers: what does that compression cost, and where?

## Pipeline

- harvest/ : run trunk over corpora, capture hidden states + token distributions
- train/   : train the JEPA drafter (entry: train/train.py), resumable
- probe/   : diagnostic suite measuring where predictions fail
- donkey-{cli,runtime,bridge}/ : Swift/MLX inference path for Apple Silicon

## Results

See FINDINGS.md. The drafter learns the trunk's conditional dynamics near-perfectly; the remaining gap is an information limit at the conditioning interface, not capacity. Capacity, decoder size, loss pressure, and window width were each tested and ruled out with measurements. The fix is more information (a recent-token side-channel, prompt logging), not a bigger model.

## Layout

harvest/ train/ probe/ docs/ archive/ (old trainers+logs) donkey-*/ (Swift) ckpts/ (gitignored)

## Implementations

Two implementations of the pipeline exist:
- **MLX** (Apple Silicon): the primary track. `train/train.py`, `harvest/`, `probe/`.
- **PyTorch**: an earlier/parallel implementation under `donkey-trainer/pytorch/` (training, trunk-topk extraction, acceptance measurement, dataset prep).

The Swift modules (`donkey-cli`, `donkey-runtime`, `donkey-bridge`) are the deployment/inference path.
