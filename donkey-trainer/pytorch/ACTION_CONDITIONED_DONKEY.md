# Action-Conditioned Donkey: A LeWorldModel-Faithful Architecture

Filed 2026-05-18, after the Convention-2 bootstrap completed (K=0 = 58%
on verify) and while KL distillation training is in flight.

## What this changes

Donkey today predicts K=3 future hidden states **in parallel** from
history alone:

    predict(history[t..t+W-1]) → (z_pred[0], z_pred[1], z_pred[2])

Each z_pred[k] is computed independently. Slot k+1's prediction is
*not* conditioned on what we'd pick at slot k. This means donkey's
top-N candidates at slot 1 are the same regardless of which token we
choose at slot 0.

LeWorldModel (LeWM, Maes et al. 2026) shows the principled JEPA
architecture for this kind of dynamics modeling: action-conditioned,
autoregressive prediction. The predictor takes (state, action) → 
next_state and is rolled out step by step:

    predict_step(z_t, a_t) → z_{t+1}
    predict_step(z_{t+1}, a_{t+1}) → z_{t+2}
    ...

Action conditioning is done via Adaptive LayerNorm (AdaLN) — the same
scale-shift trick used in DiT. Crucially, the AdaLN parameters are
zero-initialized, so the predictor begins as identity-on-action and
gradually learns to use the action signal. This zero-init is the
detail that makes training stable end-to-end.

## Why this fits donkey unusually well

We already have ~2/3 of LeWM:

| LeWM piece               | Donkey equivalent          | Status         |
|--------------------------|----------------------------|----------------|
| Encoder o → z            | HistoryProjection (Φ_O)    | ✅             |
| Predictor (z,a) → ẑ      | Predictor stack            | ✅ (no action) |
| L_pred (MSE on z)        | Huber on z_target          | ✅             |
| SIGReg                   | SIGReg                     | ✅             |
| Action conditioning      | -                          | ❌             |
| Autoregressive rollout   | parallel K=3               | ❌             |
| AdaLN, zero-init         | plain LayerNorm            | ❌             |

The architectural commitment is small. We're not redesigning donkey —
we're closing the gap between what we have and what the published
LeWM recipe says works.

## What "action" means in our LM setting

In LeWM the action is a motor command. In ours:

    a_t = embed_tokens[token_t]

That is, the action at position t is the embedding of the token MiMo
emitted at position t. This is a 4096-d vector. It is exactly the
thing that gets injected into MiMo's residual stream when that token
is fed back at step t+1.

Why this is the right action representation:
- It is the actual input that produces h[t+1] in the true trunk
  rollout, so donkey learning (z_t, embed(a)) → z_{t+1} is learning
  the dynamics of the real trunk
- Different tokens have different embeddings → different actions →
  different predicted next states. This is exactly the counterfactual
  capacity we need for tree-spec deploy
- We have embed_tokens for free in MiMo's safetensors

## The AdaLN mechanism (concrete)

In each predictor transformer block, replace plain LayerNorm with:

    AdaLN(x, a) = LayerNorm(x) * (1 + γ(a)) + β(a)
    γ, β = MLP(a)        # MLP output projection: zero-init

There are typically two LayerNorms per transformer block (pre-attention,
pre-MLP). Both become AdaLN, both conditioned on the same action
embedding for that step.

The zero-init means at step 0 of training, γ=β=0, so AdaLN(x, a) =
LayerNorm(x). The predictor starts as if there's no action conditioning
at all. Gradients gradually teach it to use the action signal. LeWM
attributes training stability to this single detail.

## Architecture, concretely

Inputs at training:
- history: [B, W, 4096] (residual stream from trace)
- actions: [B, K, 4096] (token embeddings for positions W..W+K-1)
- z_targets: [B, K, 1024] (Φ_O applied to future hidden, the labels)

Forward:
1. z_history = Φ_O(history)         # [B, W, 1024]
2. For k in 0..K-1:
   - z_input = z_history (for k=0) or rolling buffer (for k>0)
   - a = actions[:, k]               # the action taken at this slot
   - z_pred[k] = predictor(z_input, a)
   - append z_pred[k] to rolling buffer for next step
3. Compute losses on (z_pred, z_targets)

Loss: per LeWM:

    L = L_pred + λ · SIGReg(Z)
      = Huber(z_pred, z_target.detach()) + λ · SIGReg(z_pred ∪ z_target)

No CE term. No decoder-aware loss. Pure JEPA-style world model.
Decoding via lm_head is post-hoc, only at evaluation/deploy.

Initial λ = 0.01 (our current value, in the LeWM-recommended range
[0.01, 0.2]).

## Inference: tree rollout

This is where action conditioning pays off:

1. z_0 = Φ_O(trunk_history)
2. Generate top-N candidate first tokens:
   - decode(z_0) → token distribution → top-N IDs a^1, ..., a^N
3. For each candidate i, predict z_1^i = predictor(z_0, embed(a^i))
4. For each branch i, decode z_1^i → top-N candidate second tokens b^{i,j}
5. For each (i,j), predict z_2^{i,j} = predictor(z_1^i, embed(b^{i,j}))
6. Recurse to depth K=3 → tree with up to N^3 leaves
7. Walk the tree by joint probability, pick top-M paths
8. Submit to MiMo as one batched verify forward (tree attention mask)

Each branch is a *coherent* rollout: slot 2 in branch (A, B) is
conditioned on having taken A at slot 0 and B at slot 1, not on the
unconditional argmax future.

## Honest gaps

### 1. Training data is greedy-only

We only have (z_t, argmax_action, z_{t+1}) triples. Donkey learns
dynamics for argmax actions only. At inference we'll query with
non-argmax actions (top-2, top-3, ...). This is extrapolation.

The token embedding space has semantic structure — similar tokens
have similar embeddings — so extrapolation might be smooth for
semantically-close alternatives (`return` vs `yield`) and rough for
wild alternatives.

We won't know without measuring. The post-hoc fix is branch-exploration
data collection: at training-time, fork at high-entropy positions,
collect both branches. ~14h decode if pursued, but only justified if
greedy-only training shows clear extrapolation failure.

### 2. Autoregressive training is slower than parallel

Predicting K=3 steps autoregressively means 3 sequential forwards per
training example. About 3× training wall time vs current parallel
predictor. Acceptable: ~2-4h overnight runs instead of 1.5h.

### 3. We add embed_tokens (1.25 GB) to the deploy footprint

Same dequantization recipe as lm_head dump. Need to run
`dump_token_embeddings.py` once. Storage cost: 1.25 GB next to the
existing 1.25 GB lm_head. Negligible.

## What we drop

CE loss is gone. This is principled (LeWM doesn't have it) but worth
flagging: we lose direct token-level signal during training. The model
is optimized to predict the residual stream's evolution; how well that
residual stream decodes to tokens is measured at eval time, not
trained directly.

This is what makes donkey a "world model" rather than a "token
predictor." The architectural commitment is real.

## Code changes (files and sizes)

1. **dump_token_embeddings.py** (new, ~80 lines)
   Mirror of dump_lm_head.py, source_kind="embed_tokens".

2. **donkey_world.py** (modify, ~120 lines added)
   - New `AdaLN` module
   - Predictor block uses AdaLN for both LayerNorms
   - Predictor forward: takes (z, a) instead of (history)
   - `forward_for_training_ar`: autoregressive K-step loop

3. **trace_dataset.py** (modify, ~30 lines)
   - Load embed_tokens.pt once at dataset init
   - Return `actions = embed_tokens[tokens[t+W..t+W+K-1]]` in batch

4. **train.py** (modify, ~60 lines)
   - `--action-conditioned` flag
   - When set: drop CE term, use AR forward, log AdaLN gamma/beta norms
   - Same patience/val/checkpoint mechanics as before

5. **losses.py** (modify, ~20 lines)
   - Pure JEPA loss helper: Huber(z_pred, z_target) + SIGReg
   - No new loss conceptually, just a configuration without CE

6. **evaluate_run.py** (modify, ~80 lines)
   - Add tree-N accept measurement
   - Walk tree at N=2, 4, 8 candidates per slot
   - Report per-K-trunk top-N accept

7. **donkey-runtime** Swift (defer)
   - Tree rollout at deploy. Significant Swift work. After Python
     architecture is validated.

Total Python work: ~6 hours careful coding + smoke tests.

## Experiment plan

**Phase 0** (in progress): KL distillation completes. Record verify
result as second baseline. Compare to CE baseline (K=0 = 58%).

**Phase 1**: Implement files 1-6. Smoke on humaneval (~40K windows)
to confirm:
- AdaLN modules don't NaN
- Predictor with AdaLN converges to non-trivial latent quality
- Tree-N accept is non-zero at random init (sanity)

**Phase 2**: Full training on all three corpora with
`--action-conditioned`. ~3-4h wall. Patience-driven stop. Record verify
results.

**Phase 3**: Three-way comparison table:
- CE baseline: K=0=58%, K=1=45%, K=2=33%, any-K=75%
- KL distill:  (filled in from current run)
- AR JEPA:     (filled in from Phase 2)

The win condition: AR JEPA has comparable K=0 to CE baseline AND
substantially higher tree-N accept (because conditional rollout
produces coherent branches).

If win condition is met, proceed to deploy infrastructure (Swift tree
rollout). If not, investigate failure mode — most likely candidate is
greedy-only training data limitation, which we'd address by branch
exploration data collection.

## Why this is the right next step

Three reasons in order of importance:

1. **Faithful to a recipe that works.** LeWM is published, peer
   reviewed (well, on arxiv with 2 months of community discussion),
   and tested on robotic manipulation. The action-conditioning +
   AdaLN-with-zero-init recipe is not novel; we're porting it from
   a domain where it's been shown to converge stably.

2. **It addresses the architectural confusion in current donkey.**
   We called donkey a world model but trained it as a token predictor
   (CE dominates gradients). AR JEPA actually trains a world model.
   Either we commit to the framing or we don't; this is the commit.

3. **Tree-N accept is the deploy-relevant metric.** The bottleneck
   for tree-speculative decoding on Apple Silicon is "does donkey
   produce coherent N-wide branching candidates." Current parallel-
   prediction donkey can't, by construction. Action-conditioned AR
   donkey can.

## What we keep from prior work

- Convention 2 traces (self-aligned hidden + token)
- SIGReg (already there)
- Huber/MSE on z_target with adaptive delta
- 90/5/5 prompt-id splits
- Patience-driven training infrastructure
- evaluate_run.py forensics
- HistoryProjection (Φ_O) including LayerNorm at output

All of these compose cleanly with AR JEPA. Nothing has to be unwound.

## Status

- Design doc filed: this file
- Convention 2 baseline: K=0=58% on 135K verify windows (logged)
- KL distillation: in flight, ETA tonight
- Action-conditioned donkey: not started, next after distillation lands
