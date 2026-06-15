# Findings

An honest account of what the single-step investigation established. The framing: measure, don't speculate. Every claim is backed by a probe in probe/, and several proposed fixes were killed by measurement rather than argued away.

## Headline

The drafter learns the trunk's conditional dynamics with high fidelity. Given the correct target latent, the decoder reads out the right token at near-ceiling rates including for rare tokens. The gap to that ceiling is not a model-capacity problem -- it is an information limit at the conditioning interface, and it decomposes into named, separately-measured components.

## What was ruled out (with receipts)

- Latent too narrow: doubled latent width, trained to convergence -> tracked the narrower model, no gain.
- Decoder/encoder too small: round-trip probe (decode the true encoded hidden) -> near-perfect readout incl. rare tokens. Not the bottleneck.
- Loss under-optimized: pushed the latent-fidelity term harder -> predictions collapsed toward the conditional mean, capture fell.
- Window too short: stratified capture by distance to the token's last occurrence -> possessing the occurrence in-window was worth ~1.6 points. Widening buys almost nothing.
- Fidelity is the target: compared decode fidelity vs token capture across training -> they decoupled; the readout ceiling is reached at modest fidelity.

## The decomposition

The predictor-vs-ceiling gap on rare tokens splits into:

1. A distance-independent extraction deficit. Even with the target token's own occurrence inside the window, a third of achievable rare capture is missing. The compressed per-position code cannot carry arbitrary lexical identity -- a routing limit, not parameter count.
2. Smooth long-range attenuation. A token's trace fades in the trunk's hiddens with distance; no cliff at the window edge, because the information was never localized at one position.
3. Novel / prompt tokens. The largest bucket of rare misses involves tokens absent from the visible emission history -- retrieval territory, partly unmeasurable until prompt tokens are logged during harvest.

## Geometry of the failure

Two probes characterize how predictions miss, not just how often:

- Interpolation probe: walking from prediction toward the true hidden, capture rises smoothly and monotonically, with a continuous distribution of deficit depths. At the miss, the truth direction is largely absent -- not a hedged distribution containing the truth. This rules out both a hedging story and a categorical-boundary story in favor of a continuum of information deficits.
- Latent-distance probe: rare misses sit at latent distances statistically indistinguishable from rare hits. The failure is directional, not magnitude -- the readout-critical component is a sliver of the residual, invisible to any norm. Common-token misses, by contrast, are genuine magnitude errors -- a cleanly separated second failure mode.

## Why a point predictor blurs

A regression objective's optimum is the conditional mean of the targets. Where the compressed input is consistent with several continuations, that mean is their superposition -- off-manifold, between modes. This is not a bug to optimize away; it is the correct output of a point predictor under ambiguity. Three independently-proposed fixes (a corrective/denoising net, a mean-plus-drift head, manifold projection) were each shown to be no-ops on the same conditioning, because the residual is orthogonal to every function of the input. The only escape is to change the input (supply missing information) or change the output semantics (predict modes, not means, and let the verifier adjudicate).

## What the fixes are, and their measured ceilings

- Exact recent-token side-channel: feed the last K token embeddings directly, bypassing the compression. Targets the extraction deficit; the strata bound its value.
- Prompt-token logging in harvest: unblocks the largest rare bucket; zero model risk.
- Logit-space distillation toward the decode of the true hidden: closes calibration slack on common tokens (the larger share of acceptance loss in the temp-1 currency). The teacher's own acceptance sets a measured ceiling.
- Tree drafting + verifier reset: converts set-capture into accepted length; the system absorbs the irreducible floor because the trunk re-anchors after every block.

## The currency

Single-step token capture is a proxy. The deployment-relevant quantity is temp-1 acceptance (the overlap between draft and trunk distributions) and, ultimately, accepted tokens per draft round under multi-step rollout. The probe suite measures the former directly; the latter is what wave-1 exists to produce, and is the number against which the whole approach should be judged.
