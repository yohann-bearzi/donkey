# Stage 3 -- Probe

The diagnostic suite -- the core research contribution. Each probe loads a trained checkpoint, runs an analysis on a frozen eval subset (comparable across checkpoints), and reports. Most are invoked as --probe-* modes of train/train.py; the standalone scripts here are earlier or specialized analyses.

Probe questions:
- round-trip : decode the true encoded hidden -- do we recover the token? (the readout ceiling)
- recency : how does capture degrade with distance to the token's last occurrence? (the decomposition)
- interpolation : walking prediction -> truth, how does the readout flip? (geometry of misses)
- latent-distance : are misses farther in latent space than hits? (directional vs magnitude)
- acceptance : temp-1 speculative acceptance, overlap of draft and trunk distributions (the deployment currency)

Standalone scripts: cap_probe (capture/set-size), rank_fidelity_probe / lmhead_rank_probe (token-rank behavior), stationarity_test (target stability), action_test (action conditioning sensitivity), inspect_weights / quarot_int4_probe (weights / quantization).

See FINDINGS.md for what these established.
