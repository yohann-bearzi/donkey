# Donkey training - Option B (LeWM-faithful latent world model)

Trunk = MiMo-V2.5 block_fp8 (frozen, "renderer"). Donkey = JEPA world model over the
trunk pre-norm hidden stream. Predict next ENCODED hidden in a learned latent, as
LeWorldModel (arXiv:2603.19312) predicts the next encoded frame.

Analogy: image:world :: trunk_hidden(4096):meaning.
  encoder phi: h->z(d). predictor: z_t,a_t->z_t+1. decoder psi: z->h(4096).
  psi is LOAD-BEARING for us (LeWM's is viz-only): we need lm_head(psi(z))->token.

Choices vs paper:
 1. Predict in latent: L_pred=||zhat-phi(h_next)||^2 (target=ENCODED next).
 2. SIGReg only, lambda=0.1, M=1024, Epps-Pulley. No cos/CE/stopgrad/EMA in P1.
 3. AdaLN-zero action conditioning per predictor layer (one action per transition).
 4. Causal predictor, window W=13, 6 layers (LeWM depth), projector head.
 5. Encoder shallow 2L (input already semantic). phi ends in plain Linear (norm fights SIGReg).
 6. Decoder post-trained P2, frozen phi+pred, CE through frozen lm_head.
 7. latent d in {64,128,256,512}; d=64=sqrt(4096).

Phases:
 P1: train phi+pred. loss=MSE+0.1*SIGReg. report predMSE/cos, latent_std (collapse monitor).
 P2: freeze, train psi via CE through lm_head. 3 diagnostics:
   (1) recon ceiling argmax(lm(psi(phi(h))))==trunk  (2) dynamics MSE/cos  (3) end-to-end accept.

SRAM 128MB: Dp=256/384 train on-chip fp16; Dp=512 inference/4bit tier.
Option A (--option a): phi=psi=identity, predict in trunk 4096-space (baseline).
