// donkey_v1.h — Donkey world-model drafter v1
//
// 2-layer MHA transformer. Predicts K future trunk hidden states per
// draft round in a single ANE dispatch (parallel-K, not autoregressive).
// Loss: JEPA L2 + cosine + CE-through-trunk-lm_head + calibration on
// the confidence head. See docs/DONKEY_NEURAL_NET_V1.md for full design.
//
// Sizing (~45M params, ~92 MB fp16, ~25 MB at p4 palettization):
//   - hidden_dim 1024 (down-projected from MiMo trunk's 4096)
//   - 2 transformer blocks, MHA (16 heads x 64 head_dim)
//   - SiLU FFN (gate-less): up 1024->4096, silu, down 4096->1024
//     (not SwiGLU; +33% FFN params not worth it for v1, per Q4)
//   - Input adapter: 3 EAGLE-3 taps fused via 12288->1024 row-blocked
//     sum (3 x conv 4096->1024, no concat op needed on ANE)
//   - K=3 parallel draft positions, K_PAD=8 reserves headroom for the
//     ANE minimum-SP question (measure first; may need 16/32)
#pragma once

#define MODEL_NAME "Donkey-v1"

#define DIM       1024
#define HIDDEN    4096           // 4x DIM, SiLU FFN (gate-less, not SwiGLU)
#define HEADS     16
#define KV_HEADS  16             // MHA
#define HD        (DIM/HEADS)    // = 64
#define GQA_RATIO 1
#define Q_DIM     (HEADS * HD)    // = 1024 = DIM
#define KV_DIM    (KV_HEADS * HD) // = 1024 = DIM

// Trunk plumbing
#define TRUNK_DIM      4096      // MiMo-V2-Flash hidden size
#define N_TAPS         3         // EAGLE-3: low, mid, high
#define TAP_LAYER_LO   2         // for 32-layer MiMo-V2-Flash
#define TAP_LAYER_MID  16
#define TAP_LAYER_HI   29

// Draft geometry
#define K              3         // draft positions per round
#define K_PAD          16        // pad to satisfy ANE SP minimum;
                                 // empirically SP=8 fails at eval (not compile),
                                 // SP=16 is the floor for SDPA-shaped graphs on
                                 // M3 Ultra ANE. Mask positions >= K causally.
#define NLAYERS        2

// Output head
#define OUT_HIDDEN     4096      // predicted next hidden state dim
#define OUT_CONF       1         // confidence logit (sigmoid CPU-side)
#define OUT_CH         (OUT_HIDDEN + OUT_CONF)  // = 4097

// VOCAB stays 1 — donkey doesn't project to vocab on ANE; trunk's
// lm_head runs MLX-side on the up-projected hidden.
#define VOCAB          1

#define CKPT_PATH         "donkey_v1_ckpt.bin"
#define DEFAULT_DATA_PATH "../trunk_dump.bin"
