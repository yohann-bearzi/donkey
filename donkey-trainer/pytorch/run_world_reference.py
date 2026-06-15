"""Phase 4.5: Load weights dumped by Swift's world-dump-smoke, run the
PyTorch DonkeyWorldRef on the same cold-start hidden, write predictions
for cosine comparison.

usage: run_world_reference.py <dump_dir>
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donkey_config import DonkeyConfig
from donkey_world import DonkeyWorldRef, fill_history_cold_start


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "spec" / "donkey_v2_default.json"


def load_bin(path: Path, shape: tuple) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: expected {int(np.prod(shape))} floats, got {arr.size}")
    return torch.from_numpy(arr.reshape(shape).copy())


def load_weights_from_dump(model: DonkeyWorldRef, cfg: DonkeyConfig, dump_dir: Path):
    """Copy Swift-dumped bins into PyTorch model parameters.

    Swift writes weights row-major [out, in]. PyTorch nn.Parameters in our
    DonkeyWorldRef are stored as [out, in] (same convention). No transposition.
    """
    D  = cfg.architecture.hidden_dim
    F_ = cfg.architecture.ffn_dim
    TH = cfg.trunk.hidden_dim
    K  = cfg.sequence.draft_size
    SP = cfg.sequence.spatial_pad
    OUT = cfg.out_ch

    with torch.no_grad():
        model.input_proj.W.copy_(load_bin(dump_dir / "inputProj.bin", (D, TH)))
        model.input_proj.ln_gamma.copy_(load_bin(dump_dir / "inputProjLnGamma.bin", (D,)))
        model.input_proj.ln_beta.copy_(load_bin(dump_dir / "inputProjLnBeta.bin", (D,)))
        model.draft_queries.copy_(load_bin(dump_dir / "draftQueries.bin", (D, K)))
        model.pos_bias.copy_(load_bin(dump_dir / "positionalBias.bin", (D, SP)))
        for l in range(cfg.architecture.n_layers):
            ly = model.layers[l]
            ly.gamma_att.copy_(load_bin(dump_dir / f"layer_{l}_gammaAtt.bin", (D,)))
            ly.attn.Wq.copy_(load_bin(dump_dir / f"layer_{l}_Wq.bin", (D, D)))
            ly.attn.Wk.copy_(load_bin(dump_dir / f"layer_{l}_Wk.bin", (D, D)))
            ly.attn.Wv.copy_(load_bin(dump_dir / f"layer_{l}_Wv.bin", (D, D)))
            ly.attn.Wo.copy_(load_bin(dump_dir / f"layer_{l}_Wo.bin", (D, D)))
            ly.gamma_ffn.copy_(load_bin(dump_dir / f"layer_{l}_gammaFfn.bin", (D,)))
            ly.ffn.W_up.copy_(load_bin(dump_dir / f"layer_{l}_Wup.bin", (F_, D)))
            ly.ffn.W_down.copy_(load_bin(dump_dir / f"layer_{l}_Wdown.bin", (D, F_)))
        model.gamma_final.copy_(load_bin(dump_dir / "gammaFinal.bin", (D,)))
        model.W_head.copy_(load_bin(dump_dir / "head.bin", (OUT, D)))


def main():
    if len(sys.argv) != 2:
        print("usage: run_world_reference.py <dump_dir>", file=sys.stderr)
        sys.exit(1)
    dump_dir = Path(sys.argv[1])

    print(f"[world-ref] loading config from {SPEC}")
    cfg = DonkeyConfig.from_json(SPEC)

    print("[world-ref] building DonkeyWorldRef")
    model = DonkeyWorldRef(cfg).eval()

    print(f"[world-ref] loading Swift-dumped weights from {dump_dir}")
    t0 = time.time()
    load_weights_from_dump(model, cfg, dump_dir)
    print(f"[world-ref] loaded in {time.time()-t0:.1f}s")

    h_first = load_bin(dump_dir / "h_first.bin", (cfg.trunk.hidden_dim,))
    history = fill_history_cold_start(h_first, [], cfg.sequence.window_size)

    print("[world-ref] running forward")
    t0 = time.time()
    with torch.no_grad():
        pred_hidden, confidence = model(history)
    print(f"[world-ref] forward done in {(time.time()-t0)*1000:.1f}ms")
    print(f"[world-ref]   pred_hidden[:4, 0] = {pred_hidden[:4, 0].tolist()}")
    print(f"[world-ref]   confidence         = {confidence.tolist()}")

    # Save as channel-major [TH, K] to match Swift's layout, then flatten.
    pred_path = dump_dir / "python_pred_hidden.bin"
    conf_path = dump_dir / "python_confidence.bin"
    pred_hidden.contiguous().numpy().astype(np.float32).tofile(pred_path)
    confidence.contiguous().numpy().astype(np.float32).tofile(conf_path)
    print(f"[world-ref] wrote {pred_path.name} ({pred_hidden.numel() * 4} bytes)")
    print(f"[world-ref] wrote {conf_path.name} ({confidence.numel() * 4} bytes)")


if __name__ == "__main__":
    main()
