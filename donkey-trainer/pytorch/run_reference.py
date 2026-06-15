"""Run the donkey-v1 PyTorch reference.

Loads weights and taps written by Swift's donkey-dump-smoke, runs forward,
writes pred_hidden.bin and confidence.bin for Swift to compare against.

usage: run_reference.py <dump_dir> <output_dir>
"""
import sys, os, time
import numpy as np
import torch
from donkey_v1_ref import (
    DonkeyV1Ref, DIM, HIDDEN, NLAYERS, SP, TRUNK_DIM, TAP_IN_TOTAL, OUT_CH, OUT_HIDDEN
)


def load(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {int(np.prod(shape))} floats, got {arr.size}")
    return torch.from_numpy(arr.reshape(shape))


def main():
    if len(sys.argv) != 3:
        print("usage: run_reference.py <dump_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    dump_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ref] loading from {dump_dir}")
    t0 = time.time()
    model = DonkeyV1Ref().eval()
    with torch.no_grad():
        model.W_tap.copy_(load(f"{dump_dir}/W_tap.bin", (DIM, TAP_IN_TOTAL)))
        for L in range(NLAYERS):
            ly = model.layers[L]
            ly.gamma_att.copy_(load(f"{dump_dir}/layer_{L}_gamma_att.bin", (DIM,)))
            ly.Wq.copy_(load(f"{dump_dir}/layer_{L}_Wq.bin", (DIM, DIM)))
            ly.Wk.copy_(load(f"{dump_dir}/layer_{L}_Wk.bin", (DIM, DIM)))
            ly.Wv.copy_(load(f"{dump_dir}/layer_{L}_Wv.bin", (DIM, DIM)))
            ly.Wo.copy_(load(f"{dump_dir}/layer_{L}_Wo.bin", (DIM, DIM)))
            ly.gamma_ffn.copy_(load(f"{dump_dir}/layer_{L}_gamma_ffn.bin", (DIM,)))
            ly.W_up.copy_(load(f"{dump_dir}/layer_{L}_Wup.bin", (HIDDEN, DIM)))
            ly.W_down.copy_(load(f"{dump_dir}/layer_{L}_Wdown.bin", (DIM, HIDDEN)))
        model.gamma_final.copy_(load(f"{dump_dir}/gamma_final.bin", (DIM,)))
        model.W_head.copy_(load(f"{dump_dir}/W_head.bin", (OUT_CH, DIM)))

    tap_lo  = load(f"{dump_dir}/tap_lo.bin",  (TRUNK_DIM, SP))
    tap_mid = load(f"{dump_dir}/tap_mid.bin", (TRUNK_DIM, SP))
    tap_hi  = load(f"{dump_dir}/tap_hi.bin",  (TRUNK_DIM, SP))
    print(f"[ref] loaded in {time.time()-t0:.1f}s")

    print("[ref] forward ...")
    t0 = time.time()
    with torch.no_grad():
        pred_hidden, confidence = model(tap_lo, tap_mid, tap_hi)
    print(f"[ref] forward done in {(time.time()-t0)*1000:.1f}ms")
    print(f"[ref] pred_hidden[:4, 0] = {pred_hidden[:4, 0].tolist()}")
    print(f"[ref] confidence = {confidence.tolist()}")

    pred_hidden.contiguous().numpy().astype(np.float32).tofile(f"{out_dir}/pred_hidden.bin")
    confidence.contiguous().numpy().astype(np.float32).tofile(f"{out_dir}/confidence.bin")
    print(f"[ref] wrote pred_hidden.bin + confidence.bin to {out_dir}")


if __name__ == "__main__":
    main()
