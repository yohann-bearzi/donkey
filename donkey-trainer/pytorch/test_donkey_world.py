"""Phase 3 test: DonkeyWorldRef instantiates from DonkeyConfig and
produces shape-correct, finite, deterministic output.

Also writes a golden checksum on first run; subsequent runs compare against it.
"""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donkey_config import DonkeyConfig
from donkey_world import DonkeyWorldRef, fill_history_cold_start


# === paths ===
REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "spec" / "donkey_v2_default.json"
GOLDEN_DIR = REPO / "tests" / "golden"
GOLDEN_FILE = GOLDEN_DIR / "donkey_world_v2_default.json"


def init_weights_deterministic(model: torch.nn.Module, seed: int = 42):
    """Fill all parameters with seeded random fp32. Mirrors what the Swift
    smoke does so PyTorch and Swift can later be compared with bit-identical
    weights."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            # Tiny init scaled by 1/sqrt(in_dim) for matrices, 0.1 for gammas.
            if p.ndim == 2:
                scale = 1.0 / (p.shape[1] ** 0.5)
                p.copy_(torch.randn(p.shape, generator=g) * scale)
            elif "gamma" in name.lower():
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))


def test_instantiate_and_forward():
    """Build model from canonical spec, run forward, assert shapes + finite."""
    cfg = DonkeyConfig.from_json(SPEC)
    model = DonkeyWorldRef(cfg).eval()
    init_weights_deterministic(model, seed=42)

    # Synthesize a cold-start history: a single h_first, repeated.
    g = torch.Generator().manual_seed(123)
    h_first = 0.5 * torch.randn(cfg.trunk.hidden_dim, generator=g)
    history = fill_history_cold_start(h_first, [], cfg.sequence.window_size)
    assert history.shape == (cfg.trunk.hidden_dim, cfg.sequence.window_size), \
        f"history shape: {history.shape}"

    with torch.no_grad():
        pred_hidden, confidence = model(history)
    assert pred_hidden.shape == (cfg.trunk.hidden_dim, cfg.sequence.draft_size), \
        f"pred_hidden shape: {pred_hidden.shape}"
    assert confidence.shape == (cfg.sequence.draft_size,), \
        f"confidence shape: {confidence.shape}"
    assert torch.isfinite(pred_hidden).all(), "pred_hidden has NaN/Inf"
    assert torch.isfinite(confidence).all(), "confidence has NaN/Inf"
    assert (confidence >= 0).all() and (confidence <= 1).all(), \
        f"confidence out of [0,1]: min={confidence.min()} max={confidence.max()}"
    print(f"  pred_hidden shape={tuple(pred_hidden.shape)}  dtype={pred_hidden.dtype}")
    print(f"  confidence       ={[f'{c:.4f}' for c in confidence.tolist()]}")
    return pred_hidden, confidence


def test_determinism():
    """Same seed twice must give bit-identical output."""
    cfg = DonkeyConfig.from_json(SPEC)
    out_a, conf_a = None, None
    for run in range(2):
        model = DonkeyWorldRef(cfg).eval()
        init_weights_deterministic(model, seed=42)
        g = torch.Generator().manual_seed(123)
        h_first = 0.5 * torch.randn(cfg.trunk.hidden_dim, generator=g)
        history = fill_history_cold_start(h_first, [], cfg.sequence.window_size)
        with torch.no_grad():
            out, conf = model(history)
        if run == 0:
            out_a, conf_a = out, conf
        else:
            assert torch.equal(out_a, out), "non-deterministic pred_hidden"
            assert torch.equal(conf_a, conf), "non-deterministic confidence"
    print("  determinism: pred_hidden + confidence bit-identical across 2 runs")


def _checksum(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().astype(np.float32).tobytes()).hexdigest()


def test_golden_output():
    """Frozen checksum of output for canonical (config, weights, history).

    First run writes the golden file; subsequent runs compare.
    """
    cfg = DonkeyConfig.from_json(SPEC)
    model = DonkeyWorldRef(cfg).eval()
    init_weights_deterministic(model, seed=42)
    g = torch.Generator().manual_seed(123)
    h_first = 0.5 * torch.randn(cfg.trunk.hidden_dim, generator=g)
    history = fill_history_cold_start(h_first, [], cfg.sequence.window_size)
    with torch.no_grad():
        pred_hidden, confidence = model(history)

    ph_hash = _checksum(pred_hidden)
    cf_hash = _checksum(confidence)
    summary = {
        "pred_hidden_sha256": ph_hash,
        "confidence_sha256": cf_hash,
        "pred_hidden_first4": pred_hidden[:4, 0].tolist(),
        "confidence": confidence.tolist(),
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.write_text(json.dumps(summary, indent=2))
        print(f"  BOOTSTRAP: wrote golden file {GOLDEN_FILE.name}")
        print(f"  pred_hidden sha256: {ph_hash[:16]}...")
        print(f"  confidence  sha256: {cf_hash[:16]}...")
        print("  (next run will compare against this.)")
        return

    golden = json.loads(GOLDEN_FILE.read_text())
    if golden["pred_hidden_sha256"] != ph_hash:
        raise AssertionError(
            f"pred_hidden changed!\n"
            f"  expected: {golden['pred_hidden_sha256'][:32]}\n"
            f"  got:      {ph_hash[:32]}\n"
            f"  golden first4: {golden['pred_hidden_first4']}\n"
            f"  got first4:    {pred_hidden[:4, 0].tolist()}")
    if golden["confidence_sha256"] != cf_hash:
        raise AssertionError(
            f"confidence changed!\n"
            f"  expected: {golden['confidence_sha256'][:32]}\n"
            f"  got:      {cf_hash[:32]}")
    print(f"  pred_hidden  matches golden ({ph_hash[:16]}...)")
    print(f"  confidence   matches golden ({cf_hash[:16]}...)")


def run_all():
    for name, fn in [
        ("instantiate + forward + shape check", test_instantiate_and_forward),
        ("determinism (same seed -> same out)", test_determinism),
        ("golden output (frozen sha256)", test_golden_output),
    ]:
        print(f"[3.{name[:30]}] ")
        fn()
    print("[python world v2] all checks passed")


if __name__ == "__main__":
    run_all()
