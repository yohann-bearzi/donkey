"""Test donkey_config.py against the canonical spec and three negative cases.

Mirrors ConfigSmoke.swift one-to-one so Swift and Python enforce identical
invariants and reject identical bad inputs.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donkey_config import DonkeyConfig, DonkeyConfigError


def _base_good() -> dict:
    return json.loads(Path(__file__).resolve().parents[2].joinpath(
        "spec/donkey_v2_default.json").read_text())


def test_load_canonical_spec():
    cfg = DonkeyConfig.from_dict(_base_good())
    assert cfg.schema_version == 1
    assert cfg.model_name == "donkey_v2_default"
    assert cfg.trunk.hidden_dim == 4096
    assert cfg.architecture.hidden_dim == 1024
    assert cfg.architecture.heads == 16
    assert cfg.architecture.head_dim == 64
    assert cfg.architecture.hidden_dim == cfg.architecture.heads * cfg.architecture.head_dim
    assert cfg.sequence.window_size + cfg.sequence.draft_size <= cfg.sequence.spatial_pad
    # Derived sizes. out_ch projects back to trunk hidden so verify through
    # trunk's lm_head works: trunk.hidden_dim + out_conf_dim, NOT
    # architecture.hidden_dim + out_conf_dim (which would be the buggy version).
    expected_out_ch = cfg.trunk.hidden_dim + cfg.architecture.out_conf_dim
    assert cfg.out_ch == expected_out_ch, f"out_ch={cfg.out_ch} expected {expected_out_ch}"
    assert cfg.qkv_ch == 3 * cfg.architecture.hidden_dim
    print(f"  loaded {cfg.model_name}: hidden={cfg.architecture.hidden_dim} W={cfg.sequence.window_size} K={cfg.sequence.draft_size} SP={cfg.sequence.spatial_pad}")


def test_bad_invariant_hidden_dim():
    d = _base_good()
    d["architecture"]["heads"] = 8  # 8 * 64 = 512 != 1024
    try:
        DonkeyConfig.from_dict(d)
    except DonkeyConfigError as e:
        print(f"  (expected) rejected: {e}")
        return
    raise AssertionError("should have rejected heads*head_dim != hidden_dim")


def test_bad_spatial_pad():
    d = _base_good()
    d["sequence"]["spatial_pad"] = 24  # not in {16,32,64,128}
    try:
        DonkeyConfig.from_dict(d)
    except DonkeyConfigError as e:
        print(f"  (expected) rejected: {e}")
        return
    raise AssertionError("should have rejected SP=24")


def test_bad_schema_version():
    d = _base_good()
    d["schema_version"] = 999
    try:
        DonkeyConfig.from_dict(d)
    except DonkeyConfigError as e:
        print(f"  (expected) rejected: {e}")
        return
    raise AssertionError("should have rejected schema_version=999")


def run_all():
    for name, fn in [
        ("load canonical spec",        test_load_canonical_spec),
        ("invariant: heads*head_dim",  test_bad_invariant_hidden_dim),
        ("unsupported SP value",       test_bad_spatial_pad),
        ("schema_version mismatch",    test_bad_schema_version),
    ]:
        print(f"[python config] {name}")
        fn()
    print("[python config] all checks passed")


if __name__ == "__main__":
    run_all()
