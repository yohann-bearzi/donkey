"""Windowed sampler over one or more donkey trace directories.

Each example yields (history[H,W], targets[H,K], target_tokens[K], prompt_id,
trace_id). Windows that span prompt boundaries are excluded.

Multi-trace mode: pass a list of trace dirs. Datasets are concatenated; each
example carries a trace_id so forensics can disaggregate.
"""
import json
from pathlib import Path
from typing import List, Optional, Union, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class _SingleTraceDataset:
    """Internal: one trace directory, optionally filtered to a subset of prompt_ids."""
    def __init__(self, trace_dir, window_size, draft_size, trunk_hidden,
                 allowed_prompt_ids=None):
        self.trace_dir = Path(trace_dir)
        self.W = window_size
        self.K = draft_size
        self.H = trunk_hidden

        meta = json.loads((self.trace_dir / "meta.json").read_text())
        self.T_total = meta["total_positions"]
        assert meta["hidden_dim"] == trunk_hidden, \
            f"{self.trace_dir.name}: hidden_dim mismatch ({meta['hidden_dim']} != {trunk_hidden})"

        self.hidden = np.memmap(
            self.trace_dir / "lastHiddenState.bin",
            dtype=np.float32, mode="r", shape=(self.T_total, self.H))
        self.tokens = np.fromfile(self.trace_dir / "tokens.bin", dtype=np.int32)
        assert len(self.tokens) == self.T_total

        # Optional top-K trunk distribution (for Wasserstein distillation).
        # If present, expose as memmaps shaped [T, K]; consumer code reads
        # them only when needed.
        self.topk_K = None
        self.topk_ids = None
        self.topk_probs = None
        topk_ids_path = self.trace_dir / "trunk_topk_ids.bin"
        topk_probs_path = self.trace_dir / "trunk_topk_probs.bin"
        if topk_ids_path.exists() and topk_probs_path.exists():
            topk_K = meta.get("topk_K")
            if topk_K is not None:
                self.topk_K = int(topk_K)
                self.topk_ids = np.memmap(topk_ids_path, dtype=np.int32, mode="r",
                                          shape=(self.T_total, self.topk_K))
                self.topk_probs = np.memmap(topk_probs_path, dtype=np.float32, mode="r",
                                            shape=(self.T_total, self.topk_K))

        pidx_path = self.trace_dir / "prompt_idx.bin"
        if pidx_path.exists():
            self.prompt_idx = np.fromfile(pidx_path, dtype=np.int32)
            assert len(self.prompt_idx) == self.T_total
        else:
            self.prompt_idx = np.zeros(self.T_total, dtype=np.int32)

        span = self.W + self.K
        if self.T_total >= span:
            head = self.prompt_idx[: self.T_total - span + 1]
            tail = self.prompt_idx[span - 1 :]
            valid_mask = head == tail
            valid_starts = np.flatnonzero(valid_mask).astype(np.int64)
            if allowed_prompt_ids is not None:
                allowed = np.asarray(sorted(allowed_prompt_ids), dtype=np.int32)
                pid_at_start = self.prompt_idx[valid_starts]
                keep = np.isin(pid_at_start, allowed)
                valid_starts = valid_starts[keep]
            self.valid_starts = valid_starts
        else:
            self.valid_starts = np.array([], dtype=np.int64)

    def __len__(self):
        return len(self.valid_starts)

    def get(self, idx):
        t = int(self.valid_starts[idx])
        history = np.ascontiguousarray(self.hidden[t : t + self.W].T)
        targets = np.ascontiguousarray(self.hidden[t + self.W : t + self.W + self.K].T)
        toks    = self.tokens[t + self.W : t + self.W + self.K].astype(np.int64)
        out = {
            "history": torch.from_numpy(history),
            "targets": torch.from_numpy(targets),
            "target_tokens": torch.from_numpy(toks),
            "prompt_id": int(self.prompt_idx[t]),
        }
        # If top-K available, include the trunk distribution at position
        # t + self.W (the FIRST predicted slot, used by stochastic donkey for
        # Wasserstein loss). The K=3 parallel path uses target_topk_ids_K3 below.
        if self.topk_ids is not None:
            # Top-K data for ALL K predicted slots [t+W, t+W+K-1].
            # Shape: [K, K_topk] — K=draft slots, K_topk=trunk top-K dim (256).
            target_positions = slice(t + self.W, t + self.W + self.K)
            out["target_topk_ids"] = torch.from_numpy(
                np.ascontiguousarray(self.topk_ids[target_positions]).astype(np.int64))
            out["target_topk_probs"] = torch.from_numpy(
                np.ascontiguousarray(self.topk_probs[target_positions]).astype(np.float32))
        return out


class DonkeyTraceDataset(Dataset):
    """Single-trace or multi-trace dataset.

    Usage:
        single  : DonkeyTraceDataset("/path/to/trace_dir", W, K)
        multi   : DonkeyTraceDataset(["/p1", "/p2"], W, K)
    """
    def __init__(self,
                 trace_dirs: Union[str, Path, List[Union[str, Path]]],
                 window_size: int,
                 draft_size: int,
                 trunk_hidden: int = 4096,
                 allowed_prompt_ids_per_trace: Optional[List[Optional[set]]] = None):
        if isinstance(trace_dirs, (str, Path)):
            trace_dirs = [trace_dirs]
        self.W = window_size
        self.K = draft_size
        self.H = trunk_hidden

        if allowed_prompt_ids_per_trace is None:
            allowed_prompt_ids_per_trace = [None] * len(trace_dirs)
        assert len(allowed_prompt_ids_per_trace) == len(trace_dirs)

        self._traces = [_SingleTraceDataset(d, window_size, draft_size, trunk_hidden,
                                             allowed_prompt_ids=allowed)
                        for d, allowed in zip(trace_dirs, allowed_prompt_ids_per_trace)]
        self._trace_names = [Path(d).name for d in trace_dirs]
        # cumulative-length index for global -> (trace_idx, local_idx) lookup
        sizes = [len(t) for t in self._traces]
        self._cum = np.cumsum([0] + sizes).astype(np.int64)
        self._total = int(self._cum[-1])

    @property
    def trace_names(self) -> List[str]:
        return list(self._trace_names)

    @property
    def trace_sizes(self) -> List[int]:
        return [len(t) for t in self._traces]

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        trace_idx = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local_idx = idx - int(self._cum[trace_idx])
        ex = self._traces[trace_idx].get(local_idx)
        ex["trace_id"] = trace_idx
        return ex





def _split_by_prompt_id(trace_dir: Union[str, Path],
                        fractions: tuple = (0.9, 0.05, 0.05),
                        seed: int = 42) -> tuple:
    """Deterministically split a trace's prompt_ids into (train, val, verify) sets.

    Returns three sets of prompt_id integers. The split is by *prompt*, not by
    *window* — this guarantees no within-prompt leakage between splits, which
    matters for honest validation: a model that memorizes window k from prompt P
    can trivially predict adjacent window k+1 from P, so they must go in the
    same split.

    Args:
        trace_dir: directory with meta.json + prompt_idx.bin
        fractions: (train, val, verify); must sum to 1.0
        seed:     RNG seed for the prompt-id shuffle

    Returns:
        (train_ids, val_ids, verify_ids) — three sets of int prompt_ids
    """
    import json
    trace_dir = Path(trace_dir)
    pidx = np.fromfile(trace_dir / "prompt_idx.bin", dtype=np.int32)
    unique = np.unique(pidx)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n = len(shuffled)
    n_train  = int(round(n * fractions[0]))
    n_val    = int(round(n * fractions[1]))
    train_ids  = set(int(x) for x in shuffled[:n_train])
    val_ids    = set(int(x) for x in shuffled[n_train:n_train + n_val])
    verify_ids = set(int(x) for x in shuffled[n_train + n_val:])
    return train_ids, val_ids, verify_ids


def split_datasets(trace_dirs: List[Union[str, Path]],
                   window_size: int, draft_size: int,
                   trunk_hidden: int = 4096,
                   fractions: tuple = (0.9, 0.05, 0.05),
                   seed: int = 42) -> tuple:
    """Build (train_ds, val_ds, verify_ds) datasets from multiple trace dirs.

    Each trace is split independently by prompt_id with the same seed +
    fractions, then the three resulting subsets are concatenated.
    """
    train_ids_per, val_ids_per, verify_ids_per = [], [], []
    for d in trace_dirs:
        tr, va, ve = _split_by_prompt_id(d, fractions, seed)
        train_ids_per.append(tr); val_ids_per.append(va); verify_ids_per.append(ve)

    train_ds  = DonkeyTraceDataset(trace_dirs, window_size, draft_size, trunk_hidden,
                                   allowed_prompt_ids_per_trace=train_ids_per)
    val_ds    = DonkeyTraceDataset(trace_dirs, window_size, draft_size, trunk_hidden,
                                   allowed_prompt_ids_per_trace=val_ids_per)
    verify_ds = DonkeyTraceDataset(trace_dirs, window_size, draft_size, trunk_hidden,
                                   allowed_prompt_ids_per_trace=verify_ids_per)
    return train_ds, val_ds, verify_ds

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: trace_dataset.py <trace_dir> [<trace_dir2> ...]")
        sys.exit(1)
    dirs = sys.argv[1:]
    ds = DonkeyTraceDataset(dirs, window_size=13, draft_size=3)
    print(f"[ds] {len(ds)} valid windows from {len(dirs)} trace(s)")
    for name, n in zip(ds.trace_names, ds.trace_sizes):
        pct = 100.0 * n / max(1, len(ds))
        print(f"[ds]   {name:30s} {n:>7d} windows  ({pct:5.1f}%)")
    ex = ds[0]
    print(f"[ds] example: history={tuple(ex['history'].shape)} targets={tuple(ex['targets'].shape)} "
          f"tokens={ex['target_tokens'].tolist()} prompt_id={ex['prompt_id']} trace_id={ex['trace_id']}")
    import random
    for i in random.sample(range(len(ds)), min(20, len(ds))):
        _ = ds[i]
    print("[ds] OK")
