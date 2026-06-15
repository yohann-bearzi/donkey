"""Training forensics: metrics.jsonl always-on, parquet sidecar when DONKEY_DEBUG=1."""
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch


def _scalar(v):
    if isinstance(v, torch.Tensor):
        return round(float(v.item()), 6) if v.numel() == 1 else v.detach().tolist()
    if isinstance(v, float):
        return round(v, 6)
    return v


class MetricsLogger:
    def __init__(self, log_path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        self.start = time.monotonic()

    def write(self, step, metrics):
        rec = {"step": step, "wall_s": round(time.monotonic() - self.start, 3)}
        rec.update({k: _scalar(v) for k, v in metrics.items()})
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


class ExampleRecord:
    """Helper for assembling one row of per-example forensics."""
    __slots__ = ("step", "ex_in_batch", "trace_id", "prompt_id",
                 "target_tokens", "argmax_tokens", "top5_tokens", "top5_probs",
                 "L_pred", "L_ce", "cos_pred_target", "z_pred_norm", "z_target_norm")
    def __init__(self, **kw):
        for slot in self.__slots__: setattr(self, slot, kw.get(slot))
    def as_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


class ForensicsDumper:
    """Write per-example forensics to parquet files.

    Each dump file has B*K rows (one per draft slot per example in the batch).
    """
    def __init__(self, dump_dir, dump_every: int = 100):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.dump_every = dump_every

    def should_dump(self, step):
        return self.dump_every > 0 and (step + 1) % self.dump_every == 0

    def collect_and_dump(self, step, batch, z_pred, z_target, pred_decoded,
                         lm_head, losses_per_ex, mode="parquet"):
        """Compute per-example records and write to disk.

        Args:
            batch:        DataLoader batch dict, on device
            z_pred:       [B, D, K]
            z_target:     [B, D, K]
            pred_decoded: [B, H, K]
            lm_head:      [V, H] frozen
            losses_per_ex: optional dict with per-example L_pred, L_ce, cos
        """
        import torch
        B = z_pred.shape[0]; K = z_pred.shape[2]
        V = lm_head.shape[0]
        target_tokens = batch["target_tokens"]
        trace_ids = batch["trace_id"]; prompt_ids = batch["prompt_id"]

        # Compute argmax + top5 of pred_decoded through lm_head, per slot.
        with torch.no_grad():
            # pred_decoded [B, H, K] -> [B, K, V] logits
            pf = pred_decoded.permute(0, 2, 1).float()         # [B, K, H]
            logits = pf @ lm_head.float().t()                  # [B, K, V]
            top5_vals, top5_idx = logits.topk(5, dim=-1)       # [B, K, 5]
            top5_prob = torch.softmax(top5_vals, dim=-1)
            argmax = logits.argmax(dim=-1)                     # [B, K]

            # Cosine z_pred vs z_target, per slot
            zp = z_pred.detach().permute(0, 2, 1)              # [B, K, D]
            zt = z_target.detach().permute(0, 2, 1)            # [B, K, D]
            cos = torch.nn.functional.cosine_similarity(zp, zt, dim=-1)  # [B, K]
            zp_norm = zp.norm(dim=-1)
            zt_norm = zt.norm(dim=-1)

        records = []
        for b in range(B):
            for k in range(K):
                records.append({
                    "step": step,
                    "ex_in_batch": b,
                    "k_slot": k,
                    "trace_id": int(trace_ids[b].item() if torch.is_tensor(trace_ids) else trace_ids[b]),
                    "prompt_id": int(prompt_ids[b].item() if torch.is_tensor(prompt_ids) else prompt_ids[b]),
                    "target_token":  int(target_tokens[b, k].item()),
                    "argmax_token":  int(argmax[b, k].item()),
                    "argmax_correct": bool(int(argmax[b, k].item()) == int(target_tokens[b, k].item())),
                    "top5_tokens": [int(x) for x in top5_idx[b, k].tolist()],
                    "top5_probs":  [float(x) for x in top5_prob[b, k].tolist()],
                    "cos_pred_target": float(cos[b, k].item()),
                    "z_pred_norm":    float(zp_norm[b, k].item()),
                    "z_target_norm":  float(zt_norm[b, k].item()),
                })

        if mode == "parquet":
            try:
                import pandas as pd
                path = self.dump_dir / f"step_{step:06d}.parquet"
                pd.DataFrame(records).to_parquet(path)
                return path
            except ImportError:
                pass
        path = self.dump_dir / f"step_{step:06d}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path


class AdaptiveHuberDelta:
    """Running p95 of |pred - target| → used as Huber delta + as a metric."""
    def __init__(self, window: int = 1000):
        self.values: deque = deque(maxlen=window)
        self.current_delta: float = 1.0

    def update(self, abs_errs: torch.Tensor) -> float:
        flat = abs_errs.detach().flatten()
        n = min(flat.numel(), 256)
        if flat.numel() > n:
            idx = torch.randint(0, flat.numel(), (n,), device=flat.device)
            flat = flat[idx]
        self.values.extend(flat.cpu().tolist())
        if len(self.values) >= 10:
            self.current_delta = float(torch.tensor(list(self.values)).quantile(0.95))
        return self.current_delta


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    ml = MetricsLogger(tmp / "m.jsonl")
    for s in range(3):
        ml.write(s, {"loss/total": torch.tensor(1.0 - s * 0.1), "lr": 3e-4})
    ml.close()
    print(f"[forensics] wrote {tmp/'m.jsonl'}:")
    for line in open(tmp / "m.jsonl"):
        print(f"  {line.rstrip()}")
    fd = ForensicsDumper(tmp / "f", dump_every=1)
    p = fd.dump(0, [{"step": 0, "ex": 0, "cos": 0.9},
                    {"step": 0, "ex": 1, "cos": 0.7}])
    print(f"[forensics] dump: {p}")
    hd = AdaptiveHuberDelta()
    for _ in range(50):
        hd.update(torch.rand(100) * 5.0)
    print(f"[forensics] huber delta: {hd.current_delta:.3f}  (expect ~4.7 for U[0,5] p95)")
    print("[forensics] OK")
