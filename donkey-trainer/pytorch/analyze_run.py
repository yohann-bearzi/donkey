"""Run-level forensics summary.

usage: analyze_run.py <run_dir>
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    run_dir = Path(sys.argv[1])
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"=== run: {run_dir.name} ===")
    print(f"datasets: {cfg['datasets']}  windows: {cfg['dataset_total']}")
    print(f"steps: {cfg['steps']}  batch: {cfg['batch']}  lr: {cfg['lr']} -> {cfg['lr_final']}")

    # metrics.jsonl summary
    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").open()]
    if not metrics:
        print("[!] no metrics yet"); return
    print(f"\n=== metrics: {len(metrics)} steps logged ===")
    last = metrics[-1]
    first = metrics[0]
    print(f"  step {first['step']}: L={first['loss/total']:.3f} Lp={first['loss/pred']:.3f} "
          f"Lr={first['loss/reg']:.3f} Lce={first['loss/ce']:.3f}")
    print(f"  step {last['step']}: L={last['loss/total']:.3f} Lp={last['loss/pred']:.3f} "
          f"Lr={last['loss/reg']:.3f} Lce={last['loss/ce']:.3f}")
    L_traj = [m["loss/total"] for m in metrics]
    Lp_traj = [m["loss/pred"] for m in metrics]
    Lce_traj = [m["loss/ce"] for m in metrics]
    print(f"  L min: {min(L_traj):.3f} (step {L_traj.index(min(L_traj))})")
    print(f"  Lp min: {min(Lp_traj):.3f} (step {Lp_traj.index(min(Lp_traj))})")
    print(f"  Lce min: {min(Lce_traj):.3f} (step {Lce_traj.index(min(Lce_traj))})")

    # Forensics
    forensics_dir = run_dir / "forensics"
    parquets = sorted(forensics_dir.glob("step_*.parquet"))
    if not parquets:
        print("\n[forensics] no dumps yet"); return
    print(f"\n=== forensics: {len(parquets)} dumps ===")
    for fname in [parquets[0], parquets[len(parquets)//2], parquets[-1]]:
        df = pd.read_parquet(fname)
        per_trace = df.groupby("trace_id").argmax_correct.agg(["sum", "count"])
        per_trace["acc"] = per_trace["sum"] / per_trace["count"]
        print(f"\n  {fname.name}: argmax_correct = {df.argmax_correct.sum()}/{len(df)} = "
              f"{100.0*df.argmax_correct.mean():.1f}%")
        for tid, row in per_trace.iterrows():
            tname = cfg['datasets'][int(tid)] if int(tid) < len(cfg['datasets']) else f"trace{tid}"
            print(f"    {tname:20s} {int(row['sum']):3d}/{int(row['count']):3d} = {100.0*row['acc']:.1f}%")
        print(f"    cos_pred_target: {df.cos_pred_target.min():.3f} .. "
              f"{df.cos_pred_target.max():.3f}  mean: {df.cos_pred_target.mean():.3f}")
        # By K slot
        for k in sorted(df.k_slot.unique()):
            sub = df[df.k_slot == k]
            acc = sub.argmax_correct.mean()
            cos = sub.cos_pred_target.mean()
            print(f"    K={k}  acc={100*acc:5.1f}%  cos={cos:.3f}")


if __name__ == "__main__":
    main()
