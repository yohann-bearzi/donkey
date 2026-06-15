#!/usr/bin/env python3
"""Shape sweep driver for donkey v3. Two stages:
  stage 1 (ceiling): short AE-ish runs at each d to see the reconstruction cap (lce-heavy, few epochs)
  stage 2 (predictor): full v3 joint at promising (d,dp), in-corpus val on big corpus, capped transitions
Parses each run's best val accept and tabulates. Usage:
  python3 sweep.py mbpp --stage all --max-transitions 300000
"""
import argparse, subprocess, re, os, sys, time, json

CKPTS = os.environ.get("DONKEY_CKPTS", os.path.expanduser("~/projects/donkey/ckpts"))
ENV = dict(os.environ)

def run(corpus, d, dp, epochs, max_tr, val, patience, tag):
    out = os.path.join(CKPTS, f"sweep_{tag}_{corpus}_d{d}_dp{dp}.npz")
    cmd = [sys.executable, "donkey_train_v3.py", corpus, "--d", str(d), "--dp", str(dp),
           "--epochs", str(epochs), "--patience", str(patience), "--out", out,
           "--max-transitions", str(max_tr)]
    if val: cmd += ["--val", val]
    print(f"\n{'='*70}\n[sweep] d={d} dp={dp} epochs={epochs} cap={max_tr} val={val or 'in-corpus'}\n{'='*70}")
    t0 = time.time()
    best = None; gap = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in proc.stdout:
        sys.stdout.write(line)
        m = re.search(r"val\([^)]+\)\s+([\d.]+)%\s*\*", line)
        if m: best = float(m.group(1))
        g = re.search(r"gap\s+([+\-][\d.]+)", line)
        if g: gap = float(g.group(1))
    proc.wait()
    return {"d": d, "dp": dp, "best_val": best, "final_gap": gap, "secs": time.time()-t0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--stage", choices=["ceiling","predictor","all"], default="all")
    ap.add_argument("--max-transitions", type=int, default=300000)
    ap.add_argument("--val", type=str, default=None, help="cross-corpus val; omit for in-corpus")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--ds", type=str, default="64,128,256")
    ap.add_argument("--dps", type=str, default="256,384,512")
    a = ap.parse_args()
    ds = [int(x) for x in a.ds.split(",")]; dps = [int(x) for x in a.dps.split(",")]
    results = []

    if a.stage in ("ceiling","all"):
        # ceiling proxy: d-sweep at fixed small dp, the val accept ceiling at each d
        # (full joint but short; the plateau approximates achievable-at-d). dp fixed=256.
        print("\n##### STAGE 1: ceiling vs d (dp=256) #####")
        for d in ds:
            results.append({**run(a.corpus, d, 256, a.epochs, a.max_transitions, a.val, a.patience, "ceil"), "stage":"ceiling"})

    if a.stage in ("predictor","all"):
        print("\n##### STAGE 2: predictor sweep (d x dp) #####")
        for d in ds:
            for dp in dps:
                if dp == 256 and a.stage == "all":
                    continue  # already ran dp=256 in stage 1
                results.append({**run(a.corpus, d, dp, a.epochs, a.max_transitions, a.val, a.patience, "pred"), "stage":"predictor"})

    print("\n\n##### SWEEP RESULTS #####")
    print(f"{'stage':<10}{'d':>5}{'dp':>6}{'best_val%':>11}{'gap':>8}{'min':>7}")
    for r in sorted(results, key=lambda x: -(x['best_val'] or 0)):
        bv = f"{r['best_val']:.2f}" if r['best_val'] else "—"
        gp = f"{r['final_gap']:+.1f}" if r.get('final_gap') is not None else "—"
        print(f"{r['stage']:<10}{r['d']:>5}{r['dp']:>6}{bv:>11}{gp:>8}{r['secs']/60:>7.1f}")
    json.dump(results, open(os.path.join(CKPTS,"sweep_results.json"),"w"), indent=2)
    print(f"\nsaved -> {os.path.join(CKPTS,'sweep_results.json')}")
    if results:
        winner = max(results, key=lambda x: x['best_val'] or 0)
        print(f"\nWINNER: d={winner['d']} dp={winner['dp']} @ {winner['best_val']:.2f}% val (gap {winner.get('final_gap')})")
        print("Confirm on FULL data: python3 donkey_train_v3.py", a.corpus,
              f"--d {winner['d']} --dp {winner['dp']} --epochs 300", f"--val {a.val}" if a.val else "(in-corpus)")

if __name__ == "__main__": main()
