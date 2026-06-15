import numpy as np, sys

ckpt = sys.argv[1] if len(sys.argv) > 1 else \
    f"{__import__('os').path.expanduser('~')}/projects/donkey/ckpts/v4l2_mbpp_codealpaca_d64_dp256.npz"
ck = np.load(ckpt)
print(f"Inspecting: {ckpt}")
print(f"__epoch__: {int(ck['__epoch__']) if '__epoch__' in ck.files else 'n/a'}\n")

def eff_rank(s):
    # participation ratio of singular values: (sum s)^2 / sum(s^2). =N if flat, <<N if peaked.
    s = s[s > 0]
    pr = (s.sum() ** 2) / (s ** 2).sum()
    # also count above 1% and 5% of max
    n1 = int((s > 0.01 * s.max()).sum())
    n5 = int((s > 0.05 * s.max()).sum())
    return pr, n1, n5

wkeys = [k for k in ck.files if not k.startswith("__")]
# only 2D weight matrices (skip biases, norms, embeddings handled separately)
mats = [(k, ck[k]) for k in wkeys if ck[k].ndim == 2]

print(f"{'weight':<34}{'shape':<14}{'eff_rank':>9}{'>1%':>6}{'>5%':>6}{'mindim':>8}")
print("-" * 80)
latent_boundary = []
for k, W in sorted(mats):
    m, n = W.shape
    mn = min(m, n)
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    pr, n1, n5 = eff_rank(s)
    flag = ""
    # flag the latent-boundary matrices (one dim == 64, the d latent)
    if 64 in (m, n):
        flag = "  <-- LATENT BOUNDARY"
        latent_boundary.append((k, pr, n1, mn))
    print(f"{k:<34}{str(W.shape):<14}{pr:>9.1f}{n1:>6}{n5:>6}{mn:>8}{flag}")

print("\n" + "=" * 80)
print("LATENT-BOUNDARY SUMMARY (the d=128 question):")
print("  d = 64. If eff_rank ~= 64 (saturated) -> latent fully used -> d=128 may help.")
print("  If eff_rank << 64 -> latent under-utilized -> capacity NOT the bottleneck -> skip d=128.\n")
for k, pr, n1, mn in latent_boundary:
    util = pr / 64.0
    verdict = "SATURATED (d=128 may help)" if util > 0.85 else \
              "under-utilized (d=128 unlikely to help)" if util < 0.65 else "moderate"
    print(f"  {k:<34} eff_rank={pr:5.1f}/64  (util {util:4.0%})  >1%={n1}  -> {verdict}")
print("\n  -> if the latent boundary matrices are under-utilized, the L2 fix already used the")
print("     capacity efficiently and d=128 won't add much. If saturated, d=128 is worth trying.")
