"""Phase 4.5 test: Swift world-dump-smoke output vs PyTorch DonkeyWorldRef
output, same random weights, same cold-start hidden. Asserts cosine >= 0.99.
"""
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 2:
    print("usage: test_world_cosine.py <dump_dir>", file=sys.stderr)
    sys.exit(1)
d = Path(sys.argv[1])

swift_ph = np.fromfile(d / "swift_pred_hidden.bin",  dtype=np.float32)
py_ph    = np.fromfile(d / "python_pred_hidden.bin", dtype=np.float32)
swift_cf = np.fromfile(d / "swift_confidence.bin",   dtype=np.float32)
py_cf    = np.fromfile(d / "python_confidence.bin",  dtype=np.float32)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

c_ph = cos(swift_ph, py_ph)
c_cf = cos(swift_cf, py_cf)
print(f"  pred_hidden cosine: {c_ph:.7f}  (need >= 0.99)")
print(f"  confidence  cosine: {c_cf:.7f}  (need >= 0.999)")

ok = (c_ph >= 0.99) and (c_cf >= 0.999)
if ok:
    print("[world cosine] PASS")
    sys.exit(0)
else:
    print("[world cosine] FAIL")
    sys.exit(1)
