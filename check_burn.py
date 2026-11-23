"""Is the radiation-era run-in long enough?

Runs the same random initial states with several burn-in lengths and compares
(1) the three pass rates, against their binomial errors, and
(2) the distribution of Omega_phi at the end of the burn-in, by a two-sample
    Kolmogorov-Smirnov test between successive burn-in lengths.
If the longest two agree, the shorter of them is long enough.

Usage: python3 check_burn.py --l1 10 --alpha 0.24 --n 200000
"""
import argparse
import json
import math

import numpy as np
from scipy.stats import ks_2samp

import gpu_ensemble as g

ap = argparse.ArgumentParser()
ap.add_argument("--l1", type=float, default=10.0)
ap.add_argument("--l2", type=float, default=0.01)
ap.add_argument("--alpha", type=float, default=0.24)
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--dN", type=float, default=2e-3)
ap.add_argument("--burns", default="150,300,600,1200")
ap.add_argument("--device", default="cuda" if g.HAVE_TORCH and g.torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="burn_check.json")
a = ap.parse_args()
if not g.HAVE_TORCH:
    a.device = "numpy"

rows, prev = [], None
for burn in [float(b) for b in a.burns.split(",")]:
    o, start, _, _ = g.run(a.l1, a.l2, a.alpha, a.n, a.dN, burn, a.device, seed=11, keep_states=True)
    res, _ = g.summarise(o)
    O_start = start[0] ** 2 + start[1] ** 2
    O_start = O_start[np.isfinite(O_start)]
    row = dict(burn=burn, n_valid=res["n_valid"],
               rates={k: res[k]["rate"] for k in ("bbn", "recomb", "today", "all_three")},
               se={k: res[k]["binomial_se"] for k in ("bbn", "recomb", "today", "all_three")})
    line = (f"burn {burn:5.0f}: BBN {100*row['rates']['bbn']:.2f}%  recomb {100*row['rates']['recomb']:.2f}%  "
            f"today {100*row['rates']['today']:.2f}%  all three {row['rates']['all_three']:.2e}")
    if prev is not None:
        ks = ks_2samp(prev[1], O_start)
        z = {k: (row["rates"][k] - prev[0]["rates"][k]) /
             max(math.hypot(row["se"][k], prev[0]["se"][k]), 1e-300) for k in ("bbn", "recomb", "today")}
        row["vs_previous"] = dict(ks_stat=float(ks.statistic), ks_p=float(ks.pvalue), z=z)
        line += (f" | vs previous: KS p = {ks.pvalue:.3f}, rate shifts in sigma: "
                 + ", ".join(f"{k} {v:+.1f}" for k, v in z.items()))
    print(line, flush=True)
    rows.append(row)
    prev = (row, O_start)
json.dump(dict(l1=a.l1, l2=a.l2, alpha=a.alpha, n=a.n, rows=rows), open(a.out, "w"), indent=1)
print("For chaotic parameter sets the rows are effectively independent samples, so if the burn-in is long\n"
      "enough the shifts should scatter like a unit normal and the KS p-values should be unremarkable.\n"
      "A steady drift with burn-in length means the ensemble has not reached the stationary distribution.")
