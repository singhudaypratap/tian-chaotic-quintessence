"""Tian's updated parameter set (Part IV: l1 = 10, l2 = 0.01, alpha = 0.24,
chaotic already in the radiation era).

(a) Largest Lyapunov exponent in radiation and matter backgrounds.
(b) Numerical predictability: the same history at three tolerances.
(c) Ensemble over N_eq: how many histories pass nucleosynthesis
    (max Omega_phi < 0.04 for 1e8 < z < 1e10, Tian's own criterion) and match
    today's dark energy (0.66 < Omega_phi < 0.72, w_phi < -0.9), and what
    Omega_phi they have at recombination and over 50 < z < 5000.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import tian as t

L1, L2, AL = 10.0, 0.01, 0.24
LAM0 = L2 + 2.0


def one(args):
    N_eq, tol = args
    h = t.history(L1, L2, AL, N_eq, lam0=LAM0, rtol=tol, atol=tol * 1e-3, n_out=8000)
    z, O = h["z"], h["Ophi"]
    def mx(lo, hi):
        s = (z > lo) & (z < hi)
        return float(O[s].max())
    i_rec = int(np.argmin(np.abs(z - 1100.0)))
    return dict(N_eq=N_eq, Ophi0=float(O[-1]), w0=float(h["wphi"][-1]),
                Ophi_rec=float(O[i_rec]), bbn_max=mx(1e8, 1e10), cmb_max=mx(50, 5000),
                cmb_min=float(O[(z > 50) & (z < 5000)].min()), mat_max=mx(2, 50))


if __name__ == "__main__":
    out = {}
    out["lyap"] = []
    for w in (1 / 3, 0.0):
        l, s_end, hist = t.lyap1(np.array([0.3, 0.3, 0.5]), (L1, L2, AL, w), T=1500.0)
        k = len(hist)
        vals = [float(hist[int(k * f) - 1]) for f in (0.25, 0.5, 1.0)]
        out["lyap"].append(dict(w=w, lyap=vals))
        print(f"(a) w={w:.3f}: lambda_1 at T=375,750,1500: " + " ".join(f"{v:.4f}" for v in vals), flush=True)
    out["tolerance"] = []
    for N_eq in (80.0, 100.0):
        vals = [one((N_eq, tol))["Ophi0"] for tol in (1e-9, 1e-11, 1e-13)]
        out["tolerance"].append(dict(N_eq=N_eq, Ophi0=vals))
        print(f"(b) N_eq={N_eq}: Omega_phi0 at tol 1e-9, 1e-11, 1e-13: " + " ".join(f"{v:.4f}" for v in vals), flush=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    jobs = [(float(N), 1e-10) for N in np.linspace(60.0, 110.0, n, endpoint=False)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=25)
    out["ensemble"] = res
    json.dump(out, open("results_tian_partIV.json", "w"))
    A = {k: np.array([r[k] for r in res]) for k in res[0]}
    bbn = A["bbn_max"] < 0.04
    today = (A["Ophi0"] > 0.66) & (A["Ophi0"] < 0.72) & (A["w0"] < -0.9)
    both = bbn & today
    print(f"(c) runs {n}; pass BBN {bbn.sum()} ({100*bbn.mean():.1f}%); match today {today.sum()} "
          f"({100*today.mean():.1f}%); both {both.sum()} ({100*both.mean():.2f}%)")
    for name, sel in (("all runs", np.ones(n, bool)), ("pass BBN", bbn), ("match today", today), ("both", both)):
        if sel.any():
            q = np.percentile(A["Ophi_rec"][sel], [0, 5, 50, 95, 100])
            print(f"    Omega_phi at z=1100, {name:11s}: min {q[0]:.4f}  5% {q[1]:.4f}  median {q[2]:.4f}  95% {q[3]:.4f}  max {q[4]:.4f};"
                  f"  max over 50<z<5000: min {A['cmb_max'][sel].min():.4f} median {np.median(A['cmb_max'][sel]):.4f};"
                  f"  fraction with max < 0.0036: {np.mean(A['cmb_max'][sel] < 0.0036):.4f}, < 0.02: {np.mean(A['cmb_max'][sel] < 0.02):.4f}")
