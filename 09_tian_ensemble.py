"""Full radiation + matter + field histories of Tian's model over an ensemble
of N_eq (the e-folds from the initial state to equality), which is how Tian
parametrises the initial condition.

For each run: Omega_phi and w_phi today, Omega_phi at recombination (z = 1100),
the range of Omega_phi during nucleosynthesis (1e8 < z < 1e10) and in the
window 50 < z < 5000. Also a tolerance check and Tian's three quoted points.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import tian as t

L2, AL = 0.05, 0.6


def summarise(h):
    z, O = h["z"], h["Ophi"]
    def rng(lo, hi):
        s = (z > lo) & (z < hi)
        return [float(O[s].min()), float(O[s].max())] if s.any() else [None, None]
    i_rec = int(np.argmin(np.abs(z - 1100.0)))
    late = z < 2
    return dict(Ophi0=float(O[-1]), w0=float(h["wphi"][-1]), weff0=float(h["weff"][-1]),
                Ophi_rec=float(O[i_rec]), bbn=rng(1e8, 1e10), cmb=rng(50, 5000),
                matter=rng(2, 50), w_range_z2=[float(h["wphi"][late].min()), float(h["wphi"][late].max())])


def one(args):
    l1, N_eq, tol = args
    try:
        h = t.history(l1, L2, AL, N_eq, rtol=tol, atol=tol * 1e-3, n_out=6000)
        return dict(l1=l1, N_eq=N_eq, **summarise(h))
    except Exception as e:                      # integrator failure is recorded, not hidden
        return dict(l1=l1, N_eq=N_eq, error=str(e))


if __name__ == "__main__":
    out = {}
    # Tian's quoted fit points (Paper II, Fig. 3)
    quoted = [(4.25, 81.16), (4.5, 81.94), (4.75, 84.50)]
    out["quoted"] = [one((l1, N, 1e-10)) for l1, N in quoted]
    for r in out["quoted"]:
        print("quoted", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # tolerance check on the same points
    out["tolerance"] = []
    for l1, N in quoted:
        a, b = one((l1, N, 1e-9)), one((l1, N, 1e-12))
        out["tolerance"].append(dict(l1=l1, N_eq=N, Ophi0_1e9=a["Ophi0"], Ophi0_1e12=b["Ophi0"]))
        print(f"tolerance l1={l1}: Omega_phi0 = {a['Ophi0']:.6f} (1e-9) vs {b['Ophi0']:.6f} (1e-12)", flush=True)
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    for l1 in (4.25, 4.5, 4.75):
        jobs = [(l1, float(N), 1e-10) for N in np.arange(45.0, 95.0, step)]
        with Pool(2) as pool:
            res = pool.map(one, jobs, chunksize=20)
        ok = [r for r in res if "error" not in r]
        O0 = np.array([r["Ophi0"] for r in ok])
        rec = np.array([r["Ophi_rec"] for r in ok])
        good = (O0 > 0.66) & (O0 < 0.72)
        acc = good & (np.array([r["w0"] for r in ok]) < -0.7)
        bbn_ok = np.array([r["bbn"][1] < 0.045 for r in ok])
        print(f"l1={l1}: runs {len(ok)}/{len(res)}; Omega_phi0 in [0.66,0.72]: {good.sum()}; "
              f"and w0 < -0.7: {acc.sum()}; BBN max Omega_phi < 0.045: {bbn_ok.sum()}; "
              f"Omega_phi at z=1100: min {rec.min():.3f}, median {np.median(rec):.3f}, max {rec.max():.3f}; "
              f"among today-compatible runs: min {rec[acc].min() if acc.any() else float('nan'):.3f}", flush=True)
        dO = np.abs(np.diff(O0))
        print(f"        median |change in Omega_phi0| between neighbouring N_eq (step {step}): {np.median(dO):.3f}", flush=True)
        out[f"ensemble_{l1}"] = res
        json.dump(out, open("results_tian_ensemble.json", "w"))
