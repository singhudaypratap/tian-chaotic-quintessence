"""When does the field get trapped, relative to matter-radiation equality?

Radiation + matter + field (equations of 04_viability.py). For each parameter
set, many initial phases are evolved from deep in the radiation era. Recorded:
  N_trap - N_eq : e-folds between equality and trapping (y^2 > 0.9995 is used as
                  the marker that the field has frozen and dominates), and
  N_L - N_eq    : e-folds between equality and vacuum-matter equality.
Observed value of the latter: ln(3421 / 1.306) = 7.87.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from importlib import import_module
v = import_module("04_viability")

DN = v.DN
N_EQ = DN + np.log(v.R0)            # e-folds from the start to rho_r = rho_m


def lam_matter(N, s, *a):           # vacuum-matter equality: Omega_phi = Omega_m
    x, y, r, th = s
    return (x * x + y * y) - (1 - x * x - y * y - r * r)
lam_matter.direction = 1


def one(args):
    lam, A, nu, th0 = args
    x0, y0 = v.S32 * (4 / 3) / lam, np.sqrt(4 / 3) / lam
    ratio = v.R0 * np.exp(DN)
    r0 = np.sqrt((1 - x0 * x0 - y0 * y0) * ratio / (1 + ratio))
    sol = solve_ivp(v.rhs, [0, DN + 15], [x0, y0, r0, th0], args=(lam, A, nu),
                    method="LSODA", rtol=1e-9, atol=1e-12,
                    events=[v.frozen, lam_matter])
    if len(sol.t_events[0]) == 0:
        return None, None
    N_dom = sol.t_events[0][0]
    cross = [t for t in sol.t_events[1]]
    N_L = cross[-1] if cross else None
    return N_dom - N_EQ, (N_L - N_EQ if N_L is not None else None)


if __name__ == "__main__":
    nph = 240
    out = []
    for lam, A in [(30.0, 0.7), (100.0, 0.7), (30.0, 0.9)]:
        for rho in (1.02, 1.05, 1.1, 1.2, 1.3):
            nu = rho * lam * np.sqrt(1 - A * A) / A
            jobs = [(lam, A, nu, th) for th in np.linspace(0, 2 * np.pi, nph, endpoint=False)]
            with Pool(2) as pool:
                res = pool.map(one, jobs, chunksize=8)
            NL = np.array([r[1] for r in res if r[1] is not None])
            never = sum(r[0] is None for r in res)
            early = int(np.sum(NL < -1))            # vacuum domination before equality
            late = NL[NL >= -1]
            rec = dict(lam=lam, A=A, rho=rho, nu=nu, n=nph, not_trapped=never,
                       trapped_in_radiation_era=early, n_after_equality=int(late.size),
                       NL_minus_Neq=NL.tolist())
            if late.size:
                q = np.percentile(late, [5, 25, 50, 75, 95])
                rec["late_percentiles_5_25_50_75_95"] = q.tolist()
                rec["frac_within_1_efold_of_observed"] = float(np.mean(np.abs(late - 7.87) < 1))
                qs = " ".join(f"{t:6.2f}" for t in q)
            else:
                qs = "-"
            out.append(rec)
            print(f"lam={lam:5.0f} A={A} rho={rho:.2f}: rolling forever {never:3d}, "
                  f"vacuum-dominated before equality {early:3d}, after {late.size:3d}; "
                  f"N_L-N_eq percentiles (5,25,50,75,95): {qs}", flush=True)
            json.dump(out, open("results_equality_trapping.json", "w"), indent=1)
