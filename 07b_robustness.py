"""Robustness of the trapping-time distribution to the initial state, the
starting epoch and the integrator tolerance."""
import json
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
from importlib import import_module
v = import_module("04_viability")
t7 = import_module("07_equality_triggered_trapping")


def one(args):
    lam, A, nu, th0, fx, fy, DN, tol = args
    x0, y0 = fx * v.S32 * (4 / 3) / lam, fy * np.sqrt(4 / 3) / lam
    ratio = v.R0 * np.exp(DN)
    r0 = np.sqrt((1 - x0 * x0 - y0 * y0) * ratio / (1 + ratio))
    sol = solve_ivp(v.rhs, [0, DN + 15], [x0, y0, r0, th0], args=(lam, A, nu),
                    method="LSODA", rtol=tol, atol=tol * 1e-3,
                    events=[v.frozen, t7.lam_matter])
    if len(sol.t_events[0]) == 0 or len(sol.t_events[1]) == 0:
        return None
    return sol.t_events[1][-1] - (DN + np.log(v.R0))


if __name__ == "__main__":
    out = []
    variants = [("baseline", 1, 1, 30.0, 1e-9), ("x0 x3, y0 /3", 3, 1 / 3, 30.0, 1e-9),
                ("x0 /3, y0 x3", 1 / 3, 3, 30.0, 1e-9), ("start 40 e-folds back", 1, 1, 40.0, 1e-9),
                ("tolerance 1e-11", 1, 1, 30.0, 1e-11)]
    for lam, A, rho in [(100.0, 0.7, 1.02), (30.0, 0.7, 1.10)]:
        nu = rho * lam * np.sqrt(1 - A * A) / A
        for name, fx, fy, DN, tol in variants:
            jobs = [(lam, A, nu, th, fx, fy, DN, tol)
                    for th in np.linspace(0, 2 * np.pi, 120, endpoint=False)]
            with Pool(2) as pool:
                res = pool.map(one, jobs, chunksize=8)
            NL = np.array([r for r in res if r is not None])
            late = NL[NL >= -1]
            q = np.percentile(late, [25, 50, 75]) if late.size else [np.nan] * 3
            rec = dict(lam=lam, A=A, rho=rho, variant=name, n=120, never=120 - NL.size,
                       before_equality=int(np.sum(NL < -1)), quartiles=list(map(float, q)),
                       frac_within_1=float(np.mean(np.abs(late - 7.87) < 1)) if late.size else None)
            out.append(rec)
            print(f"lam={lam:4.0f} rho={rho}: {name:22s} never={rec['never']:3d} "
                  f"before eq={rec['before_equality']:3d} quartiles={q[0]:.2f} {q[1]:.2f} {q[2]:.2f} "
                  f"within 1 e-fold of 7.87: {rec['frac_within_1']}", flush=True)
    json.dump(out, open("results_robustness.json", "w"), indent=1)
