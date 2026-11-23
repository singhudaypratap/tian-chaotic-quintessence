"""Cosmological viability scan of the modulated exponential model.

Radiation + matter + scalar field, flat FLRW. State (x, y, r, theta) with
r^2 = Omega_r. Each run starts 30 e-folds before today on the radiation-era
scaling solution; "today" is where Omega_r / Omega_m equals the measured ratio.
The only free initial datum is the phase theta_i, which is scanned.

Indicative bounds used (see RESULTS.md for sources and caveats):
  BBN:    Omega_phi < 0.045   for 1e8 < z < 1e10
  CMB:    Omega_phi < 0.0036  (strict) or < 0.02 (loose) for 50 < z < 5000
  today:  0.66 < Omega_phi < 0.72 and w_phi < -0.7
This is a feasibility scan, not a likelihood fit.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp

S32, S6 = np.sqrt(1.5), np.sqrt(6.0)
R0 = 4.18e-5 / 0.143          # Omega_r0 / Omega_m0 = (Omega_r h^2)/(Omega_m h^2)
DN = 30.0


def rhs(N, s, lam, A, nu):
    x, y, r, th = s
    le = lam - A * nu * np.cos(th) / (1 + A * np.sin(th))
    c = 1.5 * (1 + x * x - y * y + r * r / 3.0)      # = -H'/H
    return [-3 * x + S32 * le * y * y + c * x,
            -S32 * le * x * y + c * y,
            -2 * r + c * r,
            S6 * nu * x]


def today(N, s, *a):
    x, y, r, th = s
    return np.log(r * r / max(1 - x * x - y * y - r * r, 1e-300)) - np.log(R0)
today.terminal, today.direction = True, -1


def frozen(N, s, *a):
    """Field trapped in a potential minimum before today: de Sitter forever."""
    return s[1] * s[1] - 0.9995
frozen.terminal, frozen.direction = True, 1


def run_one(args):
    lam, A, nu, th0 = args
    x0, y0 = S32 * (4 / 3) / lam, np.sqrt(4 / 3) / lam
    om_rest = 1 - x0 * x0 - y0 * y0
    ratio = R0 * np.exp(DN)                           # Omega_r / Omega_m at start
    r0 = np.sqrt(om_rest * ratio / (1 + ratio))
    sol = solve_ivp(rhs, [0, 3 * DN], [x0, y0, r0, th0], args=(lam, A, nu),
                    method="LSODA", rtol=1e-9, atol=1e-12, events=[today, frozen],
                    dense_output=True)
    if sol.status != 1:
        return dict(th0=th0, reached_today=False)
    if len(sol.t_events[0]) == 0:                     # trapped before today
        return dict(th0=th0, reached_today=True, Ophi0=1.0, w0=-1.0, q0=-1.0,
                    Ophi_bbn=1.0, Ophi_cmb=1.0, Ophi_mat=1.0, w_min_z2=-1.0,
                    w_max_z2=-1.0, dH_max=float("inf"), fate="trapped before today")
    N0 = sol.t_events[0][0]
    N = np.linspace(0, N0, 6000)
    x, y, r, th = sol.sol(N)
    z = np.exp(N0 - N) - 1
    Oph = x * x + y * y
    Om = 1 - Oph - r * r

    def mx(lo, hi):
        sel = (z > lo) & (z < hi)
        return float(Oph[sel].max()) if sel.any() else None

    wphi = (x * x - y * y) / np.maximum(Oph, 1e-300)
    late = z < 2
    a = 1 / (1 + z[late])
    Om0 = Om[-1]
    E = np.sqrt(Om0 * a ** -3 / Om[late])
    Or0 = Om0 * R0
    El = np.sqrt(Om0 * a ** -3 + Or0 * a ** -4 + 1 - Om0 - Or0)
    # future fate: 40 more e-folds
    fut = solve_ivp(rhs, [0, 40], sol.y_events[0][0], args=(lam, A, nu),
                    method="LSODA", rtol=1e-9, atol=1e-12)
    xf, yf = fut.y[0, -1], fut.y[1, -1]
    fate = "trapped (de Sitter)" if abs(xf) < 1e-6 and abs(yf - 1) < 1e-4 else "rolling"
    return dict(th0=th0, reached_today=True, Ophi0=float(Oph[-1]), w0=float(wphi[-1]),
                q0=float(-1 + 1.5 * (1 + x[-1] ** 2 - y[-1] ** 2 + r[-1] ** 2 / 3)),
                Ophi_bbn=mx(1e8, 1e10), Ophi_cmb=mx(50, 5000), Ophi_mat=mx(3, 50),
                w_min_z2=float(wphi[late].min()), w_max_z2=float(wphi[late].max()),
                dH_max=float(np.max(np.abs(E / El - 1))), fate=fate)


def classify(r, cmb_bound):
    return (r["reached_today"] and 0.66 < r["Ophi0"] < 0.72 and r["w0"] < -0.7
            and r["Ophi_bbn"] < 0.045 and r["Ophi_cmb"] < cmb_bound)


def do_cell(cell, nph=24, nbis=12):
    """Scan the initial phase, then bisect on Omega_phi(today) = 0.69."""
    lam, A, nu = cell
    ths = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    rr = [run_one((lam, A, nu, th)) for th in ths]
    val = [r["Ophi0"] - 0.69 if r["reached_today"] else np.nan for r in rr]
    sols = []
    for i in range(nph):
        j = (i + 1) % nph
        if not (np.isfinite(val[i]) and np.isfinite(val[j])) or val[i] * val[j] > 0:
            continue
        a, b, fa = ths[i], ths[i] + 2 * np.pi / nph, val[i]
        r = None
        for _ in range(nbis):
            c = 0.5 * (a + b)
            r = run_one((lam, A, nu, c))
            fc = r["Ophi0"] - 0.69 if r["reached_today"] else np.nan
            if not np.isfinite(fc):
                break
            if fa * fc <= 0:
                b = c
            else:
                a, fa = c, fc
        if r is not None and r["reached_today"] and abs(r["Ophi0"] - 0.69) < 0.03:
            sols.append(r)
    reached = [r for r in rr if r["reached_today"]]
    return dict(lam=lam, A=A, nu=nu, ratio=A * nu / (lam * np.sqrt(1 - A * A)),
                n_phase=nph, n_brackets=len(sols),
                Ophi0_min=min((r["Ophi0"] for r in reached), default=None),
                Ophi0_max=max((r["Ophi0"] for r in reached), default=None),
                solutions=sols,
                n_viable_strict=sum(classify(r, 0.0036) for r in sols),
                n_viable_loose=sum(classify(r, 0.02) for r in sols))


if __name__ == "__main__":
    import os
    lams = [float(v) for v in sys.argv[1].split(",")]
    As = [0.7, 0.9, 0.97]
    ratios = [0.8, 0.95, 1.05, 1.2, 1.5]             # A nu / (lam sqrt(1-A^2)); minima exist if > 1
    cells = [(l, A, rho * l * np.sqrt(1 - A * A) / A) for l in lams for A in As for rho in ratios]
    path = "results_viability.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(round(t["lam"], 6), round(t["A"], 6), round(t["nu"], 6))
                for t in map(json.loads, open(path))}
    todo = [c for c in cells if (round(c[0], 6), round(c[1], 6), round(c[2], 6)) not in done]
    with Pool(2) as pool, open(path, "a") as fh:
        for t in pool.imap(do_cell, todo, chunksize=1):
            fh.write(json.dumps(t) + "\n")
            fh.flush()
            print(f"{t['lam']:4.0f} {t['A']:5.2f} {t['nu']:7.2f} ratio={t['ratio']:.2f} "
                  f"Ophi0 in [{t['Ophi0_min']:.4f},{t['Ophi0_max']:.4f}] sols={t['n_brackets']} "
                  f"strict={t['n_viable_strict']} loose={t['n_viable_loose']}", flush=True)
            for b in t["solutions"]:
                print(f"      w0={b['w0']:.3f} w(z<2) in [{b['w_min_z2']:.2f},{b['w_max_z2']:.2f}] "
                      f"dH/H={b['dH_max']:.3f} bbn={b['Ophi_bbn']:.4f} cmb={b['Ophi_cmb']:.4f} "
                      f"mat={b['Ophi_mat']:.4f} {b['fate']}", flush=True)
