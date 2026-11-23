"""Reproduce the regular and chaotic regimes of Tian's model and measure the
Lyapunov exponent, which Tian's papers do not report.

(a) Single fluid: largest exponent and return-map period in a radiation
    background (w = 1/3) and a matter background (w = 0), for Tian's
    parameters l2 = 0.05, alpha = 0.6, l1 = 4.25, 4.5, 4.75.
(b) Sweep of the background equation of state w at l1 = 4.5.
"""
import json
import numpy as np
import tian as t

L2, AL = 0.05, 0.6
S0 = np.array([0.3, 0.3, 0.5])
out = dict(a=[], b=[])


def classify(p):
    l1, s_end, hist = t.lyap1(S0, p, T=2000.0)
    k = len(hist)
    conv = [float(hist[int(k * f) - 1]) for f in (0.25, 0.5, 1.0)]
    pts = t.section1(s_end, p, n_pts=64, max_T=1500.0)
    period = None
    for q in (1, 2, 4, 8, 16, 3, 5, 6, 12):
        if len(pts) > 2 * q and np.max(np.abs(pts[q:] - pts[:-q])) < 1e-6:
            period = q
            break
    sol = t.evolve1(s_end, 300.0, p, t_eval=np.linspace(0, 300, 30001))
    Oph = sol.y[0] ** 2 + sol.y[1] ** 2
    weff = sol.y[0] ** 2 - sol.y[1] ** 2 + p[3] * (1 - Oph)
    return dict(lyap=conv, map_period=period, Ophi_min=float(Oph.min()),
                Ophi_max=float(Oph.max()), frac_accelerating=float(np.mean(weff < -1 / 3)))


for l1 in (4.25, 4.5, 4.75):
    for w in (1 / 3, 0.0):
        rec = dict(l1=l1, w=w, **classify((l1, L2, AL, w)))
        out["a"].append(rec)
        print(f"(a) l1={l1} w={w:.3f}: lambda_1 (T=500,1000,2000) = "
              + " ".join(f"{v:.4f}" for v in rec["lyap"])
              + f"  map period={rec['map_period']}  Omega_phi in "
              f"[{rec['Ophi_min']:.3f},{rec['Ophi_max']:.3f}]  accelerating {100*rec['frac_accelerating']:.0f}% of the time",
              flush=True)
    json.dump(out, open("results_tian_reproduction.json", "w"), indent=1)

for w in np.round(np.arange(0.36, -0.001, -0.03), 3):
    rec = dict(l1=4.5, w=float(w), **classify((4.5, L2, AL, float(w))))
    out["b"].append(rec)
    print(f"(b) w={w:5.2f}: lambda_1={rec['lyap'][-1]: .4f} map period={rec['map_period']} "
          f"Omega_phi max={rec['Ophi_max']:.3f}", flush=True)
    json.dump(out, open("results_tian_reproduction.json", "w"), indent=1)
