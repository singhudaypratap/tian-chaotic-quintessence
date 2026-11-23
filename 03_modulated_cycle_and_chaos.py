"""Modulated exponential potential: limit cycle, its stability, and chaos.

(a) Jacobian checked against finite differences.
(b) Regular cycle at (lam, A, nu, w) = (4, 0.3, 2, 0): period against
    2 pi lam / (3 (1+w) nu), Floquet multipliers, independence of initial data.
(c) Candidate chaotic point (10, 0.7, 9.085, 0): largest Lyapunov exponent by
    two independent methods, with convergence in T.
(d) Sweep in nu at lam = 10, A = 0.7, w = 0: return-map period and exponent.
"""
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp
import modsys as m

rng = np.random.default_rng(20261111)
out = {}

# (a) Jacobian
p = (10.0, 0.7, 9.085, 0.0)
worst = 0.0
for _ in range(300):
    s = np.array([rng.uniform(-.6, .6), rng.uniform(.05, .6), rng.uniform(0, 2 * np.pi)])
    J, e = m.jac(s, *p), 1e-6
    Jn = np.column_stack([(m.rhs(0, s + e * np.eye(3)[k], *p)
                           - m.rhs(0, s - e * np.eye(3)[k], *p)) / (2 * e) for k in range(3)])
    worst = max(worst, np.max(np.abs(J - Jn)) / (1 + np.max(np.abs(J))))
print(f"(a) max relative Jacobian error: {worst:.1e}")
out["jacobian_error"] = worst

# (b) regular cycle
p = (4.0, 0.3, 2.0, 0.0)
lam, A, nu, w = p
pts_all, per = [], None
for k in range(5):
    r, th = np.sqrt(rng.uniform(0.01, 0.9)), rng.uniform(0.1, np.pi - 0.1)
    s0 = np.array([r * np.cos(th), r * np.sin(th), rng.uniform(0, 2 * np.pi)])
    pts, times = m.section_points(s0, p, n_pts=20, burn=300.0, max_T=400.0)
    pts_all.append(pts[-1])
    per = np.diff(times)[-1]
pts_all = np.array(pts_all)
P_pred = 2 * np.pi * lam / (3 * (1 + w) * nu)
print(f"(b) section point spread over 5 initial conditions: {np.ptp(pts_all, axis=0).max():.1e}")
print(f"    period {per:.8f} e-folds, predicted {P_pred:.8f}")
# monodromy over one period, starting on the section
s = np.array([pts_all[0][0], pts_all[0][1], 0.0])
M = np.column_stack([
    solve_ivp(m.rhs_tan, [0, per], np.concatenate([s, np.eye(3)[k]]), args=p,
              method="DOP853", rtol=1e-12, atol=1e-14).y[3:, -1] for k in range(3)])
mult = np.linalg.eigvals(M)
mult = mult[np.argsort(-np.abs(mult))]
print("    Floquet multipliers:", np.array2string(mult, precision=6))
out["regular_cycle"] = dict(params=p, period=float(per), period_predicted=P_pred,
                            section_spread=float(np.ptp(pts_all, axis=0).max()),
                            floquet_abs=np.abs(mult).tolist())

# (c) candidate chaotic point
p = (10.0, 0.7, 9.085, 0.0)
s0 = np.array([0.3, 0.3, 0.1])
l_tan, s_end, hist = m.lyap_tangent(s0, p, T=3000.0, dt=2.0, rng=rng)
idx = [int(len(hist) * f) - 1 for f in (0.1, 0.25, 0.5, 1.0)]
print("(c) tangent-method lambda_1 at T = 300, 750, 1500, 3000:",
      "  ".join(f"{hist[i]:.4f}" for i in idx))
l_two = m.lyap_two_traj(s0, p, rng=rng)
print(f"    two-trajectory lambda_1: {l_two:.4f}")
# range of Omega_phi and of x on the attractor
sol = m.evolve(s_end, 400.0, p, t_eval=np.linspace(0, 400, 40001))
Oph = sol.y[0] ** 2 + sol.y[1] ** 2
print(f"    Omega_phi on attractor: min {Oph.min():.4f}, max {Oph.max():.4f}; "
      f"x range [{sol.y[0].min():.3f}, {sol.y[0].max():.3f}]")
out["chaotic_point"] = dict(params=p, lyap_tangent_vs_T=[float(hist[i]) for i in idx],
                            lyap_two_traj=float(l_two), Ophi_min=float(Oph.min()),
                            Ophi_max=float(Oph.max()))
json.dump(out, open("results_modulated_a_c.json", "w"), indent=1)
if "--no-sweep" in sys.argv:
    sys.exit()

# (d) sweep in nu
lam, A, w = 10.0, 0.7, 0.0
sweep = []
for nu in np.round(np.arange(6.0, 13.01, 0.25), 3):
    p = (lam, A, float(nu), w)
    l1, s_end, _ = m.lyap_tangent(np.array([0.3, 0.3, 0.1]), p, T=1500.0, rng=rng)
    trapped = bool(m.is_trapped(s_end))
    period, xs = None, []
    if not trapped:
        pts, _ = m.section_points(s_end, p, n_pts=96, burn=0.0, max_T=1500.0)
        xs = pts[:, 0].tolist()
        for k in (1, 2, 4, 8, 16, 3, 5, 6, 12):
            if len(pts) > 2 * k and np.max(np.abs(pts[k:] - pts[:-k])) < 1e-6:
                period = k
                break
    rec = dict(nu=float(nu), minima_ratio=float(A * nu / np.sqrt(1 - A * A) / lam),
               lyap=float(l1), trapped=trapped, map_period=period, section_x=xs)
    sweep.append(rec)
    print(f"(d) nu={nu:6.2f} ratio={rec['minima_ratio']:.3f} lambda_1={l1: .4f} "
          f"trapped={trapped} map period={period}", flush=True)
out["nu_sweep"] = sweep
json.dump(out, open("results_modulated.json", "w"), indent=1)
