"""Time-averaged statistics on the radiation attractor: fixed-step RK4 (u form) vs adaptive.
Usage: python3 attractor_check.py L1 ALPHA T"""
import sys, math, numpy as np
from scipy.integrate import solve_ivp
import gpu_ensemble as g, tian
l1, al, T = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); l2 = 0.01
B = g.Backend("numpy")
n = 16
rng = np.random.default_rng(3)
r = np.sqrt(rng.uniform(0.01, 0.9, n)); ang = rng.uniform(0.1, math.pi - 0.1, n)
x0, y0, th0 = r*np.cos(ang), r*np.sin(ang), rng.uniform(0, 2*math.pi, n)
def stats(O):
    return f"<O>={O.mean():.4f}  P(O>0.04)={np.mean(O>0.04):.4f}  P(O>0.5)={np.mean(O>0.5):.4f}  P(O>0.99)={np.mean(O>0.99):.4f}"
for h in (2e-3, 5e-4):
    x, y, th = B.array(x0), B.array(y0), B.array(th0); u = B.log(1-x*x-y*y)
    lb, dl = 0.5*(l1+l2), 0.5*(l1-l2); Os = []
    for i in range(int(T/h)):
        x, y, th, u = g.rk4(B, x, y, th, u, 1/3, 1/3, 1/3, h, lb, dl, al)
        if i*h > 100 and i % int(0.01/h) == 0: Os.append(x*x+y*y)
    print(f"RK4 h={h:.0e}: {stats(np.concatenate(Os))}", flush=True)
for m in sys.argv[4:] or ["DOP853"]:
    Os = []
    for i in range(n):
        tt = np.arange(100, T, 0.01)
        sol = solve_ivp(tian.rhs1u, [0, T], tian.with_u([x0[i], y0[i], th0[i]]), args=(l1, l2, al, 1/3),
                        method=m, rtol=1e-10, atol=1e-13, t_eval=tt)
        Os.append(sol.y[0]**2 + sol.y[1]**2)
    print(f"{m}: {stats(np.concatenate(Os))}", flush=True)
