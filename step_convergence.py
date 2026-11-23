"""Step-size convergence of the fixed-step ensemble against an adaptive solver,
from the same random starting states. Usage: python3 step_convergence.py L1 ALPHA N"""
import sys, math
import numpy as np
from scipy.integrate import solve_ivp
import gpu_ensemble as g
import tian

l1, al, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
l2, burn = 0.01, 300.0


def rates(o):
    bbn = o["bbn_max"] < 0.04
    cmb = o["cmb_max"] < 0.02
    today = (o["Ophi0"] > 0.6) & (o["Ophi0"] < 0.8) & (o["w0"] < -0.8)
    return f"BBN {100*bbn.mean():5.1f}%  recomb {100*cmb.mean():5.1f}%  today {100*today.mean():4.1f}%"


for dN in ([] if "--adaptive-only" in sys.argv else (2e-3, 1e-3, 5e-4)):
    o, *_ = g.run(l1, l2, al, n, dN, burn, "numpy", 1)
    print(f"fixed step {dN:.0e}: {rates(o)}", flush=True)

# adaptive reference from the same starting states (same RNG calls as gpu_ensemble.run)
rng = np.random.default_rng(1)
r = np.sqrt(rng.uniform(0.01, 0.9, n)); ang = rng.uniform(0.1, math.pi - 0.1, n)
x0, y0, th0 = r * np.cos(ang), r * np.sin(ang), rng.uniform(0, 2 * math.pi, n)
N_end = g.N_EQ + g.LN_1PZEQ
out = {k: np.zeros(n) for k in ("bbn_max", "cmb_max", "Ophi0", "w0")}
METHOD = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "DOP853"
for i in range(n):
    s = solve_ivp(tian.rhs1u, [0, burn], tian.with_u([x0[i], y0[i], th0[i]]), args=(l1, l2, al, 1 / 3),
                  method=METHOD, rtol=1e-10, atol=1e-13).y[:, -1]
    N = np.linspace(0, N_end, 20000)
    sol = solve_ivp(tian.rhs_fullu, [0, N_end], s, args=(l1, l2, al, g.N_EQ), method=METHOD,
                    rtol=1e-10, atol=1e-13, t_eval=N)
    x, y = sol.y[0], sol.y[1]
    z = np.exp(N_end - N) - 1
    O = x * x + y * y
    out["bbn_max"][i] = O[(z > g.Z_BBN[0]) & (z < g.Z_BBN[1])].max()
    out["cmb_max"][i] = O[(z > g.Z_CMB[0]) & (z < g.Z_CMB[1])].max()
    out["Ophi0"][i] = O[-1]
    out["w0"][i] = (x[-1] ** 2 - y[-1] ** 2) / O[-1]
print(f"adaptive ({METHOD}, rtol 1e-10): {rates(out)}", flush=True)
