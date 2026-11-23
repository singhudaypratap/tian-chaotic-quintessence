import math, numpy as np
from scipy.integrate import solve_ivp
import gpu_ensemble as g, tian
l1, al, l2 = 14.0, 0.8, 0.01
B = g.Backend("numpy")
rng = np.random.default_rng(3); n = 8
r = np.sqrt(rng.uniform(0.01, 0.9, n)); ang = rng.uniform(0.1, math.pi - 0.1, n)
s0 = np.array([r*np.cos(ang), r*np.sin(ang), rng.uniform(0, 2*math.pi, n)])
T = 30.0
def rk(h):
    x, y, th = (B.array(v) for v in s0); u = B.log(1-x*x-y*y)
    for i in range(int(round(T/h))):
        x, y, th, u = g.rk4(B, x, y, th, u, 1/3, 1/3, 1/3, h, 0.5*(l1+l2), 0.5*(l1-l2), al)
    return np.array([x, y, th])
ref = rk(2.5e-4)
print("rk 5e-4 vs 2.5e-4:", np.abs(rk(5e-4)-ref).max())
for m, rt in [("DOP853", 1e-10), ("DOP853", 1e-12), ("DOP853", 1e-13), ("LSODA", 1e-12), ("Radau", 1e-11)]:
    e = []
    for i in range(n):
        s = solve_ivp(tian.rhs1u, [0, T], tian.with_u(s0[:, i]), args=(l1, l2, al, 1/3),
                      method=m, rtol=rt, atol=rt*1e-2).y[:3, -1]
        e.append(np.abs(s-ref[:, i]))
    print(m, rt, "max dev per traj:", " ".join(f"{v.max():.0e}" for v in e), flush=True)
