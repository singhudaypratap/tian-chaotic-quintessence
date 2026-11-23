"""Modulated exponential quintessence with one barotropic fluid.

V = V0 exp(-lam*kappa*phi) [1 + A sin(nu*kappa*phi)],  0 <= A < 1.
State (x, y, theta), theta = nu*kappa*phi, time N = ln a:

  x'     = -3x + sqrt(3/2) lam_eff y^2 + (3/2) x [2x^2 + g(1-x^2-y^2)]
  y'     = -sqrt(3/2) lam_eff x y      + (3/2) y [2x^2 + g(1-x^2-y^2)]
  theta' = sqrt(6) nu x
  lam_eff(theta) = lam - A nu cos(theta) / (1 + A sin(theta)),  g = 1 + w.
"""
import numpy as np
from scipy.integrate import solve_ivp

S32, S6 = np.sqrt(1.5), np.sqrt(6.0)


def lam_eff(th, lam, A, nu):
    return lam - A * nu * np.cos(th) / (1 + A * np.sin(th))


def dlam_eff(th, lam, A, nu):
    return A * nu * (np.sin(th) + A) / (1 + A * np.sin(th)) ** 2


def rhs(N, s, lam, A, nu, w):
    x, y, th = s[0], s[1], s[2]
    g = 1 + w
    le = lam_eff(th, lam, A, nu)
    brk = 2 * x * x + g * (1 - x * x - y * y)
    return np.array([-3 * x + S32 * le * y * y + 1.5 * x * brk,
                     -S32 * le * x * y + 1.5 * y * brk,
                     S6 * nu * x])


def jac(s, lam, A, nu, w):
    x, y, th = s[0], s[1], s[2]
    g = 1 + w
    le, dle = lam_eff(th, lam, A, nu), dlam_eff(th, lam, A, nu)
    brk = 2 * x * x + g * (1 - x * x - y * y)
    return np.array([
        [-3 + 1.5 * brk + 1.5 * (4 - 2 * g) * x * x,
         2 * S32 * le * y - 3 * g * x * y, S32 * y * y * dle],
        [-S32 * le * y + 1.5 * (4 - 2 * g) * x * y,
         -S32 * le * x + 1.5 * brk - 3 * g * y * y, -S32 * x * y * dle],
        [S6 * nu, 0.0, 0.0]])


def rhs_tan(N, z, lam, A, nu, w):
    return np.concatenate([rhs(N, z[:3], lam, A, nu, w),
                           jac(z[:3], lam, A, nu, w) @ z[3:]])


def evolve(s0, T, p, rtol=1e-10, atol=1e-13, **kw):
    return solve_ivp(rhs, [0, T], s0, args=p, method="DOP853",
                     rtol=rtol, atol=atol, **kw)


def lyap_tangent(s0, p, T=1500.0, burn=400.0, dt=2.0, rng=None):
    """Largest Lyapunov exponent by tangent-vector renormalisation."""
    rng = rng or np.random.default_rng(0)
    s = evolve(s0, burn, p).y[:, -1]
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    acc, n = 0.0, int(T / dt)
    hist = []
    for i in range(n):
        sol = solve_ivp(rhs_tan, [0, dt], np.concatenate([s, d]), args=p,
                        method="DOP853", rtol=1e-10, atol=1e-13)
        s, d = sol.y[:3, -1], sol.y[3:, -1]
        nd = np.linalg.norm(d)
        acc += np.log(nd)
        d /= nd
        hist.append(acc / ((i + 1) * dt))
    return acc / (n * dt), s, np.array(hist)


def lyap_two_traj(s0, p, burn=400.0, T=30.0, d0=1e-9, nrep=40, rng=None):
    """Jacobian-free estimate from the divergence of two nearby orbits."""
    rng = rng or np.random.default_rng(1)
    s = evolve(s0, burn, p).y[:, -1]
    acc = 0.0
    for _ in range(nrep):
        v = rng.normal(size=3)
        v *= d0 / np.linalg.norm(v)
        a = evolve(s, T, p, rtol=1e-12, atol=1e-15).y[:, -1]
        b = evolve(s + v, T, p, rtol=1e-12, atol=1e-15).y[:, -1]
        acc += np.log(np.linalg.norm(a - b) / d0) / T
        s = a
    return acc / nrep


def is_trapped(s):
    return abs(s[0]) < 1e-7 and abs(abs(s[1]) - 1) < 1e-6


def section_points(s0, p, n_pts=120, burn=400.0, max_T=4000.0):
    """Successive (x, y) at theta = 0 mod 2 pi, crossed with theta increasing."""
    s = evolve(s0, burn, p).y[:, -1]
    if is_trapped(s):
        return np.empty((0, 2)), np.empty(0)

    def ev(N, z, *a):
        return np.sin(z[2] / 2.0)        # zero at theta = 2 pi k
    ev.direction = 0
    sol = evolve(s, max_T, p, events=ev)
    pts, times = [], []
    for t, z in zip(sol.t_events[0], sol.y_events[0]):
        if z[0] > 0:                      # theta increasing
            pts.append(z[:2])
            times.append(t)
    pts, times = np.array(pts), np.array(times)
    return pts[-n_pts:], times[-n_pts:]
