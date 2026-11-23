"""Tian's oscillating-quintessence model (Phys. Rev. D 101, 063531; 102, 063509).

V = V0 exp[-(l1+l2) phi/2 - alpha (l1-l2) sin(phi/alpha)/2]   (kappa = 1)
so   lam(phi) = -V'/V = lbar + dl cos(theta),  theta = phi/alpha,
     lbar = (l1+l2)/2,  dl = (l1-l2)/2,   l2 <= lam <= l1.

Two forms, both with N = ln a as time:
  single fluid  (x, y, theta)      background equation of state w constant
  full          (x, y, r, theta)   radiation (r^2 = Omega_r) + matter + field
"""
import numpy as np
from scipy.integrate import solve_ivp

S32, S6 = np.sqrt(1.5), np.sqrt(6.0)
Z_EQ = 3400.0                      # value used by Tian


def lam_of(th, l1, l2):
    return 0.5 * (l1 + l2) + 0.5 * (l1 - l2) * np.cos(th)


# ---------------- single fluid ----------------
def rhs1(N, s, l1, l2, al, w):
    x, y, th = s[0], s[1], s[2]
    g, le = 1 + w, lam_of(th, l1, l2)
    brk = 2 * x * x + g * (1 - x * x - y * y)
    return np.array([-3 * x + S32 * le * y * y + 1.5 * x * brk,
                     -S32 * le * x * y + 1.5 * y * brk,
                     S6 * x / al])


def jac1(s, l1, l2, al, w):
    x, y, th = s[0], s[1], s[2]
    g, le = 1 + w, lam_of(th, l1, l2)
    dle = -0.5 * (l1 - l2) * np.sin(th)
    brk = 2 * x * x + g * (1 - x * x - y * y)
    return np.array([
        [-3 + 1.5 * brk + 1.5 * (4 - 2 * g) * x * x,
         2 * S32 * le * y - 3 * g * x * y, S32 * y * y * dle],
        [-S32 * le * y + 1.5 * (4 - 2 * g) * x * y,
         -S32 * le * x + 1.5 * brk - 3 * g * y * y, -S32 * x * y * dle],
        [S6 / al, 0.0, 0.0]])


def rhs1_tan(N, z, *p):
    return np.concatenate([rhs1(N, z[:3], *p), jac1(z[:3], *p) @ z[3:]])


def evolve1(s0, T, p, rtol=1e-10, atol=1e-13, **kw):
    return solve_ivp(rhs1, [0, T], s0, args=p, method="DOP853", rtol=rtol, atol=atol, **kw)


def lyap1(s0, p, T=2000.0, burn=300.0, dt=2.0, seed=0):
    """Largest Lyapunov exponent (per e-fold) with its running estimate."""
    rng = np.random.default_rng(seed)
    s = evolve1(s0, burn, p).y[:, -1]
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    acc, hist = 0.0, []
    for i in range(int(T / dt)):
        sol = solve_ivp(rhs1_tan, [0, dt], np.concatenate([s, d]), args=p,
                        method="DOP853", rtol=1e-10, atol=1e-13)
        s, d = sol.y[:3, -1], sol.y[3:, -1]
        nd = np.linalg.norm(d)
        acc += np.log(nd)
        d /= nd
        hist.append(acc / ((i + 1) * dt))
    return hist[-1], s, np.array(hist)


def section1(s, p, n_pts=64, max_T=3000.0):
    """(x, y) at theta = 0 mod 2 pi, theta increasing, after the state s."""
    def ev(N, z, *a):
        return np.sin(z[2] / 2.0)
    sol = evolve1(s, max_T, p, events=ev)
    pts = np.array([z[:2] for z in sol.y_events[0] if z[0] > 0])
    return pts[-n_pts:]


# ---------------- radiation + matter + field ----------------
# rho_r / rho_m = exp(N_eq - N) exactly, so the two fluids act as one fluid with
# w_bg(N) = (1/3) R / (1 + R), R = exp(N_eq - N). This avoids computing Omega_m
# by subtraction, which loses all precision when Omega_m ~ exp(-N_eq).
def w_bg(N, N_eq):
    return (1.0 / 3.0) / (1.0 + np.exp(np.clip(N - N_eq, -700, 700)))


def rhs_full(N, s, l1, l2, al, N_eq):
    return rhs1(N, s, l1, l2, al, w_bg(N, N_eq))


def history(l1, l2, al, N_eq, x0=0.75, y0=0.5, lam0=0.35, n_out=4000,
            rtol=1e-10, atol=1e-13, extra=0.0):
    """Evolve from Tian's initial data to today, N_today = N_eq + ln(1+z_eq).

    N_eq is the number of e-folds from the start to rho_r = rho_m."""
    th0 = np.arccos(np.clip((lam0 - 0.5 * (l1 + l2)) / (0.5 * (l1 - l2)), -1, 1))
    N_end = N_eq + np.log(1 + Z_EQ) + extra
    N = np.linspace(0, N_end, n_out)
    sol = solve_ivp(rhs_full, [0, N_end], [x0, y0, th0], args=(l1, l2, al, N_eq),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=N)
    x, y, th = sol.y
    z = np.exp(N_eq + np.log(1 + Z_EQ) - N) - 1
    Oph = x * x + y * y
    wb = w_bg(N, N_eq)
    Or = (1 - Oph) * 3 * wb                     # R/(1+R) = 3 w_bg
    return dict(N=N, z=z, x=x, y=y, theta=th, Ophi=Oph, Or=Or, Om=1 - Oph - Or,
                wphi=(x * x - y * y) / np.maximum(Oph, 1e-300),
                weff=x * x - y * y + wb * (1 - Oph))


def attractor_states(l1, l2, al, n, spacing=4.0, burn=300.0, w=1.0 / 3.0, s0=(0.3, 0.3, 0.5)):
    """n states on the radiation-era attractor, `spacing` e-folds apart.

    In the radiation era the system is autonomous, so histories that differ only
    in N_eq follow one trajectory and are not independent. Sampling states along
    a long trajectory, further apart than the decorrelation time, is."""
    p = (l1, l2, al, w)
    s = evolve1(np.array(s0), burn, p, rtol=1e-9, atol=1e-12).y[:, -1]
    tt = np.arange(1, n + 1) * spacing
    sol = evolve1(s, tt[-1], p, rtol=1e-9, atol=1e-12, t_eval=tt)
    return sol.y.T


def history_from_state(l1, l2, al, N_eq, s0, n_out=4000, rtol=1e-9, atol=1e-12):
    """Radiation + matter + field history from a given state, N_eq e-folds before equality."""
    N_end = N_eq + np.log(1 + Z_EQ)
    N = np.linspace(0, N_end, n_out)
    sol = solve_ivp(rhs_full, [0, N_end], s0, args=(l1, l2, al, N_eq),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=N)
    if sol.status != 0 or sol.y.shape[1] != n_out:
        raise RuntimeError("integration stopped early: " + str(sol.message))
    x, y, th = sol.y
    z = np.exp(N_end - N) - 1
    Oph = x * x + y * y
    return dict(N=N, z=z, Ophi=Oph, wphi=(x * x - y * y) / np.maximum(Oph, 1e-300))
