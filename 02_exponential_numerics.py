"""Numerical checks for the pure exponential potential.

A. Full CLW system: every orbit from a random physical initial condition
   ends on a fixed point (no sustained oscillation).
B. Reduced amplitude system of the paper (rotating frame, omega removed):
   full Lyapunov spectrum by the QR method over a sweep of initial
   amplitudes and of (w + 1).
"""
import json
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(20261111)
S32 = np.sqrt(1.5)


# ---------- A. full CLW system ----------
def clw(N, s, lam, w):
    x, y = s
    g = 1.0 + w
    brk = 2 * x * x + g * (1 - x * x - y * y)
    return [-3 * x + lam * S32 * y * y + 1.5 * x * brk,
            -lam * S32 * x * y + 1.5 * y * brk]


def fixed_points(lam, w):
    g = 1.0 + w
    pts = {"origin (fluid)": (0.0, 0.0)}
    if lam * lam < 6:
        pts["B (scalar dominated)"] = (lam / np.sqrt(6), np.sqrt(1 - lam * lam / 6))
    if g > 0 and lam * lam > 3 * g:
        pts["C (scaling)"] = (S32 * g / lam, np.sqrt(1.5 * (2 - g) * g) / lam)
    return pts


def run_full():
    rows = []
    cases = [(1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 1 / 3), (1.0, -0.9),
             (3.0, -0.9), (1.0, -1.2), (3.0, -1.2), (1.0, -1 + 1e-7),
             (1.0, -1 - 1e-7)]
    for lam, w in cases:
        slow = abs(1 + w) < 1e-3          # critical slowing down near w = -1
        Nend = 4000.0 if slow else 400.0
        worst_pp, ends = 0.0, set()
        worst_dist = 0.0
        for _ in range(12):
            r = np.sqrt(rng.uniform(0.0004, 0.98))
            th = rng.uniform(0.02, np.pi - 0.02)
            s0 = [r * np.cos(th), r * np.sin(th)]
            sol = solve_ivp(clw, [0, Nend], s0, args=(lam, w), method="LSODA",
                            rtol=1e-11, atol=1e-13, dense_output=True)
            late = sol.sol(np.linspace(0.9 * Nend, Nend, 2000))
            # peak-to-peak after removing the linear trend (drift is not oscillation)
            t = np.linspace(-1, 1, late.shape[1])
            pp = 0.0
            for comp in late:
                res = comp - np.polyval(np.polyfit(t, comp, 2), t)
                pp = max(pp, np.ptp(res))
            worst_pp = max(worst_pp, pp)
            xe, ye = late[:, -1]
            name, dist = min(((k, np.hypot(xe - p[0], ye - p[1]))
                              for k, p in fixed_points(lam, w).items()),
                             key=lambda kv: kv[1])
            ends.add(name)
            worst_dist = max(worst_dist, dist)
        rows.append(dict(lam=lam, w=w, N_end=Nend, endpoints=sorted(ends),
                         max_dist_to_fixed_point=worst_dist,
                         max_detrended_peak_to_peak=worst_pp))
    return rows


# ---------- B. amplitude system, rotating frame ----------
def amp_rhs(s, Ou, Ov, al):
    ur, ui, vr, vi = s
    return np.array([Ou * ur + al * (vr * vr - vi * vi),
                     Ou * ui + 2 * al * vr * vi,
                     Ov * vr - al * (ur * vr + ui * vi),
                     Ov * vi - al * (ui * vr - ur * vi)])


def amp_jac(s, Ou, Ov, al):
    ur, ui, vr, vi = s
    return np.array([[Ou, 0, 2 * al * vr, -2 * al * vi],
                     [0, Ou, 2 * al * vi, 2 * al * vr],
                     [-al * vr, -al * vi, Ov - al * ur, -al * ui],
                     [al * vi, -al * vr, -al * ui, Ov + al * ur]])


def lyap_spectrum(s0, Ou, Ov, al, T=300.0, dt=1.0):
    def f(N, z):
        s, Q = z[:4], z[4:].reshape(4, 4)
        return np.concatenate([amp_rhs(s, Ou, Ov, al),
                               (amp_jac(s, Ou, Ov, al) @ Q).ravel()])
    z = np.concatenate([s0, np.eye(4).ravel()])
    sums, n, smax = np.zeros(4), int(T / dt), 0.0
    for _ in range(n):
        sol = solve_ivp(f, [0, dt], z, method="DOP853", rtol=1e-10, atol=1e-14)
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            return None, np.inf
        s, M = sol.y[:4, -1], sol.y[4:, -1].reshape(4, 4)
        Q, R = np.linalg.qr(M)
        sums += np.log(np.abs(np.diag(R)))
        smax = max(smax, np.linalg.norm(s))
        if smax > 1e8:
            return None, smax
        z = np.concatenate([s, Q.ravel()])
    return np.sort(sums / (n * dt))[::-1], np.linalg.norm(s)


def run_amp():
    rows = []
    for lam in (1.0, 3.0):
        al = 0.5 * S32 * lam
        for wp1 in (1e-7, -1e-7, 1e-3, 1e-1):
            w = wp1 - 1
            Ou, Ov = 1.5 * (w - 1), 1.5 * (w + 1)
            for amp in (1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0):
                for which in ("u", "v", "both"):
                    ph = rng.uniform(0, 2 * np.pi, 2)
                    small = 1e-3 * amp
                    au = amp if which in ("u", "both") else small
                    av = amp if which in ("v", "both") else small
                    s0 = np.array([au * np.cos(ph[0]), au * np.sin(ph[0]),
                                   av * np.cos(ph[1]), av * np.sin(ph[1])])
                    T = 300.0 if amp <= 100 else 60.0
                    spec, final = lyap_spectrum(s0, Ou, Ov, al, T=T)
                    rows.append(dict(lam=lam, w_plus_1=wp1, amp=amp, excited=which,
                                     T=T, final_norm=float(final),
                                     lyap=None if spec is None else spec.tolist()))
    return rows


if __name__ == "__main__":
    full = run_full()
    print("A. full CLW system")
    for r in full:
        print(f"  lam={r['lam']:<4} w={r['w']:<12.9g} -> {r['endpoints']}, "
              f"dist={r['max_dist_to_fixed_point']:.1e}, "
              f"osc p2p={r['max_detrended_peak_to_peak']:.1e}")
    amp = run_amp()
    ok = [r for r in amp if r["lyap"] is not None]
    lmax = max(r["lyap"][0] for r in ok)
    worst = max(ok, key=lambda r: r["lyap"][0])
    print(f"B. amplitude system: {len(ok)}/{len(amp)} runs bounded")
    print(f"  largest Lyapunov exponent over all runs: {lmax:.3e}")
    print("  at", {k: worst[k] for k in ("lam", "w_plus_1", "amp", "excited")},
          "spectrum", np.round(worst["lyap"], 5).tolist())
    for wp1 in sorted({r["w_plus_1"] for r in amp}):
        sub = [r["lyap"][0] for r in ok if r["w_plus_1"] == wp1]
        print(f"  w+1={wp1:<8g} max lambda_1 = {max(sub): .3e}   (Omega_v = {1.5*wp1:.2e})")
    json.dump(dict(full=full, amp=amp), open("results_exponential.json", "w"), indent=1)
