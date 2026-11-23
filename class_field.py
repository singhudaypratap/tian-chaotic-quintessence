"""Field-level CMB test: evolve Tian's scalar field itself in CLASS.

Why: cmb_spectrum.py passes the field to CAMB as an effective fluid with the history's
w(a). For histories with a large field episode before recombination (common: see
RESULTS.md section 15) the c_s^2 = 1 fluid and PPF treatments disagree by hundreds in
chi2. Here the Klein-Gordon equation for the field and its perturbations is solved by
CLASS, with Tian's potential in CLASS's scalar-field module, so no fluid approximation
is made.

Requires the modified CLASS ("classy" built from class_tian/, see CLASS_TIAN.md):
  V(phi) = m V0 exp[-(l1+l2) phi/2 - alpha (l1-l2) sin(phi/alpha)/2]
  scf_parameters = [l1, l2, alpha, m, V0, phi_ini, phi_prime_ini], scf_tuning_index = 3

Mapping a stored history to CLASS (derived; kappa = 1, phi in reduced Planck mass):
  state (x, y, theta) at z_s = 1e9 (stored by gpu_ensemble.py as "early");
  CLASS starts at a_s = exp(N_state_early - N0), today N0 = N_end + dN;
  omega_m = omega_r exp(N0 - N_eq)   (rho_m/rho_r today; omega_r from CLASS itself);
  h from Omega_m = 1 - Omega_phi(N0) - Omega_r;
  H_s^2 = H0^2 (Omega_r a_s^-4 + Omega_m a_s^-3) / (1 - x^2 - y^2);
  phi_ini = alpha (theta mod 2 pi)   (the factor exp(-(l1+l2) alpha pi k) from removing
             whole periods is absorbed into V0, so V is unchanged);
  phi'_ini = a_s sqrt(6) H_s x       (conformal-time derivative; x = phi_dot/(sqrt6 H));
  V0 = 3 H_s^2 y^2 / [exp(-(l1+l2) phi_ini/2) exp(-alpha (l1-l2) sin(theta)/2)].
CLASS then adjusts m so that the budget closes today (Omega_scf fills). With the
mapping above m should come out within about 1e-6 of 1; it is reported.

Checks made by this script (run --check-only for these alone):
  1. CLASS's Omega_scf(z) against the history recomputed from the same state with the
     pipeline's own equations (tian.py), at 10 redshifts from 1e8 to 0;
  2. the shooting parameter m.
The field is chaotic (Lyapunov exponent about 0.14 to 0.23 per e-fold), so small
differences between the two integrators grow by a factor of order e^(0.2 x 21) ~ 70
between z = 1e9 and today. Agreement to better than about 1e-3 in Omega_scf is the
target; larger differences mean CLASS is following a different history.

Usage
  python3 class_field.py FIT.json ENSEMBLE.jsonl [--top 5] [--max-gap X] [--procs 8]
         [--maxfev 120] [--check-only] [--out class_field_results.json]
Same selection as cmb_spectrum.py (best candidates by the fast fit). The fit varies
omega_b, n_s and 100 theta_s (CLASS's sound-horizon angle), solving dN from theta_s;
the amplitude is analytic as in plik_lite.py. tau = 0.054 fixed (as cmb_spectrum.py).
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq

import plik_lite
import fit_survivors as F
import tian

# tau is fitted with a Gaussian prior (Planck 2018 low-l EE), in both the model and LCDM fits,
# because plik_lite + low-l TT alone do not fix it and a fixed tau could bias n_s.
TAU_PRIOR = (0.0544, 0.0073)
# CLASS precision presets; "high" is used to check that results are converged in CLASS's settings
PRECISION = {"default": {},
             "high": {"perturbations_sampling_stepsize": 0.05, "tol_perturbations_integration": 1e-6,
                      "l_linstep": 20, "l_logstep": 1.03, "accurate_lensing": 1, "k_max_tau0_over_l_max": 3.0}}

TAU = 0.054                      # Planck 2018 VI abstract, arXiv:1807.06209 (as cmb_spectrum.py)
T_CMB = 2.7255
C_KMS = 299792.458
LMAX = 2508
G = {}


def init(low_ell=True, bao_sn=True, precision="default", rounds=3):
    G["precision"], G["rounds"] = precision, rounds
    from classy import Class
    G["Class"] = Class
    G["L"] = plik_lite.PlikLite(low_ell=low_ell)
    G["bao_sn"] = bao_sn
    if bao_sn:
        F.load_data()
    c = Class()
    c.set({"h": 0.67, "omega_b": 0.0224, "omega_cdm": 0.12, "N_ur": 3.044, "N_ncdm": 0, "T_cmb": T_CMB})
    c.compute()
    G["wr"] = c.Omega_r() * c.h() ** 2                  # photons + massless neutrinos
    c.struct_cleanup()


# ------------------------------------------------------------------ a history
def load(cell, cand):
    l1, l2, al = cell["l1"], cell["l2"], cell["alpha"]
    N_s, N_eq, N_end = cell["N_state_early"], cell["N_eq"], cell["N_end"]
    s0 = tian.with_u(cand["early"])                 # (x, y, theta, u = ln Omega_bg); old files lack u
    x, y, th = s0[:3]
    # x and y can be ~1e-15 at z = 1e9 for late-thawing histories, so an absolute tolerance of
    # 1e-14 on them let the replay drift (found 15 Nov: up to 2.5e-4 in (x, y) by z = 50). Their
    # tolerance is now relative only; theta and u keep an absolute floor (both are O(1) or exact).
    sol = solve_ivp(tian.rhs_fullu, [N_s, N_end + 0.7], s0, args=(l1, l2, al, N_eq),
                    method="DOP853", rtol=1e-11, atol=[1e-300, 1e-300, 1e-14, 1e-16], dense_output=True)
    if sol.status != 0:
        raise RuntimeError("reference integration failed")
    return dict(l1=l1, l2=l2, al=al, N_s=N_s, N_eq=N_eq, N_end=N_end, state=tuple(s0), sol=sol)


def background_numbers(H, dN):
    N0 = H["N_end"] + dN
    x0, y0, _, u0 = H["sol"].sol(N0)
    Oph0 = x0 * x0 + y0 * y0
    wm = G["wr"] * np.exp(N0 - H["N_eq"])
    # Omega_r + Omega_m + Omega_phi = 1 today, with Omega_r/Omega_m = exp(N_eq - N0)
    Om0 = np.exp(u0) / (1 + np.exp(H["N_eq"] - N0))
    h = np.sqrt(wm / Om0)
    return N0, wm, h, float(Oph0)


def class_input(H, dN, wb):
    N0, wm, h, Oph0 = background_numbers(H, dN)
    x, y, th = H["state"][:3]
    Ob_s = np.exp(H["state"][3]) if len(H["state"]) > 3 else 1 - x * x - y * y
    l1, l2, al = H["l1"], H["l2"], H["al"]
    a_s = np.exp(H["N_s"] - N0)
    H0c = h * 100.0 / C_KMS                                     # Mpc^-1
    Or, Om = G["wr"] / h ** 2, wm / h ** 2
    Hs2 = H0c ** 2 * (Or * a_s ** -4 + Om * a_s ** -3) / Ob_s
    thm = float(np.mod(th, 2 * np.pi))
    phi = al * thm
    dphi = a_s * np.sqrt(6.0) * np.sqrt(Hs2) * x
    V0 = 3 * Hs2 * y * y / (np.exp(-0.5 * (l1 + l2) * phi) * np.exp(-0.5 * al * (l1 - l2) * np.sin(thm)))
    p = {"omega_b": float(wb), "omega_cdm": float(wm - wb), "h": float(h), "T_cmb": T_CMB, "N_ur": 3.044, "N_ncdm": 0,
         "Omega_Lambda": 0.0, "Omega_fld": 0.0, "Omega_scf": -1.0,
         "scf_parameters": ",".join(repr(float(v)) for v in (l1, l2, al, 1.0, V0, phi, dphi)),
         "attractor_ic_scf": "no", "scf_tuning_index": 3,
         "a_ini_over_a_today_default": float(a_s),
         "tol_initial_Omega_r": 0.05,                 # the field may hold up to a few percent at z_s
         "tol_background_integration": 1e-12,
         "tol_shooting_deltaF": 1e-9, "tol_shooting_deltax": 1e-14, "tol_shooting_deltax_rel": 1e-14}
    return p, dict(N0=N0, wm=wm, h=h, Ophi0=Oph0, a_s=a_s)


def compare_background(H, dN, wb=0.02236):
    """Run CLASS's background only and compare Omega_scf(z) with the history."""
    p, info = class_input(H, dN, wb)
    c = G["Class"]()
    c.set(p)
    c.compute(["background"])
    bg = c.get_background()
    z = bg["z"]
    O_class = bg["(.)rho_scf"] / bg["H [1/Mpc]"] ** 2
    zs = np.array([1e8, 1e7, 1e6, 3e5, 1e5, 3e4, 1e4, 1e3, 10.0, 1.0, 0.0])
    oc = np.interp(np.log1p(zs), np.log1p(z[::-1]), O_class[::-1])
    N = info["N0"] - np.log1p(zs)
    xh, yh, _, _ = H["sol"].sol(N)
    oh = xh * xh + yh * yh
    c.struct_cleanup()
    return zs, oc, oh, info


# ------------------------------------------------------------------ spectra and fit
def spectra(p, As):
    c = G["Class"]()
    q = dict(p)
    q.update({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": LMAX + 200,
              "A_s": As, "n_s": q.pop("n_s"), "tau_reio": q.pop("tau_reio", TAU)})
    q.update(PRECISION[G.get("precision", "default")])
    c.set(q)
    c.compute()
    cl = c.lensed_cl(LMAX)
    th = c.theta_s_100()
    c.struct_cleanup()
    l = cl["ell"]
    f = l * (l + 1) / (2 * np.pi) * (T_CMB * 1e6) ** 2
    pad = lambda arr: np.concatenate([arr * f, np.zeros(max(0, plik_lite.LMAX_NEEDED + 1 - len(l)))])
    return (pad(cl["tt"]), pad(cl["te"]), pad(cl["ee"])), th


def theta_of(H, dN, wb):
    p, _ = class_input(H, dN, wb)
    c = G["Class"]()
    c.set(p)
    c.compute(["thermodynamics"])
    t = c.theta_s_100()
    c.struct_cleanup()
    return t


def solve_dN(H, wb, theta):
    key = (round(wb, 7), round(theta, 7))
    cache = H.setdefault("cache", {})
    if key in cache:
        return cache[key]
    g = lambda d: theta_of(H, d, wb) - theta
    d0 = H.get("last", H["dN0"])
    f0 = g(d0)
    d1 = d0 - f0 / 4.5
    f1 = g(d1)
    for _ in range(4):
        if abs(f1) < 2e-6 or f1 == f0:
            break
        d0, f0, d1 = d1, f1, d1 - f1 * (d1 - d0) / (f1 - f0)
        f1 = g(d1)
    if abs(f1) > 2e-6:
        lo, hi = d1 - 0.01, d1 + 0.01
        while g(lo) * g(hi) > 0 and hi - lo < 0.4:
            lo, hi = lo - 0.02, hi + 0.02
        d1 = brentq(g, lo, hi, xtol=1e-8)
    cache[key] = H["last"] = float(d1)
    return float(d1)


def evaluate(H, x, As=2.1e-9):
    wb, ns, theta, tau = (list(x) + [TAU_PRIOR[0]])[:4]
    if not (0.020 < wb < 0.025 and 0.85 < ns < 1.05 and 1.000 < theta < 1.080 and 0.02 < tau < 0.10):
        return None
    dN = solve_dN(H, wb, theta)
    p, info = class_input(H, dN, wb)
    p["n_s"] = ns
    p["tau_reio"] = tau
    spec, th = spectra(p, As)
    c, s = G["L"].chi2_best_scale(*spec)
    extra = 0.0
    if G["bao_sn"]:
        N = info["N0"] - np.log1p(F.ZGRID)
        xx, yy, _, _ = H["sol"].sol(N)
        Om = (1 - xx * xx - yy * yy) * (1 - 3 * tian.w_bg(N, H["N_eq"]))
        extra = F.chi2_from_E(np.sqrt(Om[0] / Om * (1 + F.ZGRID) ** 3))[0]
    prior = ((tau - TAU_PRIOR[0]) / TAU_PRIOR[1]) ** 2
    return dict(chi2_cmb=float(c), chi2_bao_sn=float(extra), chi2_tau=float(prior), tau=float(tau),
                scale=float(s), dN=float(dN), **info)


def fit_lcdm(maxfev):
    Class = G["Class"]

    def f(x, As=2.1e-9, parts=False):
        wb, wc, th, ns, tau = x
        if not (0.020 < wb < 0.025 and 0.09 < wc < 0.15 and 1.020 < th < 1.060 and 0.85 < ns < 1.05
                and 0.02 < tau < 0.10):
            return 1e9
        c = Class()
        c.set({"omega_b": wb, "omega_cdm": wc, "100*theta_s": th, "n_s": ns, "A_s": As, "tau_reio": tau,
               "N_ur": 3.044, "N_ncdm": 0, "T_cmb": T_CMB, "output": "tCl,pCl,lCl", "lensing": "yes",
               "l_max_scalars": LMAX + 200, **PRECISION[G.get("precision", "default")]})
        try:
            c.compute()
        except Exception:
            return 1e9
        cl, h = c.lensed_cl(LMAX), c.h()
        c.struct_cleanup()
        l = cl["ell"]
        fac = l * (l + 1) / (2 * np.pi) * (T_CMB * 1e6) ** 2
        pad = lambda arr: np.concatenate([arr * fac, np.zeros(max(0, plik_lite.LMAX_NEEDED + 1 - len(l)))])
        chi, s = G["L"].chi2_best_scale(pad(cl["tt"]), pad(cl["te"]), pad(cl["ee"]))
        extra = F.chi2_from_E(F.lcdm_E((wb + wc) / h ** 2, G["wr"] / h ** 2))[0] if G["bao_sn"] else 0.0
        prior = ((tau - TAU_PRIOR[0]) / TAU_PRIOR[1]) ** 2
        return (chi, extra, s, h, prior) if parts else chi + extra + prior

    x0 = np.array([0.0224, 0.120, 1.0420, 0.965, TAU_PRIOR[0]])
    step = np.array([1.5e-4, 1.5e-3, 3e-4, 4e-3, 4e-3])
    b, nfev = restarted_nm(f, x0, step, maxfev, G.get("rounds", 3))
    c1, e1, s1, h, p1 = f(b.x, parts=True)
    c2, e2, s2, h, p2 = f(b.x, As=2.1e-9 * s1, parts=True)
    return dict(wb=float(b.x[0]), wc=float(b.x[1]), theta_s100=float(b.x[2]), ns=float(b.x[3]), tau=float(b.x[4]),
                h=float(h), x=[float(v) for v in b.x], As=float(2.1e-9 * s1 * s2),
                chi2_cmb=float(c2), chi2_bao_sn=float(e2), chi2_tau=float(p2), chi2=float(c2 + e2 + p2), nfev=nfev,
                converged=bool(b.success), precision=G.get("precision", "default"))


def restarted_nm(f, x0, step, maxfev, rounds):
    """Nelder-Mead, restarted from the best point with a fresh simplex until it converges
    (a stalled simplex is the usual reason a fit stops at maxfev)."""
    n, total, b = len(x0), 0, None
    for k in range(max(1, rounds)):
        sc = 1.0 if k == 0 else 0.5
        simplex = np.vstack([x0] + [x0 + sc * step * np.eye(n)[i] for i in range(n)])
        b = minimize(f, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=2e-5, fatol=0.02))
        total += int(b.nfev)
        if b.success and k > 0 and abs(f_prev - b.fun) < 0.05:
            break
        x0, f_prev = b.x, b.fun
    return b, total


def fit_candidate(args):
    cell, cand, dN0, maxfev, tag = args[:5]
    start = args[5] if len(args) > 5 else None
    camb = args[6] if len(args) > 6 else None        # CAMB best fit for the same history (a second start)
    try:
        H = load(cell, cand)
        H["dN0"] = dN0
        zs, oc, oh, info = compare_background(H, dN0)
        check = dict(z=zs.tolist(), Omega_class=oc.tolist(), Omega_history=oh.tolist(),
                     max_abs_diff=float(np.max(np.abs(oc - oh))))

        def f(x):
            try:
                r = evaluate(H, x)
            except Exception:
                return 1e9
            return 1e9 if r is None else r["chi2_cmb"] + r["chi2_bao_sn"] + r["chi2_tau"]

        x0 = np.array([0.02236, 0.965, 1.0420, TAU_PRIOR[0]])
        step = np.array([1.0e-4, 3e-3, 3e-4, 4e-3])
        if start is not None:                       # refine from an earlier best fit
            x0 = np.array((list(start["x"]) + [TAU_PRIOR[0]])[:4], float)
            H["dN0"] = start.get("dN", dN0)
        b, nfev = restarted_nm(f, x0, step, maxfev, G.get("rounds", 3))
        seeded_from = "default" if start is None else "restart"
        if camb is not None:
            # Second start at CAMB's best fit, mapped onto CLASS's theta_s for the same history:
            # a simplex fit can stop in a worse valley, and the better of the two is kept.
            try:
                x1 = np.array([camb["wb"], camb["ns"], theta_of(H, camb["dN"], camb["wb"]), TAU_PRIOR[0]])
                b1, n1 = restarted_nm(f, x1, step, maxfev, G.get("rounds", 3))
                nfev += n1
                if b1.fun < b.fun - 1e-6:
                    b, seeded_from = b1, "camb"
            except Exception:
                pass
        r = evaluate(H, b.x)
        r2 = evaluate(H, b.x, As=2.1e-9 * r["scale"])
        x_s = H["state"]
        return dict(tag=tag, wb=float(b.x[0]), ns=float(b.x[1]), theta_s100=float(b.x[2]), x=[float(v) for v in b.x],
                    Ophi_at_start=float(x_s[0] ** 2 + x_s[1] ** 2), precision=G.get("precision", "default"),
                    best_start=seeded_from,
                    chi2=float(r2["chi2_cmb"] + r2["chi2_bao_sn"] + r2["chi2_tau"]), nfev=nfev, converged=bool(b.success),
                    background_check=check, **{k: v for k, v in r2.items() if k != "scale"})
    except Exception as e:
        return dict(tag=tag, error=repr(e)[:400])


def main():
    import cmb_spectrum as CS                     # selection logic only (no CAMB call is made)
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("ensemble")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-gap", type=float, default=None)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--maxfev", type=int, default=120)
    ap.add_argument("--no-low-ell", action="store_true")
    ap.add_argument("--no-bao-sn", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="only compare CLASS's background with the history")
    ap.add_argument("--restart", default=None,
                    help="results file of an earlier run: refine each candidate from its best fit")
    ap.add_argument("--out", default="class_field_results.json")
    ap.add_argument("--precision", choices=list(PRECISION), default="default")
    ap.add_argument("--seed-from", default=None,
                    help="cmb_spectrum.py results for the same candidates: their best fits are a second start")
    ap.add_argument("--rounds", type=int, default=3, help="Nelder-Mead restarts until converged")
    a = ap.parse_args()
    picked = CS.select(a.fit, a.ensemble, a.top, a.max_gap)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(a.ensemble))}
    jobs = []
    for meta, cand, dN, tag in picked:
        cell = cells[(tag["l1"], tag["l2"], tag["alpha"])]
        if "early" not in cand:
            raise SystemExit("ENSEMBLE.jsonl has no early state: rerun the current gpu_ensemble.py")
        meta = dict(meta, N_state_early=cell["N_state_early"])
        jobs.append((meta, cand, dN, a.maxfev, tag))
    if a.restart:                    # key by cell and index: cand_index repeats across cells
        kk = lambda t: (t["l1"], t["l2"], t["alpha"], t["cand_index"])
        prev = {kk(r["tag"]): r for r in json.load(open(a.restart))["results"] if "error" not in r}
        jobs = [j + ((dict(x=prev[kk(j[4])].get("x", [prev[kk(j[4])]["wb"], prev[kk(j[4])]["ns"],
                                                    prev[kk(j[4])]["theta_s100"]]),
                           dN=prev[kk(j[4])]["dN"]) if kk(j[4]) in prev else None),) for j in jobs]
    if a.seed_from:
        kk = lambda t: (t["l1"], t["l2"], t["alpha"], t["cand_index"])
        cr = {kk(r["tag"]): dict(wb=r["wb"], ns=r["ns"], dN=r["dN"])
              for r in json.load(open(a.seed_from))["results"] if "error" not in r and "dN" in r}
        jobs = [(j + (None,) if len(j) == 5 else j) + (cr.get(kk(j[4])),) for j in jobs]
        print(f"second start from CAMB best fits for {sum(kk(j[4]) in cr for j in jobs)} of {len(jobs)} candidates")
    init(not a.no_low_ell, not a.no_bao_sn, a.precision, a.rounds)
    print(f"omega_r from CLASS = {G['wr']:.6e}; {len(jobs)} candidate(s)", flush=True)
    if a.check_only:
        for meta, cand, dN, _, tag in jobs:
            H = load(meta, cand)
            zs, oc, oh, info = compare_background(H, dN)
            print(f"cand {tag['cand_index']}: h {info['h']:.4f}, z_start {1/info['a_s']-1:.2e}; Omega_phi CLASS / history:")
            for z, p, q in zip(zs, oc, oh):
                print(f"    z = {z:9.2e}:  {p:.4e}  {q:.4e}   diff {p - q:+.1e}")
        return
    ref = json.load(open(a.restart))["lcdm"] if a.restart else fit_lcdm(a.maxfev)
    print(f"LCDM (CLASS): chi2 = {ref['chi2']:.2f} (CMB {ref['chi2_cmb']:.2f}, BAO+SN {ref['chi2_bao_sn']:.2f}); "
          f"omega_b {ref['wb']:.5f}, omega_c {ref['wc']:.4f}, h {ref['h']:.4f}, n_s {ref['ns']:.4f}; "
          f"evaluations {ref['nfev']}, converged {ref['converged']}", flush=True)
    with mp.get_context("spawn").Pool(a.procs, initializer=init, initargs=(not a.no_low_ell, not a.no_bao_sn, a.precision, a.rounds)) as pool:
        res = pool.map(fit_candidate, jobs, chunksize=1)
    for r in res:
        t = r["tag"]
        if "error" in r:
            print(f"cand {t['cand_index']}: ERROR {r['error']}")
            continue
        r["delta_chi2"] = r["chi2"] - ref["chi2"]
        print(f"l1={t['l1']} alpha={t['alpha']} cand {t['cand_index']}: delta chi2 = {r['delta_chi2']:+.2f} "
              f"(CMB {r['chi2_cmb'] - ref['chi2_cmb']:+.2f}, BAO+SN {r['chi2_bao_sn'] - ref['chi2_bao_sn']:+.2f}, "
              f"tau prior {r['chi2_tau'] - ref.get('chi2_tau', 0):+.2f}); "
              f"h {r['h']:.4f}, omega_b {r['wb']:.5f}, n_s {r['ns']:.4f}, tau {r['tau']:.4f}; "
              f"Omega_phi at z_start {r['Ophi_at_start']:.1e}; background check max |dOmega| "
              f"{r['background_check']['max_abs_diff']:.1e}; converged {r['converged']}; best start {r.get('best_start')}", flush=True)
    json.dump(dict(lcdm=ref, tau=TAU, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
