"""Full CMB spectrum test of the best candidate histories (final stage, CPU).

Each selected history is turned into a tabulated dark-energy equation of state w(a),
passed to CAMB, and the lensed TT, TE, EE spectra are compared with the Planck 2018
plik_lite high-l likelihood (plik_lite.py). The same likelihood is fitted with flat
LambdaCDM and the difference in chi2 is reported.

STATUS OF TESTING (11 Nov 2026, 2-core sandbox, CAMB 2.0.5):
  TESTED    plik_lite loader: reproduces the reference log-likelihood to 2e-13.
  TESTED    CAMB with a w = -1 table (fluid and PPF) reproduces LambdaCDM spectra to 2e-11.
  TESTED    history -> w(a) table -> CAMB -> chi2 on a candidate from a small NumPy ensemble
            (l1 = 20, alpha = 0.24; field fraction 3e-8 at z = 1100, w0 = -0.994): tuning dN
            alone at omega_b = 0.02236, n_s = 0.965 gives plik_lite chi2 = 585.2 with PPF and
            586.3 with the c_s^2 = 1 fluid, for 613 bins. LambdaCDM at the Planck abstract
            values gives 583.9. A shift of 0.014 in dN raises chi2 by about 560.
  PARTLY    the script end to end, with very short minimisations (see STATUS.md).
  UNTESTED  full-length minimisations (--maxfev default) and runs with many processes.

How the field is represented
  CAMB has scalar-field classes (Quintessence, EarlyQuintessence), but a new potential
  needs a new Fortran class ("To implement a new model, need to define a new derived
  class in Fortran, defining Vofphi", CAMB documentation, dark energy page), so Tian's
  potential cannot be supplied from Python. The field is therefore passed as an
  effective fluid with the history's own w(a):
    DarkEnergyFluid, c_s^2 = 1 (--de fluid). A canonical scalar field has rest-frame
      sound speed 1, and w stays in [-1, 1], so this is the physically matched choice.
      CAMB's fluid equations are singular as w -> -1; it freezes the velocity where
      |1 + w| <= 5e-7 and warns that models approaching -1 closely "can therefore be
      significantly biased on large scales" (same page). The plik_lite range l >= 30
      is not large scales in that sense.
    DarkEnergyPPF (--de ppf, the default). "PPF is not a physical model and just designed
      to allow crossing -1 in an ad hoc smooth way" (same page). It is the default for a
      practical reason found in testing: surviving histories sit within about 1e-3 of
      w = -1 for long stretches, where the fluid equations are stiff; CAMB then prints
      "mismatch in integrated times" warnings, takes about twice as long (11 s against
      6 s per call) and gave chi2 values that moved by tens away from the best fit.
      Physically the dark-energy perturbations vanish in proportion to 1 + w, so for
      such histories the two classes should agree; at the tested best fit they differ
      by 1.2 in chi2. Use --check-other to evaluate the fluid at each best fit; treat a
      difference above about 2 as a sign that the perturbation treatment matters and
      the result for that history is not reliable.
  What is lost: w(a) for z > 50 is a bin average over 0.25 e-folds (86 bins from
  z = 1e11 to z = 50, stored by gpu_ensemble.py as W). Oscillations of w inside a bin
  are averaged, so the field's density history is right but any resonance between
  field oscillations and perturbations is not represented. For candidates that pass
  the ensemble cut (field fraction below 2% for 50 < z < 5000) the field's effect on
  the spectra is small and this is adequate; for a fraction of order 10% near
  recombination neither fluid treatment should be trusted beyond the sign and rough
  size of the effect (my judgement; not validated against a field-level Boltzmann code).

Parameters
  Model:    omega_b, n_s, dN (shift of "today" along the history; fixes omega_m and h),
            and one overall amplitude (analytic) standing for A_s exp(-2 tau)/y_cal^2.
  LCDM:     omega_b, omega_c, H0, n_s and the same amplitude.
  tau = 0.054 fixed (Planck 2018 results VI, arXiv:1807.06209, abstract). With only
  l >= 30 data tau is degenerate with the amplitude, so no tau prior is applied.
  Neutrinos are massless with N_eff = 3.044 in both model and LCDM, to match the
  single radiation fluid of the ensemble code (the Planck baseline has one 0.06 eV state).
  After the minimisation A_s is rescaled by the fitted amplitude and the spectrum is
  recomputed once, so that the lensing smoothing is consistent.
  By default the BAO + supernova chi2 of fit_survivors.py is added to both objectives,
  because dN (model) and Omega_m (LCDM) are otherwise constrained only by the CMB.

Corrections made after review (11 Nov 2026)
  - Default dark-energy class is now the c_s^2 = 1 fluid. For a canonical scalar field the
    perturbations are those of a fluid with rest-frame sound speed 1 and the adiabatic
    sound speed implied by w(a), which is what DarkEnergyFluid with cs2 = 1 evolves; PPF is
    an ad hoc construction. In testing, a candidate whose field reached Omega_phi = 0.84
    near z = 3e5 gave plik_lite chi2 900 (fluid) against 701 (PPF), while the background
    (Omega_de(z) from CAMB) matched the history to a few percent in both, so the gap is the
    perturbation treatment. PPF is still evaluated at the best fit with --check-other.
  - The shift of today is no longer a fitted parameter. The fit uses 100 theta* (CAMB's
    acoustic angle) and solves for the shift that gives it, so the objective is smooth
    (before, a change of 0.014 in the shift moved chi2 by about 560). LambdaCDM is fitted
    with theta* as well (CAMB's thetastar input).
  - CAMB's Fortran warnings ("mismatch in integrated times") are captured for every
    evaluation; their count at the best fit and over the whole fit is reported.
  - The peak field fraction for 5000 < z < 3.7e8, between the recombination and BBN test
    windows, is reported (gap_peak). Where it is large the effective-fluid description of
    the field is least certain; --max-gap restricts the selection.
  - Low-l TT bins (plik_lite.py, low_ell=True) are included by default; --no-low-ell
    turns them off.
  - --mnu sets one massive neutrino of that mass in both model and LambdaCDM (default 0,
    as before); use 0.06 as a systematic check against the Planck baseline.

Usage
  python3 cmb_spectrum.py FIT.json ENSEMBLE.jsonl [--top 50] [--procs 40] [--de fluid|ppf]
          [--check-other] [--no-bao-sn] [--maxfev 160] [--out cmb_spectrum_results.json]
  FIT.json must come from the fit_survivors.py in this folder (it stores cand_index);
  ENSEMBLE.jsonl from the gpu_ensemble.py in this folder (it stores W).
Cost on one core (lmax 2700, lensed): LambdaCDM call 2.5-3.5 s; a call with a tabulated
w(a) about 6 s (PPF) or 11 s (fluid). A history needs roughly 100-170 calls, i.e. about
10-17 core-minutes with PPF.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")           # one CAMB thread per worker process
os.environ.setdefault("GFORTRAN_UNBUFFERED_ALL", "y")   # so CAMB's warnings can be captured per call
import sys
import tempfile

import argparse
import json
import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq

import camb
from camb import dark_energy

import plik_lite
import fit_survivors as F

TAU = 0.054            # Planck 2018 VI abstract, arXiv:1807.06209
AS0 = 2.1e-9           # starting amplitude only; the fitted amplitude rescales it
LMAX = 2700
G = {}                 # per-process globals


def captured(fn, *args, **kw):
    """Run fn with file descriptor 1 redirected; return (result, number of CAMB warnings)."""
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            out = fn(*args, **kw)
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        text = tmp.read().decode(errors="replace")
    return out, text.count("mismatch in integrated times")


def nu_args():
    m = G.get("mnu", 0.0)
    return (dict(mnu=m, nnu=3.044, num_massive_neutrinos=1) if m > 0
            else dict(mnu=0.0, nnu=3.044, num_massive_neutrinos=0))


def init(use_bao_sn, low_ell=True, mnu=0.0):
    G["L"] = plik_lite.PlikLite(low_ell=low_ell)
    G["bao_sn"] = use_bao_sn
    G["mnu"] = mnu
    if use_bao_sn:
        F.load_data()
    p = camb.set_params(H0=67.0, ombh2=0.0224, omch2=0.12, mnu=0.0, nnu=3.044,
                        num_massive_neutrinos=0, omk=0, tau=TAU, As=AS0, ns=0.965)
    r = camb.get_background(p)
    G["wr"] = (r.get_Omega("photon") + r.get_Omega("neutrino")) * 0.67 ** 2   # omega_r, from CAMB


def camb_spectra(H0, wb, wc, ns, As, de=None, thetastar=None):
    """Lensed TT, TE, EE (D_l, muK^2), the number of CAMB warnings, and H0 used."""
    kw = dict(thetastar=thetastar) if thetastar is not None else dict(H0=H0)
    p = camb.set_params(ombh2=wb, omch2=wc, omk=0, tau=TAU, As=As, ns=ns, lmax=LMAX,
                        lens_potential_accuracy=1, **nu_args(), **kw)
    if de is not None:
        p.DarkEnergy = de
    r, nwarn = captured(camb.get_results, p)
    d = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=False)["total"]
    return (d[:, 0], d[:, 3], d[:, 1]), nwarn, p.H0


# ---------------------------------------------------------------- LambdaCDM
def lcdm_objective(x, As=AS0, parts=False):
    wb, wc, th, ns = x                                  # th = 100 theta*
    if not (0.020 < wb < 0.025 and 0.09 < wc < 0.15 and 1.020 < th < 1.060 and 0.9 < ns < 1.03):
        return 1e9
    try:
        spec, nwarn, H0 = camb_spectra(None, wb, wc, ns, As, thetastar=th / 100.0)
        c, s = G["L"].chi2_best_scale(*spec)
    except Exception:
        return 1e9
    extra = 0.0
    if G["bao_sn"]:
        h = H0 / 100.0
        extra = F.chi2_from_E(F.lcdm_E((wb + wc) / h ** 2, G["wr"] / h ** 2))[0]
    return (c, extra, s, H0, nwarn) if parts else c + extra


def fit_lcdm(maxfev):
    x0 = np.array([0.0224, 0.120, 1.0411, 0.965])
    step = np.array([1.5e-4, 1.5e-3, 3e-4, 4e-3])
    simplex = np.vstack([x0] + [x0 + step * np.eye(4)[i] for i in range(4)])
    b = minimize(lcdm_objective, x0, method="Nelder-Mead",
                 options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=2e-5, fatol=0.02))
    c, extra, s, H0, _ = lcdm_objective(b.x, parts=True)
    c2, extra2, s2, H0, nw = lcdm_objective(b.x, As=AS0 * s, parts=True)     # consistent lensing
    return dict(wb=float(b.x[0]), wc=float(b.x[1]), theta100=float(b.x[2]), H0=float(H0), ns=float(b.x[3]),
                As=float(AS0 * s * s2), chi2_cmb=float(c2), chi2_bao_sn=float(extra2),
                chi2=float(c2 + extra2), residual_scale=float(s2), nfev=int(b.nfev),
                converged=bool(b.success), camb_warnings=int(nw))


# ---------------------------------------------------------------- a history
def prepare(cell, cand):
    """Late-time solution and early-time tables for one candidate."""
    N_s, N_eq, N_end = cell["N_snapshot"], cell["N_eq"], cell["N_end"]
    sol = solve_ivp(F.rhs, [N_s, N_end + 0.7], [cand["x"], cand["y"], cand["theta"]],
                    args=(cell["l1"], cell["l2"], cell["alpha"], N_eq),
                    method="DOP853", rtol=1e-10, atol=1e-13, dense_output=True)
    if sol.status != 0:
        raise RuntimeError("late-time integration failed")
    edges = np.asarray(cell["S_edges"])
    return dict(sol=sol, N_s=N_s, N_eq=N_eq, N_end=N_end, Nc=0.5 * (edges[1:] + edges[:-1]),
                edges=edges, W=np.asarray(cand["W"], float), S=np.asarray(cand["S"], float))


def tables(H, dN):
    """w(a) table and background numbers for a history observed at N0 = N_end + dN."""
    N0 = H["N_end"] + dN
    Nl = np.linspace(H["N_s"], N0, 500)
    x, y, _ = H["sol"].sol(Nl)
    wl = (x * x - y * y) / (x * x + y * y)
    Oph0 = x[-1] ** 2 + y[-1] ** 2
    Om0 = (1 - Oph0) * (1 - 3 * F.w_bg(N0, H["N_eq"]))
    early = H["Nc"] < H["N_s"] - 1e-9
    N_all = np.concatenate([H["Nc"][early], Nl])
    w_all = np.clip(np.concatenate([H["W"][early], wl]), -1.0, 1.0)
    a = np.exp(N_all - N0)
    a[-1] = 1.0
    # consistency check: field fraction rebuilt from the w table against the stored S table
    edges = H["edges"]
    k_last = int(np.sum(early))
    lnrho = np.zeros(k_last + 1)                           # at edges[k], relative to N_s
    for k in range(k_last - 1, -1, -1):                    # integrate d ln rho / dN = -3 (1 + w)
        lnrho[k] = lnrho[k + 1] + 3 * (1 + H["W"][k]) * (min(edges[k + 1], H["N_s"]) - edges[k])
    xs, ys, _ = H["sol"].sol(H["N_s"])
    Os = xs * xs + ys * ys
    fluid = lambda N: np.exp(-3 * N) * (1 + np.exp(H["N_eq"] - N))
    r_s = Os / (1 - Os) * fluid(H["N_s"])
    Ne = edges[:k_last + 1]
    r = r_s * np.exp(lnrho) / fluid(Ne)
    O_rec = r / (1 + r)
    O_tab = 1 - H["S"][:k_last] ** 2
    z_e = np.exp(N0 - Ne) - 1
    win = (z_e > 50) & (z_e < 5000)
    diag = dict(Ophi_rec_z1100=float(np.interp(np.log(1101.0), np.log1p(z_e)[::-1], O_rec[::-1])),
                Ophi_max_50_5000=float(O_rec[win].max()) if win.any() else None,
                max_abs_diff_Ophi_vs_S_table=float(np.max(np.abs(0.5 * (O_rec[1:] + O_rec[:-1]) - O_tab))))
    return a, w_all, N0, Om0, Oph0, diag


def E_late(H, N0):
    N = N0 - np.log1p(F.ZGRID)
    x, y, _ = H["sol"].sol(N)
    Om = (1 - x * x - y * y) * (1 - 3 * F.w_bg(N, H["N_eq"]))
    return np.sqrt(Om[0] / Om * (1 + F.ZGRID) ** 3)


def make_de(kind, a, w):
    de = dark_energy.DarkEnergyPPF() if kind == "ppf" else dark_energy.DarkEnergyFluid()
    de.set_w_a_table(a, w)
    if kind != "ppf":
        de.cs2 = 1.0
    return de


def background_theta(H, wb, dN):
    """CAMB's 100 theta* for the history observed at N_end + dN."""
    a, w, N0, Om0, Oph0, _ = tables(H, dN)
    wm = G["wr"] * np.exp(N0 - H["N_eq"])
    h = np.sqrt(wm / Om0)
    p = camb.set_params(H0=100 * h, ombh2=wb, omch2=wm - wb, omk=0, tau=TAU, As=AS0, ns=0.965, **nu_args())
    p.DarkEnergy = make_de("ppf", a, w)                 # background is identical for both classes
    r, _ = captured(camb.get_background, p)
    return float(r.get_derived_params()["thetastar"])


def solve_dN(H, wb, theta, guess):
    """Shift of today giving 100 theta* = theta (cached; about 3 background calls)."""
    key = (round(wb, 7), round(theta, 7))
    cache = H.setdefault("dN_cache", {})
    if key in cache:
        return cache[key]
    g = lambda d: background_theta(H, wb, d) - theta
    d0 = H.get("last_dN", guess)
    f0 = g(d0)
    d1 = d0 - f0 / 4.5                                  # d(100 theta*)/d(dN) is about 4.5 in tests
    f1 = g(d1)
    for _ in range(4):                                  # secant steps
        if abs(f1) < 2e-6 or f1 == f0:
            break
        d0, f0, d1 = d1, f1, d1 - f1 * (d1 - d0) / (f1 - f0)
        f1 = g(d1)
    if abs(f1) > 2e-6:                                  # fall back to bracketing
        lo, hi = d1 - 0.01, d1 + 0.01
        while g(lo) * g(hi) > 0 and hi - lo < 0.4:
            lo, hi = lo - 0.02, hi + 0.02
        d1 = brentq(g, lo, hi, xtol=1e-8)
    cache[key] = H["last_dN"] = float(d1)
    return float(d1)


def gap_peak(H, N0):
    """Largest field fraction (from the stored S table) for 5000 < z < 3.7e8."""
    z = np.exp(N0 - H["Nc"]) - 1
    sel = (z > 5e3) & (z < 3.7e8)
    return float((1 - H["S"][sel] ** 2).max()) if sel.any() else None


def model_eval(H, x, kind, As=AS0):
    wb, ns, theta = x
    if not (0.020 < wb < 0.025 and 0.9 < ns < 1.03 and 1.000 < theta < 1.080):
        return None
    dN = solve_dN(H, wb, theta, H["dN0"])
    if abs(dN - H["dN0"]) > 0.3:
        return None
    a, w, N0, Om0, Oph0, diag = tables(H, dN)
    wm = G["wr"] * np.exp(N0 - H["N_eq"])                  # rho_m / rho_r today = exp(N0 - N_eq)
    h = np.sqrt(wm / Om0)
    wc = wm - wb
    if not (0.020 < wb < 0.025 and 0.9 < ns < 1.03 and wc > 0.05 and 0.4 < h < 1.0):
        return None
    spec, nwarn, _ = camb_spectra(100 * h, wb, wc, ns, As, make_de(kind, a, w))
    c, s = G["L"].chi2_best_scale(*spec)
    H["warnings_total"] = H.get("warnings_total", 0) + nwarn
    extra = F.chi2_from_E(E_late(H, N0))[0] if G["bao_sn"] else 0.0
    return dict(chi2_cmb=float(c), chi2_bao_sn=float(extra), scale=float(s), h=float(h), wm=float(wm),
                wc=float(wc), Om0=float(Om0), Ophi0=float(Oph0), w0=float(w[-1]), dN=float(dN),
                camb_warnings=int(nwarn), gap_peak=gap_peak(H, N0), **diag)


def fit_candidate(args):
    cell, cand, dN0, kind, maxfev, check_other, tag = args
    try:
        H = prepare(cell, cand)
        H["dN0"] = dN0

        def f(x):
            try:
                r = model_eval(H, x, kind)
            except Exception:
                return 1e9
            return 1e9 if r is None else r["chi2_cmb"] + r["chi2_bao_sn"]

        x0 = np.array([0.02236, 0.965, 1.0411])
        step = np.array([1.0e-4, 3e-3, 3e-4])
        simplex = np.vstack([x0] + [x0 + step * np.eye(3)[i] for i in range(3)])
        b = minimize(f, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=2e-5, fatol=0.02))
        r = model_eval(H, b.x, kind)
        if r is None:
            return dict(tag=tag, error="best point outside the allowed range")
        r2 = model_eval(H, b.x, kind, As=AS0 * r["scale"])  # consistent lensing
        out = dict(tag=tag, wb=float(b.x[0]), ns=float(b.x[1]), theta100=float(b.x[2]),
                   As=float(AS0 * r["scale"] * r2["scale"]), nfev=int(b.nfev), converged=bool(b.success),
                   chi2=float(r2["chi2_cmb"] + r2["chi2_bao_sn"]), de=kind,
                   warnings_during_fit=int(H.get("warnings_total", 0)),
                   **{k: v for k, v in r2.items() if k != "scale"}, residual_scale=r2["scale"])
        if check_other:
            other = "fluid" if kind == "ppf" else "ppf"
            r3 = model_eval(H, b.x, other, As=AS0 * r["scale"])
            out["chi2_cmb_other_de"] = None if r3 is None else r3["chi2_cmb"]
            out["other_de"] = other
        d_other = abs(out.get("chi2_cmb_other_de", out["chi2_cmb"]) - out["chi2_cmb"])
        out["reliable"] = bool(out["camb_warnings"] == 0 and d_other < 2.0)
        return out
    except Exception as e:
        return dict(tag=tag, error=repr(e)[:300])


def cand_gap_peak(cell, cand, dN):
    e = np.asarray(cell["S_edges"])
    z = np.exp(cell["N_end"] + dN - 0.5 * (e[1:] + e[:-1])) - 1
    sel = (z > 5e3) & (z < 3.7e8)
    return float((1 - np.asarray(cand["S"]) ** 2)[sel].max())


def select(fit_path, ens_path, top, max_gap=None):
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(ens_path))}
    rows = []
    for fc in json.load(open(fit_path))["cells"]:
        key = (fc["l1"], fc["l2"], fc["alpha"])
        if key not in cells:
            continue
        for r in fc.get("fits", []):
            if r.get("cand_index", -1) < 0:
                raise SystemExit("FIT.json has no cand_index: rerun fit_survivors.py from this folder")
            rows.append((r["chi2"], key, r["cand_index"], r["dN"]))
    rows.sort(key=lambda t: t[0])
    picked = []
    for chi2, key, i, dN in rows:
        if len(picked) >= top:
            break
        cell = cells[key]
        cand = cell["candidates"][i]
        if "W" not in cand:
            raise SystemExit("ENSEMBLE.jsonl has no W table: rerun gpu_ensemble.py from this folder")
        if max_gap is not None and cand_gap_peak(cell, cand, dN) > max_gap:
            continue
        meta = {k: cell[k] for k in ("l1", "l2", "alpha", "N_snapshot", "N_eq", "N_end", "S_edges")}
        picked.append((meta, cand, dN, dict(l1=key[0], l2=key[1], alpha=key[2], cand_index=i,
                                            fast_fit_chi2=chi2)))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit")
    ap.add_argument("ensemble")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--de", choices=("fluid", "ppf"), default="fluid")
    ap.add_argument("--max-gap", type=float, default=None,
                    help="only candidates whose field fraction stays below this for 5000 < z < 3.7e8")
    ap.add_argument("--no-low-ell", action="store_true", help="drop the two low-l TT bins")
    ap.add_argument("--mnu", type=float, default=0.0, help="one massive neutrino of this mass (eV), model and LCDM")
    ap.add_argument("--check-other", action="store_true",
                    help="also evaluate the other dark-energy class at the best fit")
    ap.add_argument("--no-bao-sn", action="store_true")
    ap.add_argument("--maxfev", type=int, default=160)
    ap.add_argument("--out", default="cmb_spectrum_results.json")
    a = ap.parse_args()
    use = not a.no_bao_sn
    picked = select(a.fit, a.ensemble, a.top, a.max_gap)
    low = not a.no_low_ell
    print(f"{len(picked)} candidate(s) selected; dark energy class: {a.de}; BAO+SN included: {use}; "
          f"low-l TT: {low}; m_nu = {a.mnu} eV; max gap peak: {a.max_gap}", flush=True)
    init(use, low, a.mnu)
    ref = fit_lcdm(a.maxfev)
    print(f"LCDM: chi2 = {ref['chi2']:.2f} (plik_lite {ref['chi2_cmb']:.2f} for {plik_lite.NBIN + (2 if low else 0)} bins, "
          f"BAO+SN {ref['chi2_bao_sn']:.2f}); omega_b {ref['wb']:.5f}, omega_c {ref['wc']:.4f}, "
          f"H0 {ref['H0']:.2f}, n_s {ref['ns']:.4f}; evaluations {ref['nfev']}, converged {ref['converged']}, "
          f"CAMB warnings {ref['camb_warnings']}",
          flush=True)
    jobs = [(m, c, dN, a.de, a.maxfev, a.check_other, t) for m, c, dN, t in picked]
    # 'spawn', not 'fork': CAMB's OpenMP runtime is already initialised in this process and a
    # forked child can deadlock on it (observed in testing).
    with mp.get_context("spawn").Pool(a.procs, initializer=init, initargs=(use, low, a.mnu)) as pool:
        res = pool.map(fit_candidate, jobs, chunksize=1)
    for r in res:
        t = r["tag"]
        head = f"l1={t['l1']} alpha={t['alpha']} cand {t['cand_index']}: "
        if "error" in r:
            print(head + "ERROR " + r["error"], flush=True)
            continue
        r["delta_chi2"] = r["chi2"] - ref["chi2"]
        line = (head + f"delta chi2 = {r['delta_chi2']:+.2f} (plik_lite {r['chi2_cmb'] - ref['chi2_cmb']:+.2f}, "
                f"BAO+SN {r['chi2_bao_sn'] - ref['chi2_bao_sn']:+.2f}); h {r['h']:.4f}, omega_b {r['wb']:.5f}, "
                f"n_s {r['ns']:.4f}, w0 {r['w0']:.3f}; field fraction at z=1100 {r['Ophi_rec_z1100']:.2e}, "
                f"peak 50<z<5000 {r['Ophi_max_50_5000']:.2e}, peak 5000<z<3.7e8 {r['gap_peak']:.2e}; "
                f"table check {r['max_abs_diff_Ophi_vs_S_table']:.1e}; CAMB warnings at best fit {r['camb_warnings']}, "
                f"during fit {r['warnings_during_fit']}; converged {r['converged']}; reliable {r['reliable']}")
        if r.get("chi2_cmb_other_de") is not None:
            line += f"; {r['other_de']} gives plik_lite {r['chi2_cmb_other_de'] - r['chi2_cmb']:+.2f}"
        print(line, flush=True)
    json.dump(dict(lcdm=ref, tau=TAU, de=a.de, bao_sn=use, low_ell=low, mnu=a.mnu, max_gap=a.max_gap,
                   n_bins=plik_lite.NBIN + (2 if low else 0), results=res),
              open(a.out, "w"), indent=1)
    ok = [r for r in res if "error" not in r]
    if ok:
        d = np.array([r["delta_chi2"] for r in ok])
        rel = np.array([r["reliable"] for r in ok])
        print(f"summary: {len(ok)} fitted, {rel.sum()} reliable (no CAMB warning at the best fit and fluid/PPF "
              f"within 2); delta chi2 best {d.min():+.2f}, median {np.median(d):+.2f}; within +4 of LCDM: {(d < 4).sum()}; "
              f"model has 3 fitted parameters + amplitude, LCDM 4 + amplitude")
        if rel.any():
            print(f"         reliable only: delta chi2 best {d[rel].min():+.2f}, median {np.median(d[rel]):+.2f}")


if __name__ == "__main__":
    main()
