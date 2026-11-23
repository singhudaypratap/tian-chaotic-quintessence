"""Background-level fit of surviving histories to DESI DR2 BAO and Pantheon+.

Input: the .jsonl written by gpu_ensemble.py. Each candidate history is stored
as its state at z = 50; the last 3.9 e-folds are recomputed here with an
adaptive integrator.

Free parameters per history
  dN : shift of "today" along the history (equivalent to the matter density
       omega_m, since it moves matter-radiation equality), |dN| <= 0.6
  A  : c / (H0 r_d), found analytically (BAO used without a sound-horizon prior)
  M  : supernova absolute magnitude, marginalised analytically
Reference: flat LambdaCDM with Omega_m, A and M free, fitted to the same data.

Data (downloaded into ./data if missing)
  DESI DR2 BAO, 13 points with covariance: CobayaSampler/bao_data, desi_bao_dr2
  Pantheon+ (zHD > 0.01, calibrators removed), STAT+SYS covariance:
  PantheonPlusSH0ES/DataRelease

Optional extra data (each adds a chi2 term for the model and for LambdaCDM):
  --cmb     Planck 2018 distance priors (R, l_A, omega_b), cmb_likelihood.py;
            adds omega_b (model and LCDM) and h (LCDM only; in the model h follows
            from the history, because both Omega_m today and z_eq are fixed by it)
  --growth  Gold-2017 f sigma8 compilation, growth_likelihood.py; adds sigma8
  --bbn     approximate Y_p and D/H test, bbn_abundance.py; no new parameter
  --cmb and --bbn need the early-time table written by the updated gpu_ensemble.py.

Usage: python3 fit_survivors.py gpu_ensemble_results.jsonl [--max 2000] [--procs 8]
       [--cmb] [--growth] [--bbn]
"""
import argparse
import json
import os
import urllib.request
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import chi2 as chi2_dist

S32, S6 = np.sqrt(1.5), np.sqrt(6.0)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URLS = {
    "desi_mean.txt": "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_cov.txt": "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
    "pantheon.dat": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "pantheon.cov": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
}
WB_DEFAULT = 0.02236            # Chen, Huang & Wang (arXiv:1808.05724) central value
ZGRID = np.concatenate([[0.0], np.geomspace(1e-4, 2.6, 1500)])
D = {}


def load_data():
    os.makedirs(DATA, exist_ok=True)
    for name, url in URLS.items():
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(url, p)
    zb, vb, kind = [], [], []
    for line in open(os.path.join(DATA, "desi_mean.txt")):
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        zb.append(float(a[0])); vb.append(float(a[1])); kind.append(a[2])
    D["zb"], D["vb"], D["kind"] = np.array(zb), np.array(vb), kind
    D["icov_b"] = np.linalg.inv(np.loadtxt(os.path.join(DATA, "desi_cov.txt")))
    hdr = open(os.path.join(DATA, "pantheon.dat")).readline().split()
    tab = np.genfromtxt(os.path.join(DATA, "pantheon.dat"), skip_header=1, dtype=None, encoding=None)
    col = lambda n: np.array([row[hdr.index(n)] for row in tab], dtype=float)
    zhd, zhel, mb, cal = col("zHD"), col("zHEL"), col("m_b_corr"), col("IS_CALIBRATOR")
    raw = np.loadtxt(os.path.join(DATA, "pantheon.cov"))
    n = int(raw[0])
    cov = raw[1:].reshape(n, n)
    keep = (zhd > 0.01) & (cal == 0)
    D["zs"], D["zhel"], D["mb"] = zhd[keep], zhel[keep], mb[keep]
    ic = np.linalg.inv(cov[np.ix_(keep, keep)])
    D["icov_s"], D["ic1"] = ic, ic.sum(axis=1)
    D["s11"] = D["ic1"].sum()
    return int(keep.sum()), len(zb)


def chi2_from_E(E):
    """E = H/H0 on ZGRID. Returns (chi2_total, chi2_bao, chi2_sn, A)."""
    dc = np.concatenate([[0.0], cumulative_trapezoid(1.0 / E, ZGRID)])    # H0 D_M / c
    # supernovae, M marginalised
    dm = np.interp(D["zs"], ZGRID, dc)
    mu = 5 * np.log10((1 + D["zhel"]) * dm)
    r = D["mb"] - mu
    c_sn = r @ D["icov_s"] @ r - (D["ic1"] @ r) ** 2 / D["s11"]
    # BAO, linear in A = c / (H0 r_d)
    dmb = np.interp(D["zb"], ZGRID, dc)
    dhb = 1.0 / np.interp(D["zb"], ZGRID, E)
    t = np.array([dmb[i] if k == "DM_over_rs" else dhb[i] if k == "DH_over_rs"
                  else (D["zb"][i] * dmb[i] ** 2 * dhb[i]) ** (1 / 3) for i, k in enumerate(D["kind"])])
    A = (t @ D["icov_b"] @ D["vb"]) / (t @ D["icov_b"] @ t)
    rb = D["vb"] - A * t
    c_bao = rb @ D["icov_b"] @ rb
    return c_sn + c_bao, c_bao, c_sn, A


def lcdm_E(Om, Or=9.0e-5):
    return np.sqrt(Om * (1 + ZGRID) ** 3 + Or * (1 + ZGRID) ** 4 + 1 - Om - Or)


def w_bg(N, N_eq):
    return (1.0 / 3.0) / (1.0 + np.exp(np.clip(N - N_eq, -700, 700)))


def rhs(N, s, l1, l2, al, N_eq):
    x, y, th = s
    w = w_bg(N, N_eq)
    le = 0.5 * (l1 + l2) + 0.5 * (l1 - l2) * np.cos(th)
    brk = 2 * x * x + (1 + w) * (1 - x * x - y * y)
    return [-3 * x + S32 * le * y * y + 1.5 * x * brk,
            -S32 * le * x * y + 1.5 * y * brk, S6 * x / al]


def extra_terms(flags, sol, N_s, N_eq, N0, edges, S):
    """chi2 of the optional data sets for a history observed at N0."""
    use_cmb, use_growth, use_bbn = flags
    out = dict(chi2_cmb=0.0, chi2_growth=0.0, chi2_bbn=0.0)
    Nl = np.linspace(N_s, N0, 900)
    x, y, _ = sol.sol(Nl)
    Oph = x * x + y * y
    wb_fl = w_bg(Nl, N_eq)
    Om = (1 - Oph) * (1 - 3 * wb_fl)
    if use_growth:
        import growth_likelihood as G
        dlnH = -1.5 * (1 + x * x - y * y + wb_fl * (1 - Oph))
        shape = G.growth_shape(Nl, Om, dlnH)
        z = np.exp(N0 - Nl) - 1
        E = np.sqrt(Om[-1] / Om * (1 + z) ** 3)
        out["chi2_growth"], out["sigma8"] = G.chi2_from_curves(z[::-1], E[::-1], shape[::-1])
    dn = (0.0, 0.0)
    if use_bbn:
        import bbn_abundance as Bn
        Nc = 0.5 * (np.asarray(edges[1:]) + np.asarray(edges[:-1]))
        dn = Bn.delta_N(np.exp(N0 - Nc) - 1, 1 - np.asarray(S) ** 2)
        out["dN_he"], out["dN_d"] = dn
    if use_cmb:
        import cmb_likelihood as C
        def g(wb):
            c = C.model_chi2(N0, N_eq, Om[-1], Nl, Oph, np.asarray(edges), S, wb)[0]
            return c + (Bn.chi2(wb, *dn)[0] if use_bbn else 0.0)
        b = minimize_scalar(g, bounds=(0.0205, 0.0245), method="bounded", options=dict(xatol=1e-6))
        cc, info = C.model_chi2(N0, N_eq, Om[-1], Nl, Oph, np.asarray(edges), S, b.x)
        out.update(chi2_cmb=cc, wb=float(b.x), h=info["h"], R=info["R"], lA=info["lA"])
        if use_bbn:
            out["chi2_bbn"], out["Yp"], out["DH"], out["bbn_out_of_table"] = Bn.chi2(b.x, *dn)
    elif use_bbn:
        out["chi2_bbn"], out["Yp"], out["DH"], out["bbn_out_of_table"] = Bn.chi2(WB_DEFAULT, *dn)
    return out


def lcdm_reference(flags):
    """Best flat LambdaCDM fit to the same data. Returns (chi2 total, dict)."""
    use_cmb, use_growth, use_bbn = flags
    if use_growth:
        import growth_likelihood as G
    if use_bbn:
        import bbn_abundance as Bn
    if use_cmb:
        import cmb_likelihood as C
        def tot(p):
            Om, h, wb = p
            if not (0.1 < Om < 0.6 and 0.5 < h < 0.9 and 0.0205 < wb < 0.0245):
                return 1e9
            Or = Om / (1 + C.z_eq_of(Om * h * h))
            c = chi2_from_E(lcdm_E(Om, Or))[0] + C.lcdm_chi2(Om, h, wb)
            c += G.lcdm_chi2(Om, Or)[0] if use_growth else 0.0
            c += Bn.chi2(wb)[0] if use_bbn else 0.0
            return c
        b = minimize(tot, [0.315, 0.674, 0.0224], method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-5, maxiter=2000))
        Om, h, wb = b.x
        Or = Om / (1 + C.z_eq_of(Om * h * h))
        t, cb, cs, A = chi2_from_E(lcdm_E(Om, Or))
        d = dict(Om=float(Om), h=float(h), wb=float(wb), A=float(A), chi2_bao=float(cb), chi2_sn=float(cs),
                 chi2_cmb=float(C.lcdm_chi2(Om, h, wb)))
    else:
        def tot(Om):
            c = chi2_from_E(lcdm_E(Om))[0]
            return c + (G.lcdm_chi2(Om, 9.0e-5)[0] if use_growth else 0.0)
        b = minimize_scalar(tot, bounds=(0.1, 0.6), method="bounded")
        Om, Or, wb = b.x, 9.0e-5, WB_DEFAULT
        t, cb, cs, A = chi2_from_E(lcdm_E(Om))
        d = dict(Om=float(Om), A=float(A), chi2_bao=float(cb), chi2_sn=float(cs), chi2_cmb=0.0)
    d["chi2_growth"], d["sigma8"] = G.lcdm_chi2(Om, Or) if use_growth else (0.0, None)
    d["chi2_bbn"] = Bn.chi2(wb)[0] if use_bbn else 0.0
    d["chi2"] = float(cb + cs + d["chi2_cmb"] + d["chi2_growth"] + d["chi2_bbn"])
    return d["chi2"], d


def fit_one(args):
    l1, l2, al, N_s, N_eq, N_end, c, flags, edges = args[:9]
    cand_index = args[9] if len(args) > 9 else -1
    extras = any(flags)
    try:
        sol = solve_ivp(rhs, [N_s, N_end + 0.7], [c["x"], c["y"], c["theta"]], args=(l1, l2, al, N_eq),
                        method="DOP853", rtol=1e-10, atol=1e-13, dense_output=True)
        if sol.status != 0:
            return None

        def E_of(dN):
            N0 = N_end + dN
            N = N0 - np.log1p(ZGRID)
            x, y, _ = sol.sol(N)
            Om = (1 - x * x - y * y) * (1 - 3 * w_bg(N, N_eq))
            return np.sqrt(Om[0] / Om * (1 + ZGRID) ** 3), x, y, Om

        if extras and (flags[0] or flags[2]) and "S" not in c:
            return dict(error="candidate has no early-time table; rerun gpu_ensemble.py (updated version)")

        def f(dN):
            base = chi2_from_E(E_of(dN)[0])[0]
            if not extras:
                return base
            e = extra_terms(flags, sol, N_s, N_eq, N_end + dN, edges, c.get("S"))
            return base + e["chi2_cmb"] + e["chi2_growth"] + e["chi2_bbn"]
        grid = np.linspace(-0.6, 0.6, 25)
        vals = [f(g) for g in grid]
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        best = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-4))
        E, x, y, Om = E_of(best.x)
        tot, cb, cs, A = chi2_from_E(E)
        ex = extra_terms(flags, sol, N_s, N_eq, N_end + best.x, edges, c.get("S")) if extras else {}
        tot = tot + ex.get("chi2_cmb", 0.0) + ex.get("chi2_growth", 0.0) + ex.get("chi2_bbn", 0.0)
        # 68% interval on dN from delta chi2 = 1, converted to Omega_m
        def edge(sign):
            g = lambda d: f(d) - tot - 1.0
            far = best.x + sign * 0.3
            try:
                return brentq(g, best.x, far) if g(far) > 0 else far
            except ValueError:
                return far
        om_lo, om_hi = sorted(E_of(edge(s))[3][0] for s in (-1, 1))
        w = (x * x - y * y) / (x * x + y * y)
        k1 = int(np.argmin(np.abs(ZGRID - 1.0)))
        return dict(cand_index=cand_index, chi2=float(tot), chi2_bao=float(cb), chi2_sn=float(cs), dN=float(best.x), A=float(A),
                    Om0=float(Om[0]), Om0_68=[float(om_lo), float(om_hi)], Ophi0=float(x[0] ** 2 + y[0] ** 2), w0=float(w[0]),
                    w_z1=float(w[k1]), at_bound=bool(abs(best.x) > 0.59),
                    dH_max=float(np.max(np.abs(E / lcdm_E(Om[0]) - 1)[ZGRID < 2.4])),
                    **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in ex.items()})
    except Exception as e:
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble")
    ap.add_argument("--max", type=int, default=2000, help="candidates fitted per cell")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default="fit_results.json")
    ap.add_argument("--cmb", action="store_true", help="add Planck 2018 distance priors")
    ap.add_argument("--growth", action="store_true", help="add the Gold-2017 f sigma8 compilation")
    ap.add_argument("--bbn", action="store_true", help="add the approximate Y_p and D/H test")
    a = ap.parse_args()
    flags = (a.cmb, a.growth, a.bbn)
    n_sn, n_bao = load_data()
    tot, L0 = lcdm_reference(flags)
    cb, cs, A = L0["chi2_bao"], L0["chi2_sn"], L0["A"]
    ref = argparse.Namespace(x=L0["Om"])
    N_data = n_sn + n_bao + 3 * a.cmb + 18 * a.growth + 2 * a.bbn
    # LCDM: Omega_m, A, M (+ h, omega_b with --cmb; + sigma8 with --growth).
    # Model: dN, A, M, l1, alpha + choice of history (+ omega_b with --cmb; + sigma8 with --growth).
    K_LCDM = 3 + 2 * a.cmb + a.growth
    K_MODEL = 6 + a.cmb + a.growth
    dof_l = N_data - K_LCDM
    print(f"data: {n_sn} supernovae, {n_bao} BAO points")
    print(f"flat LCDM: Omega_m = {ref.x:.4f}, c/(H0 r_d) = {A:.2f}, chi2 = {tot:.2f} (BAO {cb:.2f}, SN {cs:.2f})")
    print(f"           chi2/dof = {tot/dof_l:.3f}, p = {chi2_dist.sf(tot, dof_l):.3f}; "
          f"BAO alone: chi2 = {cb:.2f} for {n_bao} points")
    if any(flags):
        print("           added data, LCDM: " + ", ".join(
            f"{k} {L0['chi2_' + k]:.2f}" for k, on in zip(("cmb", "growth", "bbn"), flags) if on)
            + (f"; h = {L0['h']:.4f}, omega_b = {L0['wb']:.5f}" if a.cmb else "")
            + (f"; sigma8 = {L0['sigma8']:.3f}" if a.growth else ""))
    out = dict(lcdm=dict(Om=float(ref.x), A=float(A), chi2=float(tot), chi2_bao=float(cb), chi2_sn=float(cs),
                         extra=L0, flags=dict(cmb=a.cmb, growth=a.growth, bbn=a.bbn),
                         dof=dof_l, p_value=float(chi2_dist.sf(tot, dof_l)),
                         AIC=float(tot + 2 * K_LCDM), BIC=float(tot + K_LCDM * np.log(N_data))),
               n_sn=n_sn, n_bao=n_bao, k_lcdm=K_LCDM, k_model=K_MODEL, cells=[])
    rng = np.random.default_rng(0)
    with Pool(a.procs, initializer=load_data) as pool:
        for c in map(json.loads, open(a.ensemble)):
            cands = c.get("candidates", [])[: a.max]
            jobs = [(c["l1"], c["l2"], c["alpha"], c["N_snapshot"], c["N_eq"], c["N_end"], k, flags,
                     c.get("S_edges"), i) for i, k in enumerate(cands)]
            raw = pool.map(fit_one, jobs, chunksize=4)
            errs = sorted({r["error"] for r in raw if r and "error" in r})
            if errs:
                print("    errors:", "; ".join(errs[:3]))
            res = [r for r in raw if r and "error" not in r]
            cell = dict(l1=c["l1"], l2=c["l2"], alpha=c["alpha"], n_histories=c["n"],
                        n_candidates=c.get("n_candidates", len(cands)), n_fitted=len(res), fits=res)
            out["cells"].append(cell)
            if not res:
                print(f"l1={c['l1']} alpha={c['alpha']}: no candidates")
                continue
            d = np.array([r["chi2"] for r in res]) - tot
            b = res[int(np.argmin(d))]
            print(f"l1={c['l1']} alpha={c['alpha']}: fitted {len(res)} of {cell['n_candidates']} candidates | "
                  f"delta chi2 vs LCDM: best {d.min():+.1f}, median {np.median(d):+.1f}; "
                  f"within +4: {(d < 4).sum()}, better than LCDM: {(d < 0).sum()} | best fit: "
                  f"Omega_m {b['Om0']:.3f}, w0 {b['w0']:.3f}, w(z=1) {b['w_z1']:.3f}, max dH/H {b['dH_max']:.3f}, "
                  f"BAO {b['chi2_bao']:.1f}, SN {b['chi2_sn']:.1f}", flush=True)
            if any(flags):
                print("    best fit, added data: " + ", ".join(
                    f"{k} {b.get('chi2_' + k, 0.0):.2f}" for k, on in zip(("cmb", "growth", "bbn"), flags) if on)
                    + (f"; h = {b['h']:.4f}, omega_b = {b['wb']:.5f}, R = {b['R']:.4f}, l_A = {b['lA']:.2f}" if a.cmb else "")
                    + (f"; sigma8 = {b['sigma8']:.3f}" if a.growth else "")
                    + (f"; Y_p = {b['Yp']:.4f}, D/H = {b['DH']*1e6:.2f}e-6" if a.bbn else ""), flush=True)
            # --- statistics for the cell ---
            L = np.exp(-0.5 * d)                                   # likelihood ratio to LCDM, per history
            scale = cell["n_candidates"] / max(len(cands), 1)      # if only a subset was fitted
            n_h = cell["n_histories"]
            # Likelihood ratio averaged over initial conditions on the attractor (non-candidates
            # contribute zero). dN and A are profiled, so this is an upper estimate of a Bayes factor.
            R = scale * L.sum() / n_h
            boot = [scale * rng.choice(L, len(L)).sum() / n_h for _ in range(500)] if len(L) > 1 else [R]
            wts = L / L.sum()
            wmean = lambda key: float(np.sum(wts * np.array([r[key] for r in res])))
            wsd = lambda key: float(np.sqrt(np.sum(wts * (np.array([r[key] for r in res]) - wmean(key)) ** 2)))
            n_eff = float(1.0 / np.sum(wts ** 2))
            dof_m = N_data - K_MODEL
            # chance of a delta chi2 this low or lower among n_fitted trials, if each trial's
            # delta chi2 were an independent chi-square(1) fluctuation about LCDM
            p_single = chi2_dist.sf(max(-d.min(), 0.0), 1)
            cell["stats"] = dict(
                best_chi2=float(b["chi2"]), best_delta_chi2=float(d.min()),
                best_chi2_per_dof=float(b["chi2"] / dof_m), best_p_value=float(chi2_dist.sf(b["chi2"], dof_m)),
                delta_AIC=float(d.min() + 2 * (K_MODEL - K_LCDM)),
                delta_BIC=float(d.min() + (K_MODEL - K_LCDM) * np.log(N_data)),
                trials=len(res), p_best_after_trials=float(1 - (1 - p_single) ** len(res)),
                averaged_likelihood_ratio=float(R), averaged_likelihood_ratio_se=float(np.std(boot)),
                effective_histories=n_eff,
                weighted=dict(Om0=[wmean("Om0"), wsd("Om0")], w0=[wmean("w0"), wsd("w0")],
                              w_z1=[wmean("w_z1"), wsd("w_z1")], dH_max=[wmean("dH_max"), wsd("dH_max")]),
                best_Om0_68=b["Om0_68"], n_at_bound=int(sum(r["at_bound"] for r in res)))
            st = cell["stats"]
            print(f"    best: chi2/dof {st['best_chi2_per_dof']:.3f}, p {st['best_p_value']:.3f}, "
                  f"delta AIC {st['delta_AIC']:+.1f}, delta BIC {st['delta_BIC']:+.1f} "
                  f"(k = {K_MODEL} vs {K_LCDM}); Omega_m 68% [{b['Om0_68'][0]:.3f}, {b['Om0_68'][1]:.3f}]")
            print(f"    averaged likelihood ratio to LCDM over initial conditions: {st['averaged_likelihood_ratio']:.2e} "
                  f"+- {st['averaged_likelihood_ratio_se']:.1e} (effective histories {n_eff:.1f})")
            print(f"    likelihood-weighted: Omega_m {st['weighted']['Om0'][0]:.3f} +- {st['weighted']['Om0'][1]:.3f}, "
                  f"w0 {st['weighted']['w0'][0]:.3f} +- {st['weighted']['w0'][1]:.3f}, "
                  f"max dH/H {st['weighted']['dH_max'][0]:.4f} +- {st['weighted']['dH_max'][1]:.4f}; "
                  f"fits at the dN bound: {st['n_at_bound']}", flush=True)
            json.dump(out, open(a.out, "w"))
    print("The model has one more shape freedom than LCDM here (the choice of history), so a")
    print("negative best delta chi2 over many candidates is expected by chance; compare with")
    print("the number of candidates before reading it as a preference.")


if __name__ == "__main__":
    main()
