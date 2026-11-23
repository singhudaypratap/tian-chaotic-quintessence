"""BBN abundances with the scalar field included, using PRyMordial's nuclear network.

UNTESTED. This wrapper was written by reading PRyMordial's source (PRyM_init.py,
PRyM_main.py, PRyM_thermo.py, runPRyM_julia.py from https://github.com/vallima/PRyMordial,
arXiv:2307.07061) without running it: executing downloaded repository code was not
permitted where this file was written. Run the self-check first (see "Checks").

Why this is needed: bbn_abundance.py maps the field to a constant number of extra
neutrino species and reads Y_p and D/H from a table. That fails when the field
fraction changes strongly during BBN, which is exactly the case Tian's Part IV
relies on (field hidden during the main BBN era, rising at the end). Here the field's
actual density history enters the expansion rate of a full network calculation.

How the field is passed to PRyMordial
  PRyMclass(my_rho_NP, my_p_NP, my_drho_NP_dT, my_delta_rho_NP) accepts a new species.
  With PRyMini.NP_thermo_flag = True the species has its own "temperature" T_NP that
  PRyMordial evolves with
        dT_NP/dt = [-3 H (rho_NP + p_NP) + delta_rho_NP] / (d rho_NP / dT_NP)
  and its density is added to the Friedmann equation (PRyM_main.py, functions Hubble,
  dTNPdt). The other two switches (NP_e_flag, NP_nu_flag) put the species in thermal
  equilibrium with the plasma or the neutrinos and add its entropy, which is wrong for
  a decoupled field, so they are not used.
  T_NP is used here only as a clock. With delta_rho_NP = 0 and
        d rho_NP / dT_NP = 3 (rho_NP + p_NP) / T_NP
  the equation above gives dT_NP/dt = -H T_NP, i.e. T_NP = T_start * a_start / a
  exactly. So rho_NP(T_NP) is simply the field density at scale factor
  a = a_start * T_start / T_NP, taken from the stored history. No thermal assumption
  about the field is made. [derived]

Field density from the stored tables (written by gpu_ensemble.py in this folder)
  W: bin averages of w_phi in 86 bins of N between z = 1e11 and z = 50;
  S: bin averages of sqrt(1 - Omega_phi) in the same bins.
  ln rho_phi(N) follows from d ln rho/dN = -3 (1 + W), anchored at T = T_start (10 MeV
  by default, where photons, e+- and neutrinos share one temperature) by
        rho_phi = Omega/(1 - Omega) * rho_SM(T_start),   Omega = 1 - S^2 in that bin,
  with rho_SM from PRyMordial's own thermodynamics. The redshift of T_start comes from
  PRyMordial's entropy relation a(T) = (s0CMB / s_plasma(T))^(1/3).
  Limits: W and S are averages over 0.25 e-folds, so structure in the field's density
  faster than that is smoothed; the anchor uses a bin average of Omega.

Output: Y_p (BBN definition, 4 n_He / n_b) and D/H, and chi2 against
  Y_p = 0.245 +- 0.003 and D/H = (25.08 +- 0.29)e-6 (PDG 2025 review, Eqs. 24.3, 24.2;
  same values as bbn_abundance.py), with the theory errors of the PRyMordial table
  (data/prym_yp_dh_2023.dat) at Delta N = 0 added in quadrature.

Install on the server (PRyMordial README):
  git clone https://github.com/vallima/PRyMordial.git
  pip install numpy scipy numba numdifftools
Usage (PRyMordial reads its rate files relative to the current directory, so the
script changes into --prym before importing it):
  python3 bbn_network.py FIT.json ENSEMBLE.jsonl --prym /path/to/PRyMordial [--top 50]
          [--procs 20] [--large-network] [--out bbn_network_results.json]
  python3 bbn_network.py --self-check --prym /path/to/PRyMordial

Checks to make before trusting any number (--self-check runs 1-3):
  1. No field: Y_p and D/H must agree with the table value at Delta N = 0 for the same
     omega_b (bbn_abundance.chi2) to within the table's theory errors.
  2. A field with constant fraction Omega and w = 1/3 must reproduce the table at
     Delta N = (10.75/1.75) Omega/(1 - Omega) for Y_p to a few 1e-4 (the mapping is
     exact before e+e- annihilation and approximate after it).
  3. The printed T_NP check: PRyMordial's final T_NP must equal
     T_start * a_start / a_end to better than 1%.
Expected cost: PRyMordial's README example runs the small and large networks in turn;
its quoted run times are not reproduced here. Expect seconds to a few tens of seconds
per history with the small network once Numba has compiled (my estimate, unverified).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YP_OBS, YP_ERR = 0.245, 0.003                 # PDG 2025 BBN review, Eq. 24.3
DH_OBS, DH_ERR = 25.08e-6, 0.29e-6            # PDG 2025 BBN review, Eq. 24.2
G = {}


def load_prym(path, large):
    """Import PRyMordial from `path` (must be the repository root) and set its flags."""
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "PRyM")):
        raise SystemExit(f"{path} does not contain the PRyM package. Clone it with:\n"
                         "  git clone https://github.com/vallima/PRyMordial.git")
    os.chdir(path)                              # PRyM_init.py uses os.getcwd() for its rate files
    sys.path.insert(0, path)
    import PRyM.PRyM_init as ini
    # same settings as the repository's example script runPRyM_julia.py
    ini.aTid_flag = True
    ini.compute_bckg_flag = True
    ini.compute_nTOp_flag = True
    ini.compute_nTOp_thermal_flag = False
    ini.save_bckg_flag = False
    ini.save_nTOp_flag = False
    ini.save_nTOp_thermal_flag = False
    ini.verbose_flag = False
    ini.julia_flag = False
    ini.smallnet_flag = not large
    ini.NP_e_flag = False
    ini.NP_nu_flag = False
    import PRyM.PRyM_thermo as thermo
    import PRyM.PRyM_main as main
    G.update(ini=ini, thermo=thermo, main=main)


def set_omega_b(wb):
    ini = G["ini"]
    ini.Omegabh2 = wb
    ini.eta0b = ini.Omegabh2_to_eta0b * wb       # as in PRyM_init.py


def run_prym(field=None):
    """Run the network. field = None, or a dict with callables rho, p, drho of T_NP [MeV]."""
    ini, main = G["ini"], G["main"]
    if field is None:
        ini.NP_thermo_flag = False
        res = main.PRyMclass().PRyMresults()
    else:
        ini.NP_thermo_flag = True
        ini.xi_NP = 1.0
        ini.Tstart_NP = ini.T_start / ini.MeV_to_Kelvin      # MeV; the clock starts at T_start
        zero = lambda Tg, Tnue, Tnumu, T_NP: 0.0
        res = main.PRyMclass(field["rho"], field["p"], field["drho"], zero).PRyMresults()
    # PRyM_main.py: res = [Neff, Omega_nu_rel x1e6, 1/Omega_nu_nr, YP(CMB), YP(BBN), D/H x1e5, He3/H x1e5, Li7/H x1e10]
    return dict(Neff=float(res[0]), Yp=float(res[4]), DH=float(res[5]) * 1e-5)


def a_of_T(T_MeV):
    """PRyMordial's instantaneous-decoupling relation, a = 1 today (PRyM_main.py, a_of_T)."""
    ini, thermo = G["ini"], G["thermo"]
    return (ini.s0CMB / thermo.spl(T_MeV)) ** (1.0 / 3.0)


def field_functions(edges, S, W, N0):
    """rho_NP, p_NP, d rho_NP/dT_NP as functions of the clock T_NP [MeV].

    edges: bin edges in N (increasing); S, W: the stored tables; N0: N today."""
    ini, thermo = G["ini"], G["thermo"]
    edges, S, W = np.asarray(edges, float), np.asarray(S, float), np.asarray(W, float)
    T_start = ini.T_start / ini.MeV_to_Kelvin
    a_start = float(a_of_T(T_start))
    N_a = N0 + np.log(a_start)                       # N at T = T_start
    if not (edges[0] < N_a < edges[-1]):
        raise RuntimeError("T_start lies outside the stored table")
    # ln rho at the bin edges, relative to an arbitrary constant
    L = np.concatenate([[0.0], np.cumsum(-3.0 * (1.0 + W) * np.diff(edges))])
    k = int(np.searchsorted(edges, N_a) - 1)
    Om = float(np.clip(1.0 - S[k] ** 2, 0.0, 0.999999))
    rho_a = Om / (1.0 - Om) * float(thermo.rho_SM(T_start, T_start, T_start))    # MeV^4
    L_a = np.interp(N_a, edges, L)

    def N_of(T_NP):
        return N_a + np.log(T_start / np.asarray(T_NP, float))    # a = a_start T_start / T_NP

    def w_of(T_NP):
        idx = np.clip(np.searchsorted(edges, N_of(T_NP)) - 1, 0, len(W) - 1)
        return W[idx]

    def rho(T_NP):
        return rho_a * np.exp(np.interp(N_of(T_NP), edges, L) - L_a)

    def p(T_NP):
        return w_of(T_NP) * rho(T_NP)

    def drho(T_NP):
        return 3.0 * (1.0 + w_of(T_NP)) * rho(T_NP) / np.asarray(T_NP, float)

    return dict(rho=rho, p=p, drho=drho, Omega_at_Tstart=Om, a_start=a_start)


def theory_errors(wb):
    """Theory errors on Y_p and D/H from the PRyMordial table at Delta N = 0."""
    sys.path.insert(0, HERE)
    try:
        import bbn_abundance as B
        w = float(np.clip(wb, B._wb[0], B._wb[-1]))
        return float(B._SY([0.0, w])[0]), float(B._SD([0.0, w])[0])
    except Exception:
        return 0.0, 0.0


def chi2(r, wb):
    sy, sd = theory_errors(wb)
    return float((r["Yp"] - YP_OBS) ** 2 / (YP_ERR ** 2 + sy ** 2)
                 + (r["DH"] - DH_OBS) ** 2 / (DH_ERR ** 2 + sd ** 2))


def worker_init(prym, large):
    load_prym(prym, large)


def do_candidate(args):
    cell, cand, dN, wb, tag = args
    try:
        set_omega_b(wb)
        f = field_functions(cell["S_edges"], cand["S"], cand["W"], cell["N_end"] + dN)
        r = run_prym(f)
        out = dict(tag=tag, wb=wb, **r, chi2_bbn=chi2(r, wb), Omega_at_Tstart=f["Omega_at_Tstart"])
        # the constant-Delta-N estimate for comparison
        try:
            import bbn_abundance as B
            Nc = 0.5 * (np.asarray(cell["S_edges"][1:]) + np.asarray(cell["S_edges"][:-1]))
            dn = B.delta_N(np.exp(cell["N_end"] + dN - Nc) - 1, 1 - np.asarray(cand["S"]) ** 2)
            c, yp, dh, _ = B.chi2(wb, *dn)
            out.update(table_Yp=yp, table_DH=dh, table_chi2=c)
        except Exception:
            pass
        return out
    except Exception as e:
        return dict(tag=tag, error=repr(e)[:300])


def self_check(prym, large):
    load_prym(prym, large)
    ini = G["ini"]
    wb = 0.02236
    set_omega_b(wb)
    r0 = run_prym(None)
    print(f"1. no field, omega_b = {wb}: Y_p = {r0['Yp']:.5f}, D/H = {r0['DH']*1e6:.3f}e-6, "
          f"N_eff = {r0['Neff']:.4f}, chi2 = {chi2(r0, wb):.2f}")
    try:
        sys.path.insert(0, HERE)
        import bbn_abundance as B
        print(f"   table at Delta N = 0: Y_p = {B.chi2(wb)[1]:.5f}, D/H = {B.chi2(wb)[2]*1e6:.3f}e-6")
    except Exception as e:
        print("   (table comparison unavailable:", e, ")")
    # 2. constant fraction with w = 1/3: an exact extra-radiation case
    T_start = ini.T_start / ini.MeV_to_Kelvin
    a_start = float(a_of_T(T_start))
    for Om in (0.01, 0.04):
        N0 = 0.0
        edges = np.linspace(np.log(a_start) - 2.0, np.log(a_start) + 16.0, 73)
        S = np.full(72, np.sqrt(1 - Om))
        W = np.full(72, 1.0 / 3.0)
        f = field_functions(edges, S, W, N0)
        r = run_prym(f)
        dN_eq = (10.75 / 1.75) * Om / (1 - Om)
        line = (f"2. constant Omega = {Om}, w = 1/3: Y_p = {r['Yp']:.5f}, D/H = {r['DH']*1e6:.3f}e-6, "
                f"N_eff = {r['Neff']:.3f} (N_eff includes the field)")
        try:
            line += f"; table at Delta N = {dN_eq:.3f}: Y_p = {B.chi2(wb, dN_eq, dN_eq)[1]:.5f}"
        except Exception:
            pass
        print(line)
    print("   note: with w = 1/3 the field keeps a constant fraction of the radiation only until e+e-\n"
          "   annihilation heats the photons, so agreement with the table is expected for Y_p (set\n"
          "   before annihilation) and only approximately for D/H.")
    print("3. T_NP clock: compare PRyMordial's saved T_NP at the end with T_start*a_start/a_end\n"
          "   (set PRyMini.save_bckg_flag = True and read PRyMrates/thermo/Tgamma_Tnu_TNP.txt).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit", nargs="?")
    ap.add_argument("ensemble", nargs="?")
    ap.add_argument("--prym", required=True, help="path to the cloned PRyMordial repository")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--large-network", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--out", default="bbn_network_results.json")
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    if a.self_check:
        self_check(a.prym, a.large_network)
        return
    fit_path, ens_path = os.path.abspath(a.fit), os.path.abspath(a.ensemble)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(ens_path))}
    rows = []
    for fc in json.load(open(fit_path))["cells"]:
        key = (fc["l1"], fc["l2"], fc["alpha"])
        for r in fc.get("fits", []):
            if key in cells and r.get("cand_index", -1) >= 0:
                rows.append((r["chi2"], key, r["cand_index"], r["dN"], r.get("wb", 0.02236)))
    rows.sort(key=lambda t: t[0])
    jobs = []
    for c2, key, i, dN, wb in rows[: a.top]:
        cell = cells[key]
        cand = cell["candidates"][i]
        if "W" not in cand:
            raise SystemExit("ENSEMBLE.jsonl has no W table: rerun gpu_ensemble.py from this folder")
        meta = {k: cell[k] for k in ("N_end", "S_edges")}
        jobs.append((meta, dict(S=cand["S"], W=cand["W"]), dN, float(wb),
                     dict(l1=key[0], l2=key[1], alpha=key[2], cand_index=i, fast_fit_chi2=c2)))
    print(f"{len(jobs)} candidate(s); network: {'large' if a.large_network else 'small'}", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.procs, initializer=worker_init,
                                      initargs=(a.prym, a.large_network)) as pool:
        res = pool.map(do_candidate, jobs, chunksize=1)
    for r in res:
        t = r["tag"]
        head = f"l1={t['l1']} alpha={t['alpha']} cand {t['cand_index']}: "
        if "error" in r:
            print(head + "ERROR " + r["error"])
            continue
        line = head + (f"Y_p = {r['Yp']:.5f}, D/H = {r['DH']*1e6:.3f}e-6, chi2 = {r['chi2_bbn']:.2f}; "
                       f"field fraction at 10 MeV {r['Omega_at_Tstart']:.2e}")
        if "table_chi2" in r:
            line += f"; constant-Delta-N table: Y_p = {r['table_Yp']:.5f}, D/H = {r['table_DH']*1e6:.3f}e-6, chi2 = {r['table_chi2']:.2f}"
        print(line)
    json.dump(dict(obs=dict(Yp=[YP_OBS, YP_ERR], DH=[DH_OBS, DH_ERR]), results=res), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
