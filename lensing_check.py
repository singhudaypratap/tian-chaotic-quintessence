"""CMB lensing check of the full-spectrum best fits (ACT DR6 + Planck PR4 lensing).

The primary-CMB fits (class_field.py) use Planck 2018 TT,TE,EE only. This script evaluates the
ACT DR6 lensing likelihood (variant actplanck_baseline: ACT DR6 + Planck PR4 lensing) at each stored
best fit and at the stored LCDM best fit, and reports the difference in -2 ln L. Nothing is refitted,
so the numbers say whether the lensing data would add a penalty to the primary-CMB best fits.

The amplitude A_s is fixed as in class_precision_check.py: the profiled plik_lite scale at the best fit.

Install once:  pip install act_dr6_lenslike  and fetch its data (get-act-data.sh, or the LAMBDA tarball
               unpacked so that act_dr6_lenslike/data/v1.2 exists).

Usage:
  python3 lensing_check.py FIT_all.json ENSEMBLE.jsonl CLASS_RESULTS.json [--top 10] [--procs 6] [--out lensing_check.json]
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import class_field as CF
import cmb_spectrum as CS

LMAX_OUT = 4000
VARIANT = "actplanck_baseline"


def cls(q, As):
    """CLASS lensed CMB C_ell (muK^2) and convergence C_L^kk for parameter dict q at amplitude As."""
    c = CF.G["Class"]()
    q = dict(q)
    q.update({"output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": LMAX_OUT + 500, "A_s": As})
    q.update(CF.PRECISION[CF.G["precision"]])
    c.set(q)
    c.compute()
    cl = c.lensed_cl(LMAX_OUT)
    c.struct_cleanup()
    ell = cl["ell"].astype(float)
    t2 = (CF.T_CMB * 1e6) ** 2
    kk = (ell * (ell + 1.0)) ** 2 / 4.0 * cl["pp"]
    return ell, kk, cl["tt"] * t2, cl["ee"] * t2, cl["te"] * t2, cl["bb"] * t2


def profiled_As(spec_fn):
    spec = spec_fn(2.1e-9)
    _, s = CF.G["L"].chi2_best_scale(*spec)
    return 2.1e-9 * s


def job(args):
    meta, cand, r, ref = args
    import act_dr6_lenslike as alike
    CF.init(True, True, "default", 1)
    data = alike.load_data(VARIANT, lens_only=False, like_corrections=True)
    out = {}
    # model
    H = CF.load(meta, cand)
    H["dN0"] = r["dN"]
    wb, ns, theta, tau = r["x"]
    dN = CF.solve_dN(H, wb, theta)
    p, _ = CF.class_input(H, dN, wb)
    p["n_s"], p["tau_reio"] = ns, tau
    As = profiled_As(lambda A: CF.spectra(dict(p), A)[0])
    q = dict(p)
    q["n_s"], q["tau_reio"] = q.pop("n_s"), q.pop("tau_reio")
    ell, kk, tt, ee, te, bb = cls(q, As)
    out["model"] = float(-2 * alike.generic_lnlike(data, ell, kk, ell, tt, ee, te, bb))
    # LCDM
    wb0, wc0, th0, ns0, tau0 = ref["x"]
    q0 = {"omega_b": wb0, "omega_cdm": wc0, "100*theta_s": th0, "n_s": ns0, "tau_reio": tau0,
          "N_ur": 3.044, "N_ncdm": 0, "T_cmb": CF.T_CMB}

    def lcdm_spec(A):
        c = CF.G["Class"]()
        c.set({**q0, "A_s": A, "output": "tCl,pCl,lCl", "lensing": "yes", "l_max_scalars": CF.LMAX + 200,
               **CF.PRECISION[CF.G["precision"]]})
        c.compute()
        cl = c.lensed_cl(CF.LMAX)
        c.struct_cleanup()
        l = cl["ell"]
        f = l * (l + 1) / (2 * np.pi) * (CF.T_CMB * 1e6) ** 2
        pad = lambda arr: np.concatenate([arr * f, np.zeros(max(0, CF.plik_lite.LMAX_NEEDED + 1 - len(l)))])
        return pad(cl["tt"]), pad(cl["te"]), pad(cl["ee"])

    As0 = profiled_As(lcdm_spec)
    ell, kk, tt, ee, te, bb = cls(q0, As0)
    out["lcdm"] = float(-2 * alike.generic_lnlike(data, ell, kk, ell, tt, ee, te, bb))
    t = r["tag"]
    return dict(cand=[t["l1"], t["l2"], t["alpha"], t["cand_index"]], delta_chi2_primary=r.get("delta_chi2"),
                chi2_lens_model=out["model"], chi2_lens_lcdm=out["lcdm"],
                delta_chi2_lens=out["model"] - out["lcdm"], As_model=As, As_lcdm=As0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("ensemble"); ap.add_argument("class_results")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="lensing_check.json")
    a = ap.parse_args()
    res = json.load(open(a.class_results))
    ref = res["lcdm"]
    done = sorted([r for r in res["results"] if "error" not in r and "x" in r], key=lambda r: r["chi2"])[: a.top]
    picked = CS.select(a.fit, a.ensemble, 10 ** 6, None)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(a.ensemble))}
    key = lambda t: (t["l1"], t["l2"], t["alpha"], t["cand_index"])
    by_idx = {key(t): (m, c, t) for m, c, _, t in picked}
    jobs = []
    for r in done:
        m, c, t = by_idx[key(r["tag"])]
        meta = dict(m, N_state_early=cells[(t["l1"], t["l2"], t["alpha"])]["N_state_early"])
        jobs.append((meta, c, r, ref))
    with mp.get_context("spawn").Pool(a.procs) as pool:
        out = pool.map(job, jobs, chunksize=1)
    for o in out:
        print(f"cand {o['cand']}: primary delta chi2 {o['delta_chi2_primary']:+.2f}; lensing delta chi2 "
              f"{o['delta_chi2_lens']:+.2f} (model {o['chi2_lens_model']:.2f}, LCDM {o['chi2_lens_lcdm']:.2f}); "
              f"total {o['delta_chi2_primary'] + o['delta_chi2_lens']:+.2f}", flush=True)
    json.dump(dict(variant=VARIANT, rows=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
