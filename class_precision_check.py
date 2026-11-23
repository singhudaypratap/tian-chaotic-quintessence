"""Checks on the field-level CMB numbers from class_field.py.

1. Precision: re-evaluate chi2 at each stored best fit (model and LCDM) with CLASS's default and
   "high" precision settings. A converged result changes by much less than 1 in delta chi2.
2. Where the penalty sits: TT, TE, EE residuals of each model best fit against the LCDM best fit,
   in units of the plik_lite bin errors, saved to class_residuals.npz and class_residuals.png.

Usage:
  python3 class_precision_check.py FIT_all.json ENSEMBLE.jsonl CLASS_RESULTS.json [--top 3] [--procs 4]
CLASS_RESULTS.json must come from the current class_field.py (it stores the fitted point x, with tau).
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import class_field as CF
import cmb_spectrum as CS


def lcdm_spectra(ref, As):
    c = CF.G["Class"]()
    wb, wc, th, ns, tau = ref["x"]
    c.set({"omega_b": wb, "omega_cdm": wc, "100*theta_s": th, "n_s": ns, "A_s": As, "tau_reio": tau,
           "N_ur": 3.044, "N_ncdm": 0, "T_cmb": CF.T_CMB, "output": "tCl,pCl,lCl", "lensing": "yes",
           "l_max_scalars": CF.LMAX + 200, **CF.PRECISION[CF.G["precision"]]})
    c.compute()
    cl = c.lensed_cl(CF.LMAX)
    c.struct_cleanup()
    l = cl["ell"]
    f = l * (l + 1) / (2 * np.pi) * (CF.T_CMB * 1e6) ** 2
    pad = lambda arr: np.concatenate([arr * f, np.zeros(max(0, CF.plik_lite.LMAX_NEEDED + 1 - len(l)))])
    return pad(cl["tt"]), pad(cl["te"]), pad(cl["ee"])


def model_spectra(H, r, As):
    wb, ns, theta, tau = r["x"]
    dN = CF.solve_dN(H, wb, theta)
    p, _ = CF.class_input(H, dN, wb)
    p["n_s"], p["tau_reio"] = ns, tau
    return CF.spectra(p, As)[0]


def key(t):
    return (t["l1"], t["l2"], t["alpha"], t["cand_index"])


def job(args):
    meta, cand, r, ref, prec = args
    CF.init(True, True, prec, 1)
    out = {}
    H = CF.load(meta, cand)
    H["dN0"] = r["dN"]                       # starting guess for the theta solve
    for name, spec_fn in (("model", lambda As: model_spectra(H, r, As)), ("lcdm", lambda As: lcdm_spectra(ref, As))):
        spec = spec_fn(2.1e-9)
        c, s = CF.G["L"].chi2_best_scale(*spec)
        spec = spec_fn(2.1e-9 * s)
        c, s2 = CF.G["L"].chi2_best_scale(*spec)
        out[name] = dict(chi2_cmb=float(c), spec=[np.asarray(a).tolist() for a in spec])
    return dict(cand=key(r["tag"]), precision=prec, **out)


def binned_residuals(spec_m, spec_l):
    """Model minus LCDM in plik_lite bins (TT, TE, EE in sequence), divided by the bin errors."""
    L = CF.G["L"]
    bm = L.bin(*[np.asarray(v) for v in spec_m])
    bl = L.bin(*[np.asarray(v) for v in spec_l])
    err = np.sqrt(np.diag(L.cov))
    ell = np.arange(L.B_tt.shape[1])
    ctt = L.B_tt @ ell / L.B_tt.sum(1)
    cpol = L.B_pol @ ell / L.B_pol.sum(1)
    return (bm - bl) / err, np.concatenate([ctt, cpol, cpol])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("ensemble"); ap.add_argument("class_results")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="class_precision_check.json")
    a = ap.parse_args()
    res = json.load(open(a.class_results))
    ref = res["lcdm"]
    if "x" not in ref:
        raise SystemExit("CLASS_RESULTS.json is from the old class_field.py (no fitted point stored): rerun it")
    done = sorted([r for r in res["results"] if "error" not in r and "x" in r], key=lambda r: r["chi2"])[: a.top]
    picked = CS.select(a.fit, a.ensemble, 10 ** 6, None)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(a.ensemble))}
    by_idx = {key(t): (m, c, t) for m, c, _, t in picked}
    jobs = []
    for r in done:
        m, c, t = by_idx[key(r["tag"])]
        meta = dict(m, N_state_early=cells[(t["l1"], t["l2"], t["alpha"])]["N_state_early"])
        for prec in ("default", "high"):
            jobs.append((meta, c, r, ref, prec))
    with mp.get_context("spawn").Pool(a.procs) as pool:
        out = pool.map(job, jobs, chunksize=1)

    CF.init(True, True, "default", 1)
    rows, plots = [], {}
    for r in done:
        k = key(r["tag"])
        d = {o["precision"]: o for o in out if o["cand"] == k}
        dd = {p: d[p]["model"]["chi2_cmb"] - d[p]["lcdm"]["chi2_cmb"] for p in d}
        rows.append(dict(cand=k, delta_cmb_default=dd["default"], delta_cmb_high=dd["high"],
                         shift=dd["high"] - dd["default"], Ophi_at_start=r.get("Ophi_at_start")))
        print(f"cand {k}: CMB delta chi2 at the best fit, default precision {dd['default']:+.2f}, "
              f"high {dd['high']:+.2f}, change {dd['high'] - dd['default']:+.2f}; "
              f"Omega_phi at z_start {r.get('Ophi_at_start', float('nan')):.1e}", flush=True)
        try:
            res_b, lc = binned_residuals(d["high"]["model"]["spec"], d["high"]["lcdm"]["spec"])
            plots[k] = (lc, res_b)
        except Exception as e:
            print(f"cand {k}: residual plot skipped ({e!r})")
    json.dump(dict(rows=rows), open(a.out, "w"), indent=1)
    if plots:
        np.savez("class_residuals.npz", **{"cand_%g_%g_%d_ell" % (k[0], k[2], k[3]): v[0] for k, v in plots.items()},
                 **{"cand_%g_%g_%d_res" % (k[0], k[2], k[3]): v[1] for k, v in plots.items()})
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            L = CF.G["L"]
            ntt, npol = L.B_tt.shape[0], L.B_pol.shape[0]
            parts = [("TT", slice(0, ntt)), ("TE", slice(ntt, ntt + npol)), ("EE", slice(ntt + npol, ntt + 2 * npol))]
            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            for ax, (name, sl) in zip(axs, parts):
                for k, (lc, rb) in plots.items():
                    ax.plot(lc[sl], rb[sl], "-", lw=1, label=f"l1={k[0]:g}, alpha={k[2]:g}, cand {k[3]}")
                ax.axhline(0, color="0.5", lw=0.8)
                ax.set_ylabel(f"{name}: (model - LCDM) / error")
            axs[0].legend(frameon=False, fontsize=8)
            axs[-1].set_xlabel("multipole")
            fig.tight_layout(); fig.savefig("class_residuals.png", dpi=150)
        except ImportError:
            pass
    big = [x for x in rows if abs(x["shift"]) > 1.0]
    print("precision check:", "converged (all changes below 1)" if not big else
          f"NOT converged for {len(big)} candidate(s): use --precision high in class_field.py")


if __name__ == "__main__":
    main()
