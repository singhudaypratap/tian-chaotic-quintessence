"""Why do CLASS and CAMB disagree for a few late-thawing candidates?

For each named candidate, at CAMB's own best fit (omega_b, n_s, dN; tau = 0.054 as in CAMB), compute
the CMB chi2 and lensed spectra with:
  camb      CAMB, dark energy as a w(a) fluid (cs2 = 1), as in cmb_spectrum.py
  class     CLASS scalar field started at z = 1e9 (class_field.py default)
  class_hi   CLASS, start z = 1e9, high precision preset
  class_Y    CLASS, start z = 1e9, helium fixed at 0.2454
  class_z7_Y CLASS started later, at z = 3e7 (CLASS thermodynamics starts near z = 5e6), from the same history, helium fixed at 0.2454
             (compare with class_Y: the difference is the effect of the start time alone)
each against LCDM computed by the same code at CAMB's LCDM best fit, so code-to-code offsets cancel.
Also prints h and omega_r seen by each code and the largest TT/EE spectrum differences by l range.

Usage:
  python3 camb_class_diag.py FIT_all.json ENSEMBLE.jsonl CAMB_best.json --cands "10:0.8:1937,12:0.8:271"
  (l1:alpha:cand_index; add agreeing candidates as controls, e.g. 10:0.6:414)
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import class_field as CF
import cmb_spectrum as CS

TAU_CAMB = CS.TAU


def key(t):
    return (float(t["l1"]), float(t["alpha"]), int(t["cand_index"]))


def chi2(L, spec, As_fn):
    """Best-fit amplitude chi2 (two passes, as the fitting scripts do)."""
    s0 = spec(2.1e-9)
    c, s = L.chi2_best_scale(*s0)
    s1 = spec(2.1e-9 * s)
    c, _ = L.chi2_best_scale(*s1)
    return float(c), s1


def class_lcdm(ref):
    def spec(As):
        c = CF.G["Class"]()
        c.set({"omega_b": ref["wb"], "omega_cdm": ref["wc"], "h": ref["H0"] / 100, "n_s": ref["ns"], "A_s": As,
               "tau_reio": TAU_CAMB, "N_ur": 3.044, "N_ncdm": 0, "T_cmb": CF.T_CMB, "output": "tCl,pCl,lCl",
               "lensing": "yes", "l_max_scalars": CF.LMAX + 200, **CF.PRECISION[CF.G["precision"]]})
        c.compute()
        cl = c.lensed_cl(CF.LMAX)
        c.struct_cleanup()
        l = cl["ell"]
        f = l * (l + 1) / (2 * np.pi) * (CF.T_CMB * 1e6) ** 2
        return cl["tt"] * f, cl["te"] * f, cl["ee"] * f
    return spec


YHE_FIXED = 0.2454          # for the late-start comparison (CLASS's BBN helium needs the background at z_bbn)


def class_model(H, dN, wb, ns, z_start=None, fix_yhe=False):
    Hc = dict(H)
    if z_start is not None:                       # restart CLASS later on the same history
        N0 = H["N_end"] + dN
        Ns = N0 - np.log1p(z_start)
        Hc["state"] = tuple(float(v) for v in H["sol"].sol(Ns))
        Hc["N_s"] = Ns
    p, info = CF.class_input(Hc, dN, wb)
    p["n_s"], p["tau_reio"] = ns, TAU_CAMB
    if fix_yhe:
        p["YHe"] = YHE_FIXED
    return (lambda As: CF.spectra(dict(p), As)[0]), info


def job(args):
    meta, cand, r, ref = args
    out = dict(cand=key(r["tag"]))
    CS.init(True, True, 0.0)
    CF.init(True, True, "default", 1)
    L = CF.G["L"]
    # CAMB at its own best fit
    Hc = CS.prepare(meta, cand)
    Hc["dN0"] = r["dN"]
    a, w, N0, Om0, Oph0, _ = CS.tables(Hc, r["dN"])
    wm = CS.G["wr"] * np.exp(N0 - Hc["N_eq"])
    h = float(np.sqrt(wm / Om0))
    de = CS.make_de("fluid", a, w)
    cm, s_camb = chi2(L, lambda As: CS.camb_spectra(100 * h, r["wb"], wm - r["wb"], r["ns"], As, de)[0], None)
    cl0, _ = chi2(L, lambda As: CS.camb_spectra(ref["H0"], ref["wb"], ref["wc"], ref["ns"], As)[0], None)
    out["camb"] = dict(delta_cmb=cm - cl0, h=h, omega_r=float(CS.G["wr"]))
    spectra = {"camb": s_camb}
    # CLASS variants at the same point
    H = CF.load(meta, cand)
    for name, prec, zs, fy in (("class", "default", None, False), ("class_hi", "high", None, False),
                               ("class_Y", "default", None, True), ("class_z7_Y", "default", 3e7, True)):
        CF.G["precision"] = prec
        try:
            spec, info = class_model(H, r["dN"], r["wb"], r["ns"], zs, fy)
            c, s = chi2(L, spec, None)
            c0, _ = chi2(L, class_lcdm(ref), None)
            out[name] = dict(delta_cmb=c - c0, h=float(info["h"]), omega_r=float(CF.G["wr"]))
            spectra[name] = s
        except Exception as e:
            out[name] = dict(error=repr(e)[:200])
    CF.G["precision"] = "default"
    # where the spectra differ (CLASS default vs CAMB), relative, by l range
    if "class" in spectra:
        diff = {}
        for i, nm in ((0, "TT"), (2, "EE")):
            A = np.asarray(spectra["camb"][i])[:CF.LMAX + 1]
            B = np.asarray(spectra["class"][i])[:CF.LMAX + 1]
            n = min(len(A), len(B))
            l = np.arange(n)
            rel = np.abs(B[:n] / np.where(np.abs(A[:n]) > 0, A[:n], np.nan) - 1)
            for lo, hi in ((2, 30), (30, 500), (500, 1500), (1500, 2500)):
                m = (l >= lo) & (l < hi)
                diff[f"{nm} {lo}-{hi}"] = float(np.nanmax(rel[m])) if m.any() else None
        out["max_rel_diff_class_vs_camb"] = diff
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("ensemble"); ap.add_argument("camb_best")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="camb_class_diag.json")
    a = ap.parse_args()
    want = [(float(l1), float(al), int(k)) for l1, al, k in (c.split(":") for c in a.cands.split(","))]
    cb = json.load(open(a.camb_best))
    ref = cb["lcdm"]
    if "H0" not in ref:
        raise SystemExit(f"CAMB LCDM entry lacks H0; keys: {sorted(ref)}")
    res = {key(r["tag"]): r for r in cb["results"] if "error" not in r}
    picked = CS.select(a.fit, a.ensemble, 10 ** 6, None)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(a.ensemble))}
    jobs = []
    for m, c, _, t in picked:
        k = key(t)
        if k in want and k in res:
            meta = dict(m, N_state_early=cells[(t["l1"], t["l2"], t["alpha"])]["N_state_early"])
            jobs.append((meta, c, res[k], ref))
    missing = set(want) - {key(j[2]["tag"]) for j in jobs}
    if missing:
        print("not found in the CAMB results:", sorted(missing))
    with mp.get_context("spawn").Pool(min(a.procs, max(1, len(jobs)))) as pool:
        out = pool.map(job, jobs, chunksize=1)
    for o in out:
        l1, al, k = o["cand"]
        cols = []
        for nm in ("camb", "class", "class_hi", "class_Y", "class_z7_Y"):
            d = o.get(nm, {})
            cols.append(f"{nm} " + (f"{d['delta_cmb']:+.2f} (h {d['h']:.4f})" if "delta_cmb" in d else d.get("error", "-")))
        print(f"l1={l1:g} alpha={al:g} cand {k}: CMB delta chi2 at CAMB's best fit | " + " | ".join(cols), flush=True)
        if "max_rel_diff_class_vs_camb" in o:
            print("    largest |CLASS/CAMB - 1|: " + ", ".join(f"{k2} {v:.1e}" for k2, v in
                                                          o["max_rel_diff_class_vs_camb"].items() if v is not None))
    print("omega_r: CAMB", out[0]["camb"]["omega_r"] if out else None,
          "CLASS", out[0].get("class", {}).get("omega_r") if out else None)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
