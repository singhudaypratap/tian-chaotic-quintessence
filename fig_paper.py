"""Figures 1 and 2 of the paper.

Fig. 1 (fig1_histories.png): field fraction Omega_phi(z) from z = 1e9 to today for three CLASS-fitted
  histories: the Part IV best fit (complete early episode), the partial-episode history (30, 0.08) cand
  1023, and the best clean window history (10, 0.6) cand 414. Shaded: the nucleosynthesis and
  recombination test windows; hatched: the interval between them that no background test probes.
Fig. 2 (fig2_initial_conditions.png):
  (a) joint pass rate on the radiation-era attractor after 300, 1000, 2000 e-folds, one line per cell
      (burn-scan output, 50k histories per cell): the ensemble does not settle;
  (b) clean survival fraction against pre-history length (phys_burn_v3.jsonl, 200k per cell), grid total
      and selected cells.

Usage:
  python3 fig_paper.py histories part4.jsonl class_field_part4_v3.json grid_l2_0.01.jsonl class_field_grid_l2_0.01_v3.json
  python3 fig_paper.py mixing "burn_scan*.jsonl" phys_burn_v3.jsonl
"""
import glob
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def histories(part4_ens, part4_class, grid_ens, grid_class):
    import tian
    picks = []
    for ens, cres, want in ((part4_ens, part4_class, None), (grid_ens, grid_class, [(30.0, 0.08, 1023), (10.0, 0.6, 414)])):
        cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(ens))}
        rs = [r for r in json.load(open(cres))["results"] if "error" not in r]
        if want is None:
            rs = [min(rs, key=lambda r: r["chi2"])]
        else:
            rs = [r for w in want for r in rs if (r["tag"]["l1"], r["tag"]["alpha"], r["tag"]["cand_index"]) == w]
        for r in rs:
            t = r["tag"]
            picks.append((cells[(t["l1"], t["l2"], t["alpha"])], r))
    labels = ["Part IV best fit (complete early episode)", "partial episode, (30, 0.08)", "clean, late thawing, (10, 0.6)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvspan(3.7e8, 1e9, color="0.85", lw=0)
    ax.axvspan(50, 5000, color="0.85", lw=0)
    ax.axvspan(5000, 3.7e8, facecolor="none", edgecolor="0.75", hatch="///", lw=0)
    for k, (cell, r) in enumerate(picks):
        t = r["tag"]
        cand = cell["candidates"][t["cand_index"]]
        s0 = tian.with_u(cand["early"])
        N_s, N_eq, N_end = cell["N_state_early"], cell["N_eq"], cell["N_end"]
        N0 = N_end + r["dN"]
        from scipy.integrate import solve_ivp
        sol = solve_ivp(tian.rhs_fullu, [N_s, N0], s0, args=(t["l1"], t["l2"], t["alpha"], N_eq),
                        method="DOP853", rtol=1e-11, atol=[1e-300, 1e-300, 1e-14, 1e-16], dense_output=True)
        N = np.linspace(N_s, N0, 6000)
        x, y, _, _ = sol.sol(N)
        z = np.exp(N0 - N) - 1
        m = z > 0.01
        ax.plot(1 + z[m], np.maximum(x[m] ** 2 + y[m] ** 2, 1e-12), color=OKABE[k], lw=1.6,
                label=f"{labels[k]}: $\\Delta\\chi^2$ = {r['delta_chi2']:+.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e9, 1.01); ax.set_ylim(1e-10, 3)
    ax.set_xlabel("1 + z"); ax.set_ylabel(r"$\Omega_\phi$")
    ax.text(6.1e8, 3e-10, "BBN", ha="center", va="bottom", rotation=90, fontsize=8, color="0.3")
    ax.text(500, 2.5e-10, "recombination\nwindow", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.text(1.4e6, 2.5e-10, "not probed by background tests", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.tight_layout(); fig.savefig("fig1_histories.png", dpi=200)
    print("wrote fig1_histories.png")


def mixing(burn_glob, phys_path):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax = axs[0]
    n = 0
    for path in sorted(glob.glob(burn_glob)):
        for line in open(path):
            d = json.loads(line)
            b = [r["burn"] for r in d["rows"]]
            v = [r["rates"]["all_three"] for r in d["rows"]]
            if max(v) <= 0:
                continue
            hl = (d["l1"], d["alpha"]) in ((25, 0.6), (20, 0.8), (10, 0.24))
            ax.plot(b, v, "-o", ms=3, lw=1.6 if hl else 0.6, color=OKABE[[(25, 0.6), (20, 0.8), (10, 0.24)].index((d["l1"], d["alpha"]))] if hl else "0.7",
                    label=f"({d['l1']:g}, {d['alpha']:g})" + (" Part IV" if (d["l1"], d["alpha"]) == (10, 0.24) else "") if hl else None, zorder=3 if hl else 1)
            n += 1
    ax.set_yscale("log"); ax.set_xlabel("e-folds on the radiation-era attractor"); ax.set_ylabel("joint pass rate")
    ax.set_title(f"(a) attractor, {n} cells", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    tot = {}
    rows = [json.loads(l) for l in open(phys_path)]
    for d in rows:
        for r in d["rows"]:
            c = tot.setdefault(r["burn"], [0, 0])
            c[0] += r["clean_counts"]["below_0p01"]; c[1] += r["n_valid"]
    bs = sorted(tot)
    for k, cell in enumerate(((10, 0.6), (8, 0.6), (14, 0.8))):
        for d in rows:
            if (d["l1"], d["alpha"]) == cell:
                bb = np.array([r["burn"] for r in d["rows"]])
                vv = np.array([r["clean_rates"]["below_0p01"] for r in d["rows"]])
                nz = vv > 0
                ax.plot(bb[nz], vv[nz], "-o", ms=3, lw=1.0, color=OKABE[k], label=f"({cell[0]}, {cell[1]})")
                # zero clean histories out of n: drawn as an open triangle at the 95% upper limit 3/n
                ax.plot(bb[~nz], [3.0 / r["n_valid"] for r, z in zip(d["rows"], nz) if not z], "v",
                        mfc="none", ms=6, color=OKABE[k])
    ax.plot(bs, [tot[b][0] / tot[b][1] for b in bs], "-s", ms=4, lw=2.0, color="k", label="grid total")
    ax.set_yscale("log"); ax.set_xlabel("pre-history (radiation-era e-folds)"); ax.set_ylabel("clean survival fraction")
    ax.set_xticks(bs); ax.set_title("(b) physical initial conditions", fontsize=10, loc="left")
    ax.text(0.03, 0.95, "open triangles: none found, 95% upper limit", transform=ax.transAxes, fontsize=7, color="0.4")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("fig2_initial_conditions.png", dpi=200)
    print("wrote fig2_initial_conditions.png")


if __name__ == "__main__":
    if sys.argv[1] == "histories":
        histories(*sys.argv[2:6])
    elif sys.argv[1] == "mixing":
        mixing(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
