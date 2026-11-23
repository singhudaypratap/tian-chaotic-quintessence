"""Posterior over the model parameters from the fit output of fit_survivors.py.

For each parameter cell the fit gives R = likelihood ratio to LambdaCDM averaged
over initial conditions on the attractor. With a prior that is uniform over the
listed cells, R is proportional to the posterior probability of the cell, and
the prior average of R is the Bayes factor of the model against LambdaCDM
(an upper estimate, because two nuisance parameters are profiled, not
marginalised, in each history's fit).

Usage: python3 posterior_grid.py fit_grid.json [fit_part4.json ...]
"""
import json
import sys

import numpy as np

cells = []
for path in sys.argv[1:]:
    for c in json.load(open(path))["cells"]:
        st = c.get("stats")
        cells.append(dict(l1=c["l1"], l2=c["l2"], alpha=c["alpha"], n=c["n_histories"],
                          n_cand=c["n_candidates"],
                          R=st["averaged_likelihood_ratio"] if st else 0.0,
                          R_se=st["averaged_likelihood_ratio_se"] if st else 0.0,
                          best=st["best_delta_chi2"] if st else None,
                          w0=st["weighted"]["w0"] if st else None,
                          dH=st["weighted"]["dH_max"] if st else None))
seen, uniq = set(), []
for c in cells:                                   # one entry per parameter point
    key = (c["l1"], c["l2"], c["alpha"])
    if key not in seen:
        seen.add(key)
        uniq.append(c)
cells = uniq
R = np.array([c["R"] for c in cells])
se = np.array([c["R_se"] for c in cells])
if R.sum() <= 0:
    sys.exit("no cell has a non-zero likelihood ratio")
post = R / R.sum()
B, B_se = R.mean(), np.sqrt((se ** 2).sum()) / len(R)
order = np.argsort(-post)
cum = np.cumsum(post[order])
print(f"{len(cells)} parameter cells, prior uniform over cells")
print(f"Bayes factor, model against LambdaCDM (upper estimate): {B:.2e} +- {B_se:.1e}   ln B = {np.log(B):+.2f}")
print("Jeffreys-scale reading of |ln B|: < 1 inconclusive, 1-2.5 weak, 2.5-5 moderate, > 5 strong")
print("\n  l1    alpha   posterior  cumulative  R (ratio to LCDM)   best dchi2  weighted w0      max dH/H")
for rank, i in enumerate(order):
    c = cells[i]
    if post[i] < 1e-4 and rank > 5:
        break
    w0 = f"{c['w0'][0]:.3f}+-{c['w0'][1]:.3f}" if c["w0"] else "-"
    dH = f"{c['dH'][0]:.4f}" if c["dH"] else "-"
    best = f"{c['best']:+.1f}" if c["best"] is not None else "-"
    print(f"{c['l1']:5.0f}  {c['alpha']:5.2f}   {post[i]:8.4f}   {cum[rank]:8.4f}   {c['R']:.2e}+-{c['R_se']:.0e}   {best:>8}   {w0:>14}  {dH:>8}")
for name in ("l1", "alpha"):
    vals = sorted({c[name] for c in cells})
    marg = [sum(post[i] for i, c in enumerate(cells) if c[name] == v) for v in vals]
    print(f"\nmarginal posterior of {name}: " + ", ".join(f"{v:g}: {m:.3f}" for v, m in zip(vals, marg)))
n68 = int(np.searchsorted(cum, 0.68) + 1)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"\nsmallest sets of cells holding 68% / 95% of the posterior: {n68} / {n95} cells")
print(f"cells with no candidate history at all: {sum(c['n_cand'] == 0 for c in cells)}")
json.dump(dict(bayes_factor=float(B), bayes_factor_se=float(B_se),
               cells=[dict(**c, posterior=float(p)) for c, p in zip(cells, post)]),
          open("posterior_grid.json", "w"), indent=1)
