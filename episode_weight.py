"""How much of the grid posterior comes from histories with an early field episode?

The posterior (posterior_grid.py) uses CMB distance priors, which do not see episodes between
z = 5000 and 3.7e8. Full-spectrum CLASS fits exclude complete episodes (peak >= 0.99) by
delta chi2 >= 30 (RESULTS section 21). This script splits the posterior weight of every fitted
history, exp(-delta chi2 / 2) / n_histories of its cell, by the history's episode peak, and gives
the Bayes factor and the marginals again using only histories below a peak cut.

Usage: python3 episode_weight.py FIT_all.json ENSEMBLE.jsonl [--cuts 0.01,0.5,0.99]
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("ensemble")
    ap.add_argument("--cuts", default="0.01,0.5,0.99")
    a = ap.parse_args()
    cuts = [float(c) for c in a.cuts.split(",")]
    d = json.load(open(a.fit))
    L = d["lcdm"]["chi2"]
    fits = {(c["l1"], c["l2"], c["alpha"]): c for c in d["cells"]}
    ncell = len(fits)
    W = defaultdict(float)                 # (cut or 'all') -> total weight
    Wcell = defaultdict(lambda: defaultdict(float))
    band = defaultdict(float)
    for line in open(a.ensemble):
        cell = json.loads(line)
        k = (cell["l1"], cell["l2"], cell["alpha"])
        fc = fits.get(k)
        if fc is None or not fc["fits"]:
            continue
        edges = np.asarray(cell["S_edges"])
        Nc = 0.5 * (edges[1:] + edges[:-1])
        cands = cell["candidates"]
        nh = fc["n_histories"]
        # fit_survivors.py fits at most --max candidates per cell; scale the fitted ones up to all
        # candidates, exactly as fit_survivors/posterior_grid do (an unbiased subset is assumed)
        nfit = sum(1 for r in fc["fits"] if isinstance(r, dict) and "error" not in r)
        scale = fc.get("n_candidates", nfit) / max(nfit, 1)
        for r in fc["fits"]:
            if not isinstance(r, dict) or "error" in r:
                continue
            c = cands[r["cand_index"]]
            N0 = cell["N_end"] + r["dN"]
            z = np.exp(N0 - Nc) - 1
            sel = (z > 5e3) & (z < 3.7e8)
            S = np.asarray(c["S"])
            peak = float((1 - S[sel] ** 2).max()) if sel.any() else 0.0
            w = scale * math.exp(-0.5 * (r["chi2"] - L)) / nh / ncell     # cells equally likely a priori
            W["all"] += w
            Wcell["all"][k] += w
            for cu in cuts:
                if peak < cu:
                    W[cu] += w
                    Wcell[cu][k] += w
            b = "<0.01" if peak < 0.01 else "0.01-0.5" if peak < 0.5 else "0.5-0.99" if peak < 0.99 else ">=0.99"
            band[b] += w
    print(f"{a.fit}: Bayes factor with all fitted histories {W['all']:.3e} (ln B {math.log(W['all']):+.2f})")
    for b in ("<0.01", "0.01-0.5", "0.5-0.99", ">=0.99"):
        print(f"  share of posterior weight with episode peak {b:9s}: {band[b] / W['all']:.3f}")
    for cu in cuts:
        if W[cu] > 0:
            print(f"  keeping only peak < {cu:g}: Bayes factor {W[cu]:.3e} (ln B {math.log(W[cu]):+.2f})")
            tot = sum(Wcell[cu].values())
            for name, idx in (("l1", 0), ("alpha", 2)):
                m = defaultdict(float)
                for k, v in Wcell[cu].items():
                    m[k[idx]] += v / tot
                print(f"    marginal of {name}: " + ", ".join(f"{x:g}: {m[x]:.3f}" for x in sorted(m)))
        else:
            print(f"  keeping only peak < {cu:g}: no weight")


if __name__ == "__main__":
    main()
