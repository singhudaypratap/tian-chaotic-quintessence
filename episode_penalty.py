"""CMB penalty against the height of the early field episode (5000 < z < 3.7e8).

Joins CLASS fits (class_field.py output) with the CAMB results for the same candidates, which carry
the episode peak, and plots CMB delta chi2 (CLASS) against the peak. Prints how many candidates
with a given peak stay within +4 of LCDM in the CMB, so the claim about episodes can be stated
with its actual boundary.

Usage: python3 episode_penalty.py CLASS1.json [CLASS2.json ...] --camb CAMB1.json [CAMB2.json ...]
"""
import argparse
import json

import numpy as np


def key(t):
    return (t["l1"], t["l2"], t["alpha"], t["cand_index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("class_files", nargs="+")
    ap.add_argument("--camb", nargs="+", required=True)
    ap.add_argument("--out", default="episode_penalty.png")
    a = ap.parse_args()
    peak = {}
    for f in a.camb:
        for r in json.load(open(f))["results"]:
            if "error" not in r and "gap_peak" in r:
                peak[key(r["tag"])] = r["gap_peak"]
    pts = []
    for f in a.class_files:
        d = json.load(open(f))
        ref = d["lcdm"]
        for r in d["results"]:
            if "error" in r or key(r["tag"]) not in peak:
                continue
            pts.append((peak[key(r["tag"])], r["chi2_cmb"] - ref["chi2_cmb"], r["tag"]["l1"], r["tag"]["alpha"]))
    if not pts:
        raise SystemExit("no candidate appears in both the CLASS and the CAMB files")
    P = np.array([(p, d) for p, d, _, _ in pts])
    print(f"{len(pts)} candidates with both a CLASS fit and an episode peak")
    for lo, hi in ((0, 0.01), (0.01, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
        m = (P[:, 0] >= lo) & (P[:, 0] < hi)
        if m.any():
            print(f"  peak in [{lo:g}, {hi:g}): {m.sum():3d} candidates; CMB delta chi2 min {P[m, 1].min():+.1f}, "
                  f"median {np.median(P[m, 1]):+.1f}; within +4 of LCDM: {(P[m, 1] <= 4).sum()}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(np.maximum(P[:, 0], 1e-6), np.maximum(P[:, 1], 0.1), s=14, color="#2a6f97")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(4, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel(r"peak $\Omega_\phi$ for $5000<z<3.7\times10^8$ (values below $10^{-6}$ drawn at $10^{-6}$)")
        ax.set_ylabel(r"CMB $\Delta\chi^2$ vs $\Lambda$CDM (CLASS; below 0.1 drawn at 0.1)")
        fig.tight_layout(); fig.savefig(a.out, dpi=150)
        print("plot:", a.out)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
