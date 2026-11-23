"""Burn-in convergence for every grid cell.

For each cell the same n histories are burned once, continuously, on the radiation-era attractor.
At each checkpoint (default 300, 1000, 2000 e-folds) a copy of the states is evolved through the
full history and the three tests are applied. If the burn is long enough, the pass rates stop
moving between checkpoints: shifts of a few sigma at most, and a KS test on Omega_phi of the burned
states that is unremarkable. A cell that fails is reported as NOT CONVERGED; its grid rates at the
shortest burn are not attractor rates, and it is rerun with the burn printed at the end.

A largest Lyapunov exponent at the last checkpoint is also printed (positive: chaotic).

Usage (one process per shard, all sharing the GPU):
  python3 burn_scan.py --shard 0/3 --out burn_scan_0.jsonl
Options: --n 50000 --burns 300,1000,2000 --l2 0.01 --cells "20:0.24,25:0.4" (subset)
"""
import argparse
import json
import math

import numpy as np
from scipy.stats import ks_2samp

import gpu_ensemble as g

L1S = (6, 8, 10, 12, 14, 17, 20, 25, 30)
ALS = (0.08, 0.15, 0.24, 0.4, 0.6, 0.8)
KEYS = ("bbn", "recomb", "today", "all_three")


def draw_omega(rng, n, prior, lo, hi):
    """Initial field fraction Omega_phi = x^2 + y^2.
    uniform: uniform in [lo, hi] (the default used throughout: 0.01-0.9)
    log    : uniform in log10 between lo and hi (e.g. 1e-6 to 0.9: a field that usually starts small)"""
    if prior == "uniform":
        return rng.uniform(lo, hi, n)
    if prior == "log":
        return 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
    raise ValueError(prior)


def scan_cell(l1, l2, al, n, dN, burns, device, seed, prior="uniform", om_lo=0.01, om_hi=0.9):
    B = g.Backend(device)
    rng = np.random.default_rng(seed)
    r = np.sqrt(draw_omega(rng, n, prior, om_lo, om_hi))
    ang = rng.uniform(0.1, math.pi - 0.1, n)
    x, y = B.array(r * np.cos(ang)), B.array(r * np.sin(ang))
    th = B.array(rng.uniform(0, 2 * math.pi, n))
    u = B.log(1.0 - x * x - y * y)
    lbar, dl = 0.5 * (l1 + l2), 0.5 * (l1 - l2)
    done, rows, prev = 0.0, [], None
    for burn in burns:
        for _ in range(int(round((burn - done) / dN))):
            x, y, th, u = g.rk4(B, x, y, th, u, 1 / 3, 1 / 3, 1 / 3, dN, lbar, dl, al)
        done = burn
        st = tuple(B.numpy(v).copy() for v in (x, y, th, u))
        o, _, _, _ = g.run(l1, l2, al, n, dN, 0.0, device, seed, start_state=st)
        res, _ = g.summarise(o)
        O = st[0] ** 2 + st[1] ** 2
        O = O[np.isfinite(O)]
        g3 = (o["bbn_max"] < 0.04) & (o["cmb_max"] < 0.02) & (o["Ophi0"] > 0.6) & (o["Ophi0"] < 0.8) & (o["w0"] < -0.8)
        nv = max(res["n_valid"], 1)
        # survivors without a complete early takeover (peak field fraction for 5000 < z < 3.7e8):
        # complete domination (peak >= 0.99) is excluded by the full CMB spectrum (RESULTS section 21)
        clean = dict(below_0p01=int(np.sum(g3 & (o["gap_max"] < 0.01))),
                     below_0p5=int(np.sum(g3 & (o["gap_max"] < 0.5))),
                     below_0p99=int(np.sum(g3 & (o["gap_max"] < 0.99))))
        clean_rate = {k: v / nv for k, v in clean.items()}
        row = dict(burn=burn, n_valid=res["n_valid"], clean_counts=clean, clean_rates=clean_rate,
                   rates={k: res[k]["rate"] for k in KEYS}, se={k: res[k]["binomial_se"] for k in KEYS},
                   counts={k: res[k].get("count", None) for k in KEYS})
        if prev is not None:
            ks = ks_2samp(prev[1], O)
            z = {k: (row["rates"][k] - prev[0]["rates"][k]) /
                 max(math.hypot(row["se"][k], prev[0]["se"][k]), 1e-300) for k in KEYS}
            row["vs_previous"] = dict(ks_p=float(ks.pvalue), z=z)
        rows.append(row)
        prev = (row, O)
    try:
        lam = float(g.lyapunov(l1, l2, al, 1 / 3, device, seed, m=500, burn=burns[-1])["mean"])
    except Exception as e:                       # the exponent is a side diagnostic
        lam = None
    return rows, lam


def verdict(rows, zmax=3.0):
    """Converged if the last step moves no rate by more than zmax sigma."""
    last = rows[-1].get("vs_previous")
    if last is None:
        return True
    return all(abs(v) <= zmax for v in last["z"].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--dN", type=float, default=2e-3)
    ap.add_argument("--burns", default="300,1000,2000")
    ap.add_argument("--cells", default=None, help='subset, e.g. "20:0.24,25:0.4"')
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if g.HAVE_TORCH and g.torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="burn_scan.jsonl")
    ap.add_argument("--prior", choices=("uniform", "log"), default="uniform",
                    help="distribution of the initial field fraction (see draw_omega)")
    ap.add_argument("--omega-range", default="0.01,0.9", help="lo,hi of the initial field fraction")
    a = ap.parse_args()
    if not g.HAVE_TORCH:
        a.device = "numpy"
    burns = [float(b) for b in a.burns.split(",")]
    cells = ([tuple(float(v) for v in c.split(":")) for c in a.cells.split(",")] if a.cells
             else [(l1, al) for l1 in L1S for al in ALS])
    si, sk = (int(v) for v in a.shard.split("/"))
    cells = [c for i, c in enumerate(cells) if i % sk == si]
    with open(a.out, "a") as f:
        for l1, al in cells:
            olo, ohi = (float(v) for v in a.omega_range.split(","))
            rows, lam = scan_cell(l1, a.l2, al, a.n, a.dN, burns, a.device, a.seed, a.prior, olo, ohi)
            ok = verdict(rows)
            r0, r1 = rows[0]["rates"], rows[-1]["rates"]
            zl = rows[-1].get("vs_previous", {}).get("z", {})
            cr = rows[-1]["clean_rates"]
            print(f"l1={l1:g} alpha={al:g}: clean survivors at burn {burns[-1]:g}: peak<0.01 {cr['below_0p01']:.1e}, "
                  f"peak<0.99 {cr['below_0p99']:.1e}; all three {r0['all_three']:.2e} (burn {burns[0]:g}) -> "
                  f"{r1['all_three']:.2e} (burn {burns[-1]:g}); today {100*r0['today']:.2f}% -> {100*r1['today']:.2f}%; "
                  f"last step shifts " + ", ".join(f"{k} {v:+.1f}" for k, v in zl.items())
                  + f"; KS p {rows[-1].get('vs_previous', {}).get('ks_p', float('nan')):.3f}; "
                  f"Lyapunov {lam if lam is None else round(lam, 3)}; {'converged' if ok else 'NOT CONVERGED'}",
                  flush=True)
            f.write(json.dumps(dict(l1=l1, l2=a.l2, alpha=al, n=a.n, burns=burns, rows=rows,
                                    prior=a.prior, omega_range=[olo, ohi],
                                    lyapunov=lam, converged=ok)) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
