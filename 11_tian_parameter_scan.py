"""Scan of Tian's model over (l1, alpha) at l2 = 0.01, with independent sampling.

Each history starts from a state on the radiation-era attractor (states taken
4 e-folds apart along one long trajectory) and is carried through equality to
today, with N_eq = 30 + U(0, 5). An earlier version varied only N_eq from one
initial state; those histories were strongly correlated and its results are
superseded.

Per history: Omega_phi and w_phi today, and the peak of Omega_phi in
  bbn  : 3.7e8 < z < 1.03e10  (T from 2e10 K to 1e9 K, Tian's main BBN era)
  cmb  : 50 < z < 5000
Per cell: the three marginal pass rates, the joint count, and the product of
the marginals (valid only if the three epochs are independent).

Resumable. Usage: python3 11_tian_parameter_scan.py [seconds_budget] [n_runs]
"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
import tian as t

L2 = 0.01
PATH = "results_tian_scan.jsonl"


def one(args):
    l1, al, N_eq, s0 = args
    try:
        h = t.history_from_state(l1, L2, al, N_eq, np.array(s0))
    except Exception as e:
        return dict(error=str(e))
    z, O = h["z"], h["Ophi"]
    if not np.all(np.isfinite(O)):
        return dict(error="non-finite")
    def mx(lo, hi):
        return float(O[(z > lo) & (z < hi)].max())
    return dict(Ophi0=float(O[-1]), w0=float(h["wphi"][-1]),
                Ophi_rec=float(O[int(np.argmin(np.abs(z - 1100.0)))]),
                bbn_max=mx(3.7e8, 1.03e10), cmb_max=mx(50, 5000))


def do_cell(cell, pool, n):
    l1, al = cell
    rng = np.random.default_rng(int(1000 * l1 + 1e6 * al))
    states = t.attractor_states(l1, L2, al, n)
    jobs = [(l1, al, 30.0 + rng.uniform(0, 5), s.tolist()) for s in states]
    res = [r for r in pool.map(one, jobs, chunksize=20) if "error" not in r]
    lyap = {k: float(t.lyap1(np.array([0.3, 0.3, 0.5]), (l1, L2, al, w), T=400.0, burn=200.0)[0])
            for k, w in (("rad", 1 / 3), ("mat", 0.0))}
    return dict(l1=l1, l2=L2, alpha=al, n=len(res), lyap=lyap, runs=res)


def rates(c):
    A = {k: np.array([r[k] for r in c["runs"]]) for k in c["runs"][0]}
    bbn = A["bbn_max"] < 0.04
    cmb = A["cmb_max"] < 0.02
    today = (A["Ophi0"] > 0.6) & (A["Ophi0"] < 0.8) & (A["w0"] < -0.8)
    return dict(p_bbn=bbn.mean(), p_cmb=cmb.mean(), p_today=today.mean(),
                n_bbn_cmb=int((bbn & cmb).sum()), n_cmb_today=int((cmb & today).sum()),
                n_bbn_today=int((bbn & today).sum()), n_all=int((bbn & cmb & today).sum()),
                product=float(bbn.mean() * cmb.mean() * today.mean()))


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    cells = [(l1, al) for l1 in (6.0, 10.0, 14.0, 20.0, 30.0) for al in (0.08, 0.15, 0.24, 0.4, 0.8)]
    done = set()
    if os.path.exists(PATH):
        done = {(c["l1"], c["alpha"]) for c in map(json.loads, open(PATH))}
    t0 = time.time()
    with Pool(2) as pool, open(PATH, "a") as fh:
        for cell in cells:
            if cell in done:
                continue
            if time.time() - t0 > budget:
                print("budget reached; rerun to continue")
                break
            c = do_cell(cell, pool, n)
            fh.write(json.dumps(c) + "\n")
            fh.flush()
            r = rates(c)
            print(f"l1={cell[0]:4.0f} alpha={cell[1]:.2f} n={c['n']} lyap rad {c['lyap']['rad']:+.2f} mat {c['lyap']['mat']:+.2f} | "
                  f"pass BBN {100*r['p_bbn']:5.1f}%  CMB {100*r['p_cmb']:5.1f}%  today {100*r['p_today']:5.1f}% | "
                  f"joint: bbn&cmb {r['n_bbn_cmb']:3d} cmb&today {r['n_cmb_today']:3d} bbn&today {r['n_bbn_today']:3d} all {r['n_all']} | "
                  f"product {r['product']:.1e}", flush=True)
    print(f"cells finished: {sum(1 for _ in open(PATH))}/{len(cells)}")
