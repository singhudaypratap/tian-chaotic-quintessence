"""Rebuild the clean Part IV histories from physical initial conditions with no pre-history.

burn_scan.py (--cells 10:0.24 --n 2000000 --burns 0,20,40, seed 7, uniform prior) found 170 histories
passing all three background tests at 0 e-folds, 15 of them clean (peak field fraction < 0.01 for
5000 < z < 3.7e8). It stores only counts. This script redraws the same 2,000,000 starting states
(same seed, same order, same device arithmetic as burn_scan.scan_cell), evolves them in chunks, checks
that it recovers 170 and 15, and writes the clean ones as an ensemble file that fit_survivors.py and
class_field.py read.

Usage: python3 part4_clean.py [--chunk 500000] [--out part4_clean0.jsonl]
"""
import argparse
import json
import math

import numpy as np

import gpu_ensemble as g

L1, L2, AL, N, SEED, DN = 10.0, 0.01, 0.24, 2_000_000, 7, 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--out", default="part4_clean0.jsonl")
    ap.add_argument("--device", default="cuda" if g.HAVE_TORCH and g.torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if not g.HAVE_TORCH:
        a.device = "numpy"
    B = g.Backend(a.device)
    # identical draw to burn_scan.scan_cell (uniform prior, Omega_phi in 0.01-0.9)
    rng = np.random.default_rng(SEED)
    r = np.sqrt(rng.uniform(0.01, 0.9, N))
    ang = rng.uniform(0.1, math.pi - 0.1, N)
    x, y = B.array(r * np.cos(ang)), B.array(r * np.sin(ang))
    th = B.array(rng.uniform(0, 2 * math.pi, N))
    u = B.log(1.0 - x * x - y * y)
    st = tuple(B.numpy(v).copy() for v in (x, y, th, u))
    del x, y, th, u

    cands, n_all3, n_valid, meta = [], 0, 0, None
    for s in range(0, N, a.chunk):
        part = tuple(v[s:s + a.chunk] for v in st)
        o, _, snap, (edges, tables, tables_w) = g.run(L1, L2, AL, len(part[0]), DN, 0.0, a.device, SEED,
                                                       keep_states=True, start_state=part)
        res, _ = g.summarise(o)
        n_valid += res["n_valid"]
        g3 = (o["bbn_max"] < 0.04) & (o["cmb_max"] < 0.02) & (o["Ophi0"] > 0.6) & (o["Ophi0"] < 0.8) & (o["w0"] < -0.8)
        n_all3 += int(g3.sum())
        se = o.pop("_snap_early")
        if meta is None:
            meta = dict(N_snapshot=snap[3], N_eq=g.N_EQ, N_end=g.N_EQ + g.LN_1PZEQ, N_state_early=se[3],
                        S_edges=[float(e) for e in edges])
        for i in np.where(g3 & (o["gap_max"] < 0.01))[0]:
            cands.append(dict(x=float(snap[0][i]), y=float(snap[1][i]), theta=float(snap[2][i]),
                              Ophi0=float(o["Ophi0"][i]), w0=float(o["w0"][i]), gap_max=float(o["gap_max"][i]),
                              history_index=int(s + i),
                              S=[round(float(t[i]), 5) for t in tables],
                              W=[round(float(t[i]), 5) for t in tables_w],
                              early=[float(se[0][i]), float(se[1][i]), float(se[2][i]), float(se[4][i])]))
        print(f"histories {s}-{s + len(part[0])}: all three so far {n_all3}, clean so far {len(cands)}", flush=True)

    print(f"valid {n_valid}; all three {n_all3} (burn_scan: 170); clean {len(cands)} (burn_scan: 15)", flush=True)
    if n_all3 != 170 or len(cands) != 15:
        print("WARNING: counts differ from burn_scan; the files are still written, but report the counts used.")
    rec = dict(l1=L1, l2=L2, alpha=AL, n=N, dN=DN, burn=0.0, seed=SEED, prior="uniform 0.01-0.9",
               n_candidates=len(cands), candidates=cands, **meta)
    with open(a.out, "w") as fh:
        fh.write(json.dumps(rec) + "\n")
    for c in cands:
        print(f"  history {c['history_index']}: Omega_phi0 {c['Ophi0']:.3f}, w0 {c['w0']:.3f}, gap peak {c['gap_max']:.1e}")


if __name__ == "__main__":
    main()
