"""Do the two stored states of a candidate lie on one trajectory?

Each candidate stores an early state (z = 1e9, used by CLASS) and a late snapshot (z = 50, used by
CAMB and the fits). Evolving the early state to the snapshot time should reproduce the snapshot.
Two integrators are used:
  adaptive : class_field.load (DOP853, u form, rtol 1e-11), exactly what the CLASS stage uses
  rk4      : the GPU scheme (fixed-step RK4, u form, step 2e-3) on the CPU in float64
Prints the largest difference in (x, y) and in Omega_phi at the snapshot, and flags candidates
where it exceeds 1e-6.

Usage: python3 state_consistency.py FIT_all.json ENSEMBLE.jsonl [--top 50] [--cands "10:0.8:1937,..."]
"""
import argparse
import json

import numpy as np

import class_field as CF
import cmb_spectrum as CS
import gpu_ensemble as g


def key(t):
    return (float(t["l1"]), float(t["alpha"]), int(t["cand_index"]))


def rk4_replay(cell, early, N_from, N_to, dN=2e-3):
    B = g.Backend("numpy")
    x, y, th, u = (B.array(np.array([v], float)) for v in early)
    l1, l2, al = cell["l1"], cell["l2"], cell["alpha"]
    lb, dl = 0.5 * (l1 + l2), 0.5 * (l1 - l2)
    N = N_from
    n = int(round((N_to - N_from) / dN))
    h = (N_to - N_from) / n
    wb = lambda N: (1 / 3) / (1 + np.exp(N - cell["N_eq"]))
    for _ in range(n):
        x, y, th, u = g.rk4(B, x, y, th, u, wb(N), wb(N + h / 2), wb(N + h), h, lb, dl, al)
        N += h
    return float(x[0]), float(y[0]), float(th[0]), float(u[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("ensemble")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--cands", default=None)
    a = ap.parse_args()
    CF.init(True, True, "default", 1)
    picked = CS.select(a.fit, a.ensemble, a.top if not a.cands else 10 ** 6, None)
    cells = {(c["l1"], c["l2"], c["alpha"]): c for c in map(json.loads, open(a.ensemble))}
    want = None if not a.cands else {(float(l1), float(al), int(k)) for l1, al, k in (c.split(":") for c in a.cands.split(","))}
    bad = 0
    for m, cand, _, t in picked:
        if want is not None and key(t) not in want:
            continue
        cell = cells[(t["l1"], t["l2"], t["alpha"])]
        meta = dict(m, N_state_early=cell["N_state_early"])
        Ns, Nsnap = cell["N_state_early"], cell["N_snapshot"]
        snap = np.array([cand["x"], cand["y"]])
        H = CF.load(meta, cand)
        xa, ya, tha, ua = H["sol"].sol(Nsnap)
        da = float(np.max(np.abs(np.array([xa, ya]) - snap)))
        xr, yr, thr, ur = rk4_replay(cell, CF.tian.with_u(cand["early"]), Ns, Nsnap)
        dr = float(np.max(np.abs(np.array([xr, yr]) - snap)))
        O = snap @ snap
        flag = da > 1e-6 or dr > 1e-6
        bad += flag
        print(f"l1={t['l1']:g} alpha={t['alpha']:g} cand {t['cand_index']}: Omega_phi at z=1e9 "
              f"{cand['early'][0] ** 2 + cand['early'][1] ** 2:.1e}, at snapshot {O:.3f} | "
              f"replay from z=1e9 -> |d(x,y)| adaptive {da:.1e}, rk4 {dr:.1e} "
              f"(Omega_phi {xa * xa + ya * ya:.3f} / {xr * xr + yr * yr:.3f}){'  <-- MISMATCH' if flag else ''}",
              flush=True)
    print(f"mismatched: {bad}")


if __name__ == "__main__":
    main()
