"""Histories that pass all three tests in the scan (BBN, recombination window,
today): are they distinguishable from LambdaCDM at late times?

For each, the history is regenerated from the same attractor state and the
late-time behaviour is measured: w_phi over z < 2, and the largest fractional
difference in H(z) from flat LambdaCDM with the same matter density, z < 3."""
import json
import numpy as np
import tian as t

L2 = 0.01
out = []
for c in map(json.loads, open("results_tian_scan.jsonl")):
    runs = c["runs"]
    A = {k: np.array([r[k] for r in runs]) for k in runs[0]}
    ok = (A["bbn_max"] < 0.04) & (A["cmb_max"] < 0.02) & (A["Ophi0"] > 0.6) & (A["Ophi0"] < 0.8) & (A["w0"] < -0.8)
    if not ok.any() or c["n"] != 800:
        continue
    l1, al = c["l1"], c["alpha"]
    rng = np.random.default_rng(int(1000 * l1 + 1e6 * al))
    states = t.attractor_states(l1, L2, al, 800)
    Neqs = [30.0 + rng.uniform(0, 5) for _ in states]
    for i in np.where(ok)[0]:
        h = t.history_from_state(l1, L2, al, Neqs[i], states[i], n_out=20000)
        z, O, w = h["z"], h["Ophi"], h["wphi"]
        assert abs(O[-1] - runs[i]["Ophi0"]) < 5e-3, (O[-1], runs[i]["Ophi0"])
        late = z < 3
        a = 1 / (1 + z[late])
        wb = t.w_bg(h["N"][late], Neqs[i])
        Om = (1 - O[late]) * (1 - 3 * wb)
        Om0, Or0 = Om[-1], (1 - O[-1]) * 3 * wb[-1]
        E = np.sqrt(Om0 * a ** -3 / Om)
        El = np.sqrt(Om0 * a ** -3 + Or0 * a ** -4 + 1 - Om0 - Or0)
        z2 = z < 2
        rec = dict(l1=l1, alpha=al, Ophi0=float(O[-1]), w0=float(w[-1]),
                   w_min=float(w[z2].min()), w_max=float(w[z2].max()),
                   dH_max=float(np.max(np.abs(E / El - 1))),
                   Ophi_peak_2_50=float(O[(z > 2) & (z < 50)].max()))
        out.append(rec)
        print(f"l1={l1:4.0f} alpha={al:.2f}: Omega_phi0={rec['Ophi0']:.3f} w0={rec['w0']:.3f} "
              f"w(z<2) in [{rec['w_min']:.3f},{rec['w_max']:.3f}] max|dH/H|(z<3)={rec['dH_max']:.4f} "
              f"peak Omega_phi 2<z<50: {rec['Ophi_peak_2_50']:.4f}")
json.dump(out, open("results_passing_histories.json", "w"), indent=1)
d = np.array([r["dH_max"] for r in out]); w0 = np.array([r["w0"] for r in out])
print(f"{len(out)} passing histories; max|dH/H| median {np.median(d):.4f}, range {d.min():.4f}-{d.max():.4f}; "
      f"w0 range {w0.min():.3f} to {w0.max():.3f}; with dH/H > 1%: {(d > 0.01).sum()}")
