"""Compare CAMB's dark-energy fraction with the history's own field fraction."""
import json, os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import camb
import cmb_spectrum as C

C.init(False)
fit = json.load(open("mcmb20.json"))["results"][0]
(meta, cand, dN0, tag), = C.select("mfit20.json", "m20.jsonl", 1)
H = C.prepare(meta, cand)
a, w, N0, Om0, Oph0, diag = C.tables(H, fit["dN"])
wm = C.G["wr"] * np.exp(N0 - H["N_eq"]); h = np.sqrt(wm / Om0)
# history's own field fraction: S table (early) and the late solution
edges = H["edges"]; Nc = H["Nc"]
O_hist_early = 1 - H["S"] ** 2
z_hist = np.exp(N0 - Nc) - 1
for kind in ("ppf", "fluid"):
    p = camb.set_params(H0=100 * h, ombh2=fit["wb"], omch2=wm - fit["wb"], mnu=0.0, nnu=3.044,
                        num_massive_neutrinos=0, omk=0, tau=C.TAU, As=C.AS0, ns=fit["ns"])
    p.DarkEnergy = C.make_de(kind, a, w)
    r = camb.get_background(p)
    zs = np.array([1e9, 1e8, 1e7, 1e6, 3e5, 1e5, 3e4, 1e4, 5e3, 1100, 100, 10, 1, 0])
    d = r.get_background_densities(1 / (1 + zs), vars=["de", "tot"])
    print(kind, "CAMB Omega_de at z =", " ".join(f"{z:.0e}:{v:.2e}" for z, v in zip(zs, d["de"] / d["tot"])))
print("history Omega_phi at the same z (bin values):")
print("     ", " ".join(f"{z:.0e}:{np.interp(np.log1p(z), np.log1p(z_hist)[::-1], O_hist_early[::-1]):.2e}"
                       for z in [1e9, 1e8, 1e7, 1e6, 3e5, 1e5, 3e4, 1e4, 5e3, 1100]))
print("max history Omega_phi for 5000<z<1e8:", O_hist_early[(z_hist > 5e3) & (z_hist < 1e8)].max())
print("table: first a", a[0], "w first values", np.round(w[:8], 3))
