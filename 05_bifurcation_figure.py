"""Bifurcation diagram: Poincare-section values of x (theta = 0 mod 2 pi)
against nu at lam = 10, A = 0.7, w = 0."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import modsys as m

lam, A, w = 10.0, 0.7, 0.0
nus = np.linspace(7.0, 10.4, 171)
rows, s = [], np.array([0.3, 0.3, 0.1])
for nu in nus:
    p = (lam, A, float(nu), w)
    pts, _ = m.section_points(np.array([0.3, 0.3, 0.1]), p, n_pts=80, burn=500.0, max_T=600.0)
    rows.append(dict(nu=float(nu), x=pts[:, 0].tolist()))
json.dump(rows, open("results_bifurcation.json", "w"))

nu_min = lam * np.sqrt(1 - A * A) / A          # potential minima appear above this
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
for r in rows:
    ax.plot([r["nu"]] * len(r["x"]), r["x"], ".", ms=1.6, color="#1f4e79")
ax.axvline(nu_min, color="#888888", lw=1, ls="--")
ax.text(nu_min + 0.03, ax.get_ylim()[1] * 0.93, "potential minima appear;\nfield is trapped (de Sitter)",
        fontsize=8, color="#444444", va="top")
ax.set_xlabel(r"modulation frequency $\nu$")
ax.set_ylabel(r"$x$ on the section $\theta = 0$ (mod $2\pi$)")
ax.set_title(r"Period doubling to chaos, $\lambda=10$, $A=0.7$, $w=0$", fontsize=10, loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.2, lw=0.5)
fig.tight_layout()
fig.savefig("fig_bifurcation.png")
