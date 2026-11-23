"""Does the background fluid decide whether the field is trapped?

Single-fluid system (modsys.py). For each (lam, A) the modulation frequency nu
is raised in small steps; at each step 6 random initial conditions are evolved
for 600 e-folds. Reported: the fraction trapped in a de Sitter minimum, for a
radiation background (w = 1/3) and a matter background (w = 0).
rho = A nu / (lam sqrt(1 - A^2)); potential minima exist for rho > 1.
"""
import json
import sys
import numpy as np
import modsys as m

rng = np.random.default_rng(42)
rhos = np.round(np.arange(0.96, 1.4001, 0.02), 3)
out = []
for lam, A in [(10.0, 0.7), (30.0, 0.7), (30.0, 0.9)]:
    for w in (0.0, 1 / 3):
        row = []
        for rho in rhos:
            nu = rho * lam * np.sqrt(1 - A * A) / A
            ntrap = 0
            for _ in range(6):
                r, th = np.sqrt(rng.uniform(1e-4, 0.5)), rng.uniform(0.1, np.pi - 0.1)
                s0 = np.array([r * np.cos(th), r * np.sin(th), rng.uniform(0, 2 * np.pi)])
                s = m.evolve(s0, 600.0, (lam, A, nu, w), rtol=1e-9, atol=1e-12).y[:, -1]
                ntrap += bool(m.is_trapped(s))
            row.append(ntrap / 6)
        out.append(dict(lam=lam, A=A, w=w, rho=rhos.tolist(), trapped_fraction=row))
        print(f"lam={lam:4.0f} A={A} w={w:.2f}: " + " ".join(f"{v:.2f}" for v in row), flush=True)
print("rho:                    " + " ".join(f"{v:.2f}" for v in rhos))
json.dump(out, open("results_trapping_threshold.json", "w"), indent=1)
