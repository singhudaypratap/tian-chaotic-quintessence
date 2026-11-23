"""Finite-time largest Lyapunov exponent versus integration length T for the
worst cases of script 02. A true chaotic exponent stays constant; a transient
decays roughly like log(T)/T."""
import json
import numpy as np
from importlib import import_module
m = import_module("02_exponential_numerics")

rows = json.load(open("results_exponential.json"))["amp"]
ok = sorted([r for r in rows if r["lyap"]], key=lambda r: -r["lyap"][0])[:6]
rng = np.random.default_rng(7)
out = []
for r in ok:
    lam, wp1, amp, which = r["lam"], r["w_plus_1"], r["amp"], r["excited"]
    al, Ou, Ov = 0.5 * m.S32 * lam, 1.5 * (wp1 - 2), 1.5 * wp1
    ph = rng.uniform(0, 2 * np.pi, 2)
    small = 1e-3 * amp
    au = amp if which in ("u", "both") else small
    av = amp if which in ("v", "both") else small
    s0 = np.array([au*np.cos(ph[0]), au*np.sin(ph[0]), av*np.cos(ph[1]), av*np.sin(ph[1])])
    line = []
    for T in (100.0, 300.0, 1000.0, 3000.0):
        spec, _ = m.lyap_spectrum(s0, Ou, Ov, al, T=T)
        line.append(float(spec[0]))
    out.append(dict(lam=lam, w_plus_1=wp1, amp=amp, excited=which, T=[100,300,1000,3000], lam1=line))
    print(f"lam={lam} w+1={wp1:g} amp={amp:g} {which:5s} lambda_1(T=100,300,1000,3000) =",
          "  ".join(f"{v: .2e}" for v in line))
json.dump(out, open("results_lyap_convergence.json", "w"), indent=1)
