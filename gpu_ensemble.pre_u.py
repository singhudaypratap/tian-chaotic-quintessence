"""Large ensembles of Tian's oscillating-quintessence model on a GPU.

All histories are advanced together with a fixed-step RK4 in float64.
Each history starts from its own random state, is run for --burn e-folds in a
pure radiation background so that it sits on the radiation-era attractor, and
is then carried through matter-radiation equality to today. Because every
history has an independent start, the samples are independent (unlike an
ensemble that varies only the start time).

Tests per history (same as 11_tian_parameter_scan.py):
  BBN    : peak Omega_phi < 0.04 for 3.7e8 < z < 1.03e10
  recomb : peak Omega_phi < 0.02 for 50 < z < 5000
  today  : 0.6 < Omega_phi < 0.8 and w_phi < -0.8
Histories passing the first two with 0.45 < Omega_phi(today) < 0.9 are saved, as
their state at z = 50, for fit_survivors.py.

Examples
  python3 gpu_ensemble.py --l1 10 --alpha 0.24 --n 2000000
  python3 gpu_ensemble.py --grid --n 500000
  python3 gpu_ensemble.py --l1 10 --alpha 0.24 --n 4000 --device cpu   (quick test)

Runs with PyTorch if installed (CUDA if available), otherwise with NumPy.
"""
import argparse
import json
import math
import time

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:                       # NumPy fallback, CPU only
    HAVE_TORCH = False

S32, S6 = math.sqrt(1.5), math.sqrt(6.0)
LN_1PZEQ = math.log(1.0 + 3400.0)
N_EQ = 35.0                               # e-folds of the tracked history before equality
Z_BBN = (3.7e8, 1.03e10)
Z_CMB = (50.0, 5000.0)
Z_STATE = 1.0e9          # candidates also store their state here, the starting point for class_field.py
# Between the two windows (5000 < z < 3.7e8) the field is not tested, but large episodes
# there were found to be common; their peak is recorded as gap_max (reported, not cut on).
# Early-time table for the CMB and BBN modules: bin averages of sqrt(1 - Omega_phi)
# in N_TAB bins of N between z = 1e11 and z = 50 (about 0.25 e-folds per bin). The
# average, not a point sample, is stored so that narrow spikes of the field fraction
# are not missed; sqrt(1 - Omega_phi) is the factor by which the field changes 1/H.
# W (added for cmb_spectrum.py and bbn_network.py): the bin average of the field's equation
# of state w = (x^2 - y^2)/(x^2 + y^2). The field density then follows exactly from
# d ln rho / dN = -3 (1 + w), which is far better conditioned than differencing 1 - S^2
# where the field fraction is tiny.
# The table comes from the GPU run itself because recomputing 20 e-folds of a chaotic
# history (Lyapunov exponent ~0.2 per e-fold) from a stored state would not reproduce it.
Z_TAB_MAX = 1.0e11
N_TAB = 86


class Backend:
    def __init__(self, device):
        self.torch = HAVE_TORCH and device != "numpy"
        if self.torch:
            self.dev = torch.device(device)
            torch.set_grad_enabled(False)

    def array(self, a):
        return torch.as_tensor(a, dtype=torch.float64, device=self.dev) if self.torch else np.asarray(a, float)

    def cos(self, a):
        return torch.cos(a) if self.torch else np.cos(a)

    def maximum(self, a, b):
        return torch.maximum(a, b) if self.torch else np.maximum(a, b)

    def project(self, x, y):
        """Keep (x, y) in the physical region x^2 + y^2 <= 1, y >= 0.

        The exact flow preserves this region (its boundary is invariant), but a fixed-step
        integrator can step just outside when the field nearly dominates, and outside it
        the equations run away (found in testing: up to all histories lost at alpha = 0.8)."""
        r2 = x * x + y * y
        if self.torch:
            f = torch.where(r2 > 1.0, torch.sqrt((1.0 - 1e-12) / r2), torch.ones_like(r2))
            return x * f, torch.abs(y * f)
        f = np.where(r2 > 1.0, np.sqrt((1.0 - 1e-12) / np.maximum(r2, 1e-300)), 1.0)
        return x * f, np.abs(y * f)

    def sqrt_one_minus(self, O):
        """sqrt(max(1 - O, 0))"""
        return torch.sqrt(torch.clamp(1 - O, min=0.0)) if self.torch else np.sqrt(np.clip(1 - O, 0.0, None))

    def numpy(self, a):
        return a.detach().cpu().numpy() if self.torch else a


def rhs(B, x, y, th, w, lbar, dl, al):
    le = lbar + dl * B.cos(th)
    brk = 2 * x * x + (1 + w) * (1 - x * x - y * y)
    return (-3 * x + S32 * le * y * y + 1.5 * x * brk,
            -S32 * le * x * y + 1.5 * y * brk,
            (S6 / al) * x)


def rk4(B, x, y, th, w0, w1, w2, h, lbar, dl, al):
    """One RK4 step; w0, w1, w2 are the background w at N, N + h/2, N + h."""
    a1 = rhs(B, x, y, th, w0, lbar, dl, al)
    a2 = rhs(B, x + 0.5 * h * a1[0], y + 0.5 * h * a1[1], th + 0.5 * h * a1[2], w1, lbar, dl, al)
    a3 = rhs(B, x + 0.5 * h * a2[0], y + 0.5 * h * a2[1], th + 0.5 * h * a2[2], w1, lbar, dl, al)
    a4 = rhs(B, x + h * a3[0], y + h * a3[1], th + h * a3[2], w2, lbar, dl, al)
    xn, yn = B.project(x + h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
                       y + h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]))
    return xn, yn, th + h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])


def w_bg(N):
    """Radiation + matter as one fluid; N counted from the start of the tracked history."""
    e = N - N_EQ
    return (1.0 / 3.0) / (1.0 + math.exp(min(max(e, -700.0), 700.0)))


def run(l1, l2, al, n, dN, burn, device, seed, keep_states=False):
    B = Backend(device)
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0.01, 0.9, n))
    ang = rng.uniform(0.1, math.pi - 0.1, n)
    x, y = B.array(r * np.cos(ang)), B.array(r * np.sin(ang))
    th = B.array(rng.uniform(0, 2 * math.pi, n))
    lbar, dl = 0.5 * (l1 + l2), 0.5 * (l1 - l2)

    for _ in range(int(round(burn / dN))):            # radiation-era attractor
        x, y, th = rk4(B, x, y, th, 1 / 3, 1 / 3, 1 / 3, dN, lbar, dl, al)
    start = (B.numpy(x).copy(), B.numpy(y).copy(), B.numpy(th).copy()) if keep_states else None

    N_end = N_EQ + LN_1PZEQ
    steps = int(math.ceil(N_end / dN))
    h = N_end / steps
    bbn_max, cmb_max, gap_max = 0 * x, 0 * x, 0 * x
    O_rec, best = 0 * x, float("inf")
    snap = None                                       # state when z first drops to 50
    snap_early = None                                 # state when z first drops to Z_STATE (for CLASS)
    edges = np.linspace(N_end - math.log(1 + Z_TAB_MAX), N_end - math.log(1 + Z_CMB[0]), N_TAB + 1)
    width = edges[1] - edges[0]
    tables = [None] * N_TAB
    tables_w = [None] * N_TAB                         # bin averages of w_phi
    cur, acc, accw, cnt = -1, None, None, 0
    for i in range(steps):
        N = i * h
        x, y, th = rk4(B, x, y, th, w_bg(N), w_bg(N + 0.5 * h), w_bg(N + h), h, lbar, dl, al)
        z = math.exp(N_end - (N + h)) - 1.0
        if snap_early is None and z <= Z_STATE:
            snap_early = (B.numpy(x).copy(), B.numpy(y).copy(), B.numpy(th).copy(), N + h)
        k = int(math.floor((N + h - edges[0]) / width))
        if k != cur:
            if 0 <= cur < N_TAB and cnt:
                tables[cur] = B.numpy(acc / cnt).astype(np.float32)
                tables_w[cur] = B.numpy(accw / cnt).astype(np.float32)
            cur, acc, accw, cnt = k, None, None, 0
        if 0 <= k < N_TAB:
            S = B.sqrt_one_minus(x * x + y * y)
            acc = S if acc is None else acc + S
            wi = (x * x - y * y) / (x * x + y * y)
            accw = wi if accw is None else accw + wi
            cnt += 1
        if Z_BBN[0] < z < Z_BBN[1]:
            bbn_max = B.maximum(bbn_max, x * x + y * y)
        elif Z_CMB[1] <= z <= Z_BBN[0]:
            gap_max = B.maximum(gap_max, x * x + y * y)
        elif Z_CMB[0] < z < Z_CMB[1]:
            cmb_max = B.maximum(cmb_max, x * x + y * y)
            if abs(z - 1100.0) < best:
                best, O_rec = abs(z - 1100.0), x * x + y * y
        elif snap is None and z <= Z_CMB[0]:
            snap = (B.numpy(x).copy(), B.numpy(y).copy(), B.numpy(th).copy(), N + h)
    if 0 <= cur < N_TAB and cnt:
        tables[cur] = B.numpy(acc / cnt).astype(np.float32)
        tables_w[cur] = B.numpy(accw / cnt).astype(np.float32)
    O0 = B.numpy(x * x + y * y)
    w0 = B.numpy((x * x - y * y) / (x * x + y * y))
    out = dict(Ophi0=O0, w0=w0, bbn_max=B.numpy(bbn_max), cmb_max=B.numpy(cmb_max),
               gap_max=B.numpy(gap_max), Ophi_rec=B.numpy(O_rec))
    out["_snap_early"] = snap_early
    return out, start, snap, (edges, tables, tables_w)


def lyapunov(l1, l2, al, w, device, seed, m=2000, T=100.0, dN=4e-3, burn=150.0, renorm=1.0):
    """Finite-time (T e-folds) largest Lyapunov exponent per e-fold, constant w.

    Two-trajectory method on m independent histories: a copy displaced by 1e-8 is
    evolved alongside and rescaled every `renorm` e-folds. Returns the mean over
    histories and its standard error."""
    B = Backend(device)
    rng = np.random.default_rng(seed + 7)
    r = np.sqrt(rng.uniform(0.01, 0.9, m))
    ang = rng.uniform(0.1, math.pi - 0.1, m)
    x, y = B.array(r * np.cos(ang)), B.array(r * np.sin(ang))
    th = B.array(rng.uniform(0, 2 * math.pi, m))
    lbar, dl = 0.5 * (l1 + l2), 0.5 * (l1 - l2)
    for _ in range(int(round(burn / dN))):
        x, y, th = rk4(B, x, y, th, w, w, w, dN, lbar, dl, al)
    d0 = 1e-8
    u = rng.normal(size=(3, m))
    u *= d0 / np.linalg.norm(u, axis=0)
    xp, yp, tp = x + B.array(u[0]), y + B.array(u[1]), th + B.array(u[2])
    acc = np.zeros(m)
    per = int(round(renorm / dN))
    for k in range(int(round(T / renorm))):
        for _ in range(per):
            x, y, th = rk4(B, x, y, th, w, w, w, dN, lbar, dl, al)
            xp, yp, tp = rk4(B, xp, yp, tp, w, w, w, dN, lbar, dl, al)
        dx, dy, dt = xp - x, yp - y, tp - th
        d = B.numpy((dx * dx + dy * dy + dt * dt) ** 0.5)
        d = np.where(np.isfinite(d) & (d > 0), d, d0)
        acc += np.log(d / d0)
        f = B.array(d0 / d)
        xp, yp, tp = x + dx * f, y + dy * f, th + dt * f
    lam = acc / T
    lam = lam[np.isfinite(lam)]
    return dict(mean=float(lam.mean()), stderr=float(lam.std(ddof=1) / math.sqrt(len(lam))),
                frac_positive=float(np.mean(lam > 0.02)), n=int(len(lam)), T=T)


def block_se(sel, good, blocks=20):
    """Standard error of a rate from the scatter of `blocks` equal sub-samples.
    If this is much larger than the binomial error, the histories are correlated."""
    idx = np.where(good)[0]
    parts = np.array_split(idx, blocks)
    rates = np.array([sel[p].mean() for p in parts if len(p)])
    return float(rates.std(ddof=1) / math.sqrt(len(rates)))


def wilson(k, n, zc=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + zc * zc / n
    c = (p + zc * zc / (2 * n)) / d
    hw = zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / d
    return [max(0.0, c - hw), min(1.0, c + hw)]


def summarise(o):
    good = np.isfinite(o["Ophi0"]) & np.isfinite(o["bbn_max"]) & np.isfinite(o["cmb_max"]) \
        & (o["Ophi0"] <= 1.0 + 1e-6)
    n = int(good.sum())
    bbn = good & (o["bbn_max"] < 0.04)
    cmb = good & (o["cmb_max"] < 0.02)
    today = good & (o["Ophi0"] > 0.6) & (o["Ophi0"] < 0.8) & (o["w0"] < -0.8)
    strict = good & (o["Ophi0"] > 0.66) & (o["Ophi0"] < 0.72) & (o["w0"] < -0.9)
    res = dict(n_valid=n, n_invalid=int((~good).sum()))
    for name, sel in (("bbn", bbn), ("recomb", cmb), ("today", today), ("today_strict", strict),
                      ("bbn_and_recomb", bbn & cmb), ("recomb_and_today", cmb & today),
                      ("bbn_and_today", bbn & today), ("all_three", bbn & cmb & today),
                      ("all_three_strict", bbn & cmb & strict)):
        k = int(sel.sum())
        p = k / n if n else float("nan")
        res[name] = dict(count=k, rate=p, ci95=wilson(k, n),
                         binomial_se=math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan"),
                         block_se=block_se(sel, good))
    prod = res["bbn"]["rate"] * res["recomb"]["rate"] * res["today"]["rate"]
    res["product_of_marginals"] = prod
    all3 = bbn & cmb & today
    res["gap_peak_among_all_three"] = dict(
        frac_below_0p01=float(np.mean(o["gap_max"][all3] < 0.01)) if all3.any() else None,
        frac_below_0p05=float(np.mean(o["gap_max"][all3] < 0.05)) if all3.any() else None,
        median=float(np.median(o["gap_max"][all3])) if all3.any() else None)
    # Independence of the three epochs: observed joint count against n * product.
    expect = n * prod
    k = res["all_three"]["count"]
    res["independence"] = dict(expected_if_independent=expect,
                               ratio_observed_to_expected=(k / expect if expect > 0 else float("nan")),
                               ratio_ci95=[c * n / expect if expect > 0 else float("nan")
                                           for c in res["all_three"]["ci95"]])
    return res, (bbn & cmb & today)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", type=float, default=10.0)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.24)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dN", type=float, default=2e-3)
    ap.add_argument("--burn", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid", action="store_true", help="scan l1 in 6..30 and alpha in 0.08..0.8")
    ap.add_argument("--no-lyap", action="store_true", help="skip the Lyapunov estimates")
    ap.add_argument("--check-step", action="store_true",
                    help="repeat with half the step on up to 200000 histories and report the change in rates")
    ap.add_argument("--out", default="gpu_ensemble_results.jsonl")
    a = ap.parse_args()
    if not HAVE_TORCH:
        a.device = "numpy"
    cells = ([(l1, al) for l1 in (6, 8, 10, 12, 14, 17, 20, 25, 30)
              for al in (0.08, 0.15, 0.24, 0.4, 0.6, 0.8)] if a.grid else [(a.l1, a.alpha)])
    print(f"device {a.device}, {a.n} histories per cell, step {a.dN}, {len(cells)} cell(s)", flush=True)
    for l1, al in cells:
        t0 = time.time()
        o, start, snap, (edges, tables, tables_w) = run(float(l1), a.l2, float(al), a.n, a.dN, a.burn, a.device, a.seed, keep_states=True)
        res, ok = summarise(o)
        rec = dict(l1=float(l1), l2=a.l2, alpha=float(al), n=a.n, dN=a.dN, burn=a.burn, seed=a.seed, **res)
        if not a.no_lyap:
            rec["lyapunov"] = dict(radiation=lyapunov(float(l1), a.l2, float(al), 1 / 3, a.device, a.seed),
                                   matter=lyapunov(float(l1), a.l2, float(al), 0.0, a.device, a.seed))
        if a.check_step:
            nc = min(a.n, 200000)
            rates = {}
            for label, step in (("step", a.dN), ("half_step", a.dN / 2)):
                oc, _, _, _ = run(float(l1), a.l2, float(al), nc, step, a.burn, a.device, a.seed + 1)
                rc, _ = summarise(oc)
                rates[label] = {k: rc[k]["rate"] for k in ("bbn", "recomb", "today", "all_three")}
                rates[label]["n_invalid"] = rc["n_invalid"]
                rates[label + "_se"] = {k: rc[k]["binomial_se"] for k in ("bbn", "recomb", "today", "all_three")}
            rec["step_check"] = dict(n=nc, **rates)
        rec["seconds"] = time.time() - t0
        # Candidates for the data fit: pass BBN and recombination, dark energy roughly right
        # today. The state at z = 50 is stored, so the late history (3.9 e-folds, well inside
        # the predictability horizon) can be recomputed exactly by fit_survivors.py.
        cand = (o["bbn_max"] < 0.04) & (o["cmb_max"] < 0.02) & (o["Ophi0"] > 0.45) & (o["Ophi0"] < 0.9)
        idx = np.where(cand)[0][:5000]
        rec["N_snapshot"], rec["N_eq"], rec["N_end"] = snap[3], N_EQ, N_EQ + LN_1PZEQ
        se = o.pop("_snap_early")
        rec["N_state_early"] = se[3]
        rec["n_candidates"] = int(cand.sum())
        rec["S_edges"] = [float(e) for e in edges]
        rec["candidates"] = [dict(x=float(snap[0][i]), y=float(snap[1][i]), theta=float(snap[2][i]),
                                  Ophi0=float(o["Ophi0"][i]), w0=float(o["w0"][i]),
                                  S=[round(float(t[i]), 5) for t in tables],
                                  W=[round(float(t[i]), 5) for t in tables_w],
                                  early=[float(se[0][i]), float(se[1][i]), float(se[2][i])]) for i in idx]
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        r = res
        print(f"l1={l1} alpha={al}: valid {r['n_valid']} | BBN {100*r['bbn']['rate']:.2f}%  "
              f"recomb {100*r['recomb']['rate']:.2f}%  today {100*r['today']['rate']:.2f}% | "
              f"all three {r['all_three']['count']} (rate {r['all_three']['rate']:.2e}, "
              f"95% CI {r['all_three']['ci95'][0]:.1e}-{r['all_three']['ci95'][1]:.1e}) | "
              f"product {r['product_of_marginals']:.1e} | {rec['seconds']:.0f} s", flush=True)
        print(f"    standard errors (binomial / 20-block): BBN {r['bbn']['binomial_se']:.1e} / {r['bbn']['block_se']:.1e}, "
              f"recomb {r['recomb']['binomial_se']:.1e} / {r['recomb']['block_se']:.1e}, "
              f"today {r['today']['binomial_se']:.1e} / {r['today']['block_se']:.1e}; diverged {r['n_invalid']}; "
              f"joint / independent expectation = {r['independence']['ratio_observed_to_expected']:.2f}", flush=True)
        gp = r["gap_peak_among_all_three"]
        if gp["median"] is not None:
            print(f"    field fraction for 5000<z<3.7e8 among histories passing all three: median peak {gp['median']:.3f}, "
                  f"below 0.01: {100*gp['frac_below_0p01']:.0f}%, below 0.05: {100*gp['frac_below_0p05']:.0f}%", flush=True)
        if "lyapunov" in rec:
            L = rec["lyapunov"]
            print(f"    finite-time Lyapunov exponent per e-fold (100 e-folds): radiation {L['radiation']['mean']:.3f} +- {L['radiation']['stderr']:.3f}, "
                  f"matter {L['matter']['mean']:.3f} +- {L['matter']['stderr']:.3f}", flush=True)
        if "step_check" in rec:
            c = rec["step_check"]
            print("    step check (n=%d): " % c["n"] + ", ".join(
                f"{k} {100*c['step'][k]:.2f}% -> {100*c['half_step'][k]:.2f}%" for k in ("bbn", "recomb", "today")), flush=True)


if __name__ == "__main__":
    main()
