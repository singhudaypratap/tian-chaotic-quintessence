"""Compressed CMB likelihood: Planck 2018 distance priors (R, l_A, omega_b).

Source of every number and formula here:
  Chen, Huang & Wang, "Distance priors from Planck final release",
  arXiv:1808.05724 (JCAP 02 (2019) 028), fetched 11 Nov 2026.
  - Table I, base LambdaCDM, Planck TT,TE,EE+lowE:
      R = 1.7502 +- 0.0046, l_A = 301.471 +0.089/-0.090, Omega_b h^2 = 0.02236 +- 0.00015
      correlations: R-l_A 0.46, R-omega_b -0.66, l_A-omega_b -0.33
    (n_s is dropped; the 3x3 block is used.)
  - Eqs. 1-2:  l_A = pi D_M(z*)/r_s(z*),  R = sqrt(Omega_m H0^2) D_M(z*)/c
  - Eqs. 3-6:  r_s(z) = (c/H0) int_0^a da / (a^2 E sqrt(3 (1 + Rb a))),
               Rb = 31500 Omega_b h^2 (T_cmb/2.7 K)^-4,  T_cmb = 2.7255 K,
               z_eq = 2.5e4 Omega_m h^2 (T_cmb/2.7 K)^-4,  Omega_r = Omega_m/(1+z_eq)
  - Eqs. 8-10: z* fitting formula of Hu & Sugiyama. The paper's text prints the g1
    prefactor as 0.0738; its appendix code uses 0.0783, which is the Hu-Sugiyama
    value and is used here.

Validity: distance priors summarise the CMB only when physics before and at
recombination is standard. They are a necessary condition for histories whose
scalar-field fraction is negligible for z >~ 50; they are NOT a substitute for a
Boltzmann calculation when that fraction is at the percent level near z*, because
the peak heights (early ISW, damping) then change as well. Use them to reject, and
to rank LambdaCDM-like survivors, not to claim a detection.

Not built: CAMB + Planck-lite. CAMB 2.0.5 installs here and DarkEnergyPPF accepts a
w(a) table, but (i) PPF with an arbitrary w(a) is an effective fluid and does not
reproduce a canonical field's perturbations when the field is non-negligible early,
and (ii) a profiled 6-parameter spectrum fit costs minutes per history. See report.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

T_CMB = 2.7255
T27 = (T_CMB / 2.7) ** -4
MEAN = np.array([1.7502, 301.471, 0.02236])
SIG = np.array([0.0046, 0.090, 0.00015])
CORR = np.array([[1.0, 0.46, -0.66], [0.46, 1.0, -0.33], [-0.66, -0.33, 1.0]])
ICOV = np.linalg.inv(CORR * np.outer(SIG, SIG))


def z_star(wb, wm):
    g1 = 0.0783 * wb ** -0.238 / (1 + 39.5 * wb ** 0.763)
    g2 = 0.560 / (1 + 21.1 * wb ** 1.81)
    return 1048 * (1 + 0.00124 * wb ** -0.738) * (1 + g1 * wm ** g2)


# Note on the radiation density. This module keeps the z_eq = 2.5e4 omega_m (T/2.7 K)^-4
# relation of Chen, Huang & Wang (implied omega_r = 4.153e-5) rather than CAMB's 4.183e-5
# (photons + 3.044 massless neutrinos). Tested: with CAMB's value the formulas below give
# 100 theta* = 1.0392 for LambdaCDM at the Planck values, against 1.0410 with the CHW relation;
# the compressed priors are calibrated with their own relation. cmb_spectrum.py does not take
# the shift of today from this fit (it solves it from theta* with CAMB), so the 0.7% difference
# in omega_m between the two stages affects only the ranking of candidates, not the CMB result.
def z_eq_of(wm):
    return 2.5e4 * wm * T27


def chi2(R, lA, wb):
    d = np.array([R, lA, wb]) - MEAN
    return float(d @ ICOV @ d)


def _trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def observables(u, inv_E, Om0, wm, wb):
    """R and l_A from 1/E tabulated on u = ln(1+z) (increasing, from 0 to >= ln(1e9)).

    inv_E is H0/H. The tail above the last point is added assuming radiation
    domination with the same 1/E scaling (1/E ~ (1+z)^-2)."""
    zs = z_star(wb, wm)
    us = np.log1p(zs)
    opz = np.exp(u)
    # D_M: integral of dz/E = (1+z) du / E from 0 to z*
    lo = u <= us
    uu = np.append(u[lo], us)
    f = np.append((opz * inv_E)[lo], np.exp(us) * np.interp(us, u, inv_E))
    dm = _trap(f, uu)
    # r_s: integral over z > z* of dz / (E sqrt(3 (1 + Rb/(1+z))))
    cs = 1.0 / np.sqrt(3.0 * (1.0 + 31500.0 * wb * T27 / opz))
    g = opz * inv_E * cs
    hi = u >= us
    uu = np.insert(u[hi], 0, us)
    gg = np.insert(g[hi], 0, np.interp(us, u, g))
    rs = _trap(gg, uu) + g[-1]                      # tail: g ~ exp(-u), integral = g_last
    return np.sqrt(Om0) * dm, np.pi * dm / rs, zs


# ---------------- LambdaCDM reference ----------------
U_L = np.linspace(0.0, np.log(1e9), 6000)


def lcdm_observables(Om, h, wb):
    wm = Om * h * h
    Or = Om / (1 + z_eq_of(wm))
    opz = np.exp(U_L)
    E = np.sqrt(Or * opz ** 4 + Om * opz ** 3 + 1 - Om - Or)
    return observables(U_L, 1.0 / E, Om, wm, wb)


def lcdm_chi2(Om, h, wb):
    R, lA, _ = lcdm_observables(Om, h, wb)
    return chi2(R, lA, wb)


# ---------------- a model history ----------------
def history_inv_E(N0, N_eq, Om0, late_N, late_Ophi, tab_edges, tab_S, n_sub=12):
    """1/E on u = ln(1+z) for a history observed at N0.

    late_N, late_Ophi : the recomputed late history (N increasing, ends at or after N0)
    tab_edges, tab_S  : bin edges in N (increasing) and bin-averaged sqrt(1 - Omega_phi)
                        from the ensemble run, covering earlier times."""
    def fluid(N):                                   # Omega_m0 (1+z)^3 (1 + rho_r/rho_m)
        return Om0 * np.exp(3 * (N0 - N)) * (1 + np.exp(N_eq - N))
    # late part, fine grid from N0 back to the first late point
    Nl = np.linspace(N0, max(late_N[0], tab_edges[-1]), 600)
    Sl = np.sqrt(np.clip(1 - np.interp(Nl, late_N, late_Ophi), 0, 1))
    u_l, iE_l = N0 - Nl, Sl / np.sqrt(fluid(Nl))
    # tabulated part: sub-sample each bin, constant S within the bin
    k = np.arange(len(tab_S))[::-1]                 # walk back in time
    Ne = np.concatenate([np.linspace(tab_edges[i + 1], tab_edges[i], n_sub, endpoint=False) for i in k])
    Se = np.repeat(np.asarray(tab_S)[k], n_sub)
    keep = Ne < Nl[-1]
    Ne, Se = Ne[keep], Se[keep]
    u = np.concatenate([u_l, N0 - Ne])
    iE = np.concatenate([iE_l, Se / np.sqrt(fluid(Ne))])
    return u, iE


def model_chi2(N0, N_eq, Om0, late_N, late_Ophi, tab_edges, tab_S, wb):
    wm = (1 + np.exp(N0 - N_eq)) / (2.5e4 * T27)    # 1 + z_eq = exp(N0 - N_eq); CHW z_eq relation
    u, iE = history_inv_E(N0, N_eq, Om0, late_N, late_Ophi, tab_edges, tab_S)
    R, lA, zs = observables(u, iE, Om0, wm, wb)
    return chi2(R, lA, wb), dict(R=R, lA=lA, wm=wm, h=float(np.sqrt(wm / Om0)), z_star=zs)


if __name__ == "__main__":
    # Validation on flat LambdaCDM with the Planck 2018 abstract values (arXiv:1807.06209):
    # Omega_m = 0.315, H0 = 67.4, Omega_b h^2 = 0.0224.
    R, lA, zs = lcdm_observables(0.315, 0.674, 0.0224)
    print(f"LCDM (0.315, 0.674, 0.0224): z* = {zs:.2f}, R = {R:.4f}, l_A = {lA:.3f}, chi2 = {chi2(R, lA, 0.0224):.2f}")
    print(f"100 theta* = 100 pi / l_A = {100 * np.pi / lA:.4f}  (Planck abstract: 1.0411 +- 0.0003)")
    best = minimize(lambda p: lcdm_chi2(*p), [0.315, 0.674, 0.0224], method="Nelder-Mead",
                    options=dict(xatol=1e-6, fatol=1e-6))
    print("best LCDM fit to the priors alone:", np.round(best.x, 5), "chi2", round(best.fun, 4))
