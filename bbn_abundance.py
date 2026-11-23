"""Approximate BBN abundance test for a history with scalar-field energy during BBN.

APPROXIMATE. A full BBN code could not be run here (executing downloaded code was
not permitted in this environment). What is used instead:

  Theory: the table of Y_p and D/H as functions of Omega_b h^2 and Delta N
  (extra relativistic species) distributed with PRyMordial
  (Burns, Tait & Valli, arXiv:2307.07061; file PRyM_Yp_DH_cosmoMC_2023.dat from
  https://github.com/vallima/PRyMordial, copied to data/prym_yp_dh_2023.dat).
  Columns used: ombh2, DeltaN, Yp, D/H, sig(Yp^BBN), sig(D/H).

  Observations (Particle Data Group, Review of Particle Physics 2025, "Big-Bang
  nucleosynthesis", https://pdg.lbl.gov/2025/reviews/rpp2025-rev-bbang-nucleosynthesis.pdf):
    Y_p = 0.245 +- 0.003            (Eq. 24.3)
    D/H = (25.08 +- 0.29) x 10^-6   (Eq. 24.2)
  The same review gives T_fr ~ 1 MeV for neutron-proton freeze-out and T ~ 0.1 MeV
  for the deuterium bottleneck, which set the two windows below.

  Mapping (derived): a smooth extra component with fraction Omega of the total
  density raises the expansion rate exactly as Delta N extra neutrino species with
      Delta N = c * Omega / (1 - Omega),
      c = 10.75 / 1.75 = 6.143                    before e+e- annihilation,
      c = 3.044 + 1 / [(7/8) (4/11)^(4/3)] = 7.447  after it.
  Delta N for helium is the average of this over 0.08 < T < 2 MeV (freeze-out to the
  bottleneck); for deuterium, over 0.03 < T < 0.08 MeV. Averages are over ln T.

What this can support: a threshold-level statement (a history with Omega of several
percent in either window is excluded; one with Omega << 1% is safe). What it cannot:
a quantitative abundance for a history whose field fraction changes strongly within
a window, because the table assumes a constant Delta N throughout BBN. Tian's own
code (Part IV) integrates the network with the field included and is more accurate
for such cases; expect differences of tens of percent in the inferred Delta N.
Photon temperature: T = T0 (1+z) after annihilation, T0 (1+z) (4/11)^(1/3) before,
switched at T = 0.2 MeV (the true transition is gradual).
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
YP_OBS, YP_ERR = 0.245, 0.003
DH_OBS, DH_ERR = 25.08e-6, 0.29e-6
K_B = 8.617333262e-11                    # MeV per kelvin (exact SI definition)
T0_MEV = 2.7255 * K_B
C_BEFORE = 10.75 / 1.75
C_AFTER = 3.044 + 1.0 / (0.875 * (4.0 / 11.0) ** (4.0 / 3.0))

_t = np.loadtxt(os.path.join(HERE, "data", "prym_yp_dh_2023.dat"))
_wb, _dn = np.unique(_t[:, 0]), np.unique(_t[:, 2])
_order = np.lexsort((_t[:, 0], _t[:, 2]))            # sorted by DeltaN, then ombh2
_grid = lambda col: _t[_order, col].reshape(len(_dn), len(_wb))
_YP = RegularGridInterpolator((_dn, _wb), _grid(3))
_DH = RegularGridInterpolator((_dn, _wb), _grid(6))
_SY = RegularGridInterpolator((_dn, _wb), _grid(5))
_SD = RegularGridInterpolator((_dn, _wb), _grid(7))
DN_MAX, DN_MIN = _dn[-1], _dn[0]


def temperature_MeV(z):
    T = T0_MEV * (1 + z)
    return np.where(T > 0.2, T * (4.0 / 11.0) ** (1.0 / 3.0), T)


def delta_N(z, Ophi):
    """Effective Delta N for helium and for deuterium from Omega_phi(z) (arrays)."""
    T = temperature_MeV(np.asarray(z, float))
    O = np.clip(np.asarray(Ophi, float), 0, 0.999999)
    dn = np.where(T > 0.5, C_BEFORE, C_AFTER) * O / (1 - O)
    lnT = np.log(T)
    def avg(lo, hi):
        s = (T > lo) & (T < hi)
        if s.sum() < 2:
            return float("nan")
        o = np.argsort(lnT[s])
        x, y = lnT[s][o], dn[s][o]
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)) / (x[-1] - x[0]))
    return avg(0.08, 2.0), avg(0.03, 0.08)


def chi2(wb, dN_he=0.0, dN_d=0.0):
    """chi2 for Y_p and D/H; returns (chi2, Y_p, D/H, flag_out_of_table)."""
    out = bool(dN_he > DN_MAX or dN_d > DN_MAX or not np.isfinite(dN_he + dN_d))
    a = np.clip(np.nan_to_num(dN_he, nan=DN_MAX), DN_MIN, DN_MAX)
    b = np.clip(np.nan_to_num(dN_d, nan=DN_MAX), DN_MIN, DN_MAX)
    wb = float(np.clip(wb, _wb[0], _wb[-1]))
    yp, sy = float(_YP([a, wb])[0]), float(_SY([a, wb])[0])
    dh, sd = float(_DH([b, wb])[0]), float(_SD([b, wb])[0])
    c = (yp - YP_OBS) ** 2 / (YP_ERR ** 2 + sy ** 2) + (dh - DH_OBS) ** 2 / (DH_ERR ** 2 + sd ** 2)
    if out:                               # beyond Delta N = 7: far outside any allowed range
        c += 1e4
    return float(c), yp, dh, out


if __name__ == "__main__":
    for wb in (0.02236, 0.0224, 0.02205):
        c, yp, dh, _ = chi2(wb)
        print(f"standard BBN, omega_b = {wb}: Y_p = {yp:.4f}, D/H = {dh*1e6:.2f}e-6, chi2 = {c:.2f} (2 data points)")
    print(f"PDG: Y_p = {YP_OBS} +- {YP_ERR}, D/H = {DH_OBS*1e6} +- {DH_ERR*1e6} e-6")
    print(f"dY_p/dDeltaN at Delta N = 0 from the table: {(chi2(0.02236, 1.0)[1] - chi2(0.02236, 0.0)[1]):.4f} (PDG: about 0.013)")
    for O in (0.01, 0.04, 0.1):
        z = np.geomspace(1e7, 1e11, 400)
        a, b = delta_N(z, np.full_like(z, O))
        c, yp, dh, _ = chi2(0.02236, a, b)
        print(f"constant Omega_phi = {O}: Delta N (He, D) = {a:.2f}, {b:.2f}; Y_p = {yp:.4f}, D/H = {dh*1e6:.2f}e-6, chi2 = {c:.1f}")
    print("T(z): z = 4.3e9 ->", np.round(temperature_MeV(np.array([4.3e9, 4.3e8, 1.3e8])), 3), "MeV")
