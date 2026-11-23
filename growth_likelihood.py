"""Growth-rate likelihood from the 'Gold-2017' f sigma8 compilation.

Source: Nesseris, Pantazis & Perivolaropoulos, arXiv:1703.10538 (Phys. Rev. D 96,
023542), Table III and Eqs. 2.1-2.8, fetched 11 Nov 2026. Columns below:
z, f sigma8, error, fiducial Omega_m, survey. Where the table leaves the fiducial
cosmology of a second or third point of the same survey blank, the value of the
first point of that survey is used (my assumption, marked '*').
WiggleZ points carry the covariance of their Eq. 2.8; all others are independent.

Method (their Eqs. 2.1-2.7): linear growth with a smooth dark energy,
    delta'' + (2 + H'/H) delta' - (3/2) Omega_m(N) delta = 0,   ' = d/dN,
f sigma8(z) = sigma8 * (delta'/delta) * delta(z)/delta(0), and each prediction is
multiplied by the Alcock-Paczynski ratio H(z) D_A(z) / [H_fid(z) D_A,fid(z)].
sigma8 is a free amplitude, found analytically (the likelihood is quadratic in it).

Assumptions and limits:
- The scalar field is unclustered. For a canonical field (sound speed 1) this holds
  well inside the horizon; it fails only on scales near the horizon.
- sigma8 is not tied to the CMB amplitude, so this tests the shape of f sigma8(z)
  only. A history with strong early suppression of growth is not penalised here.
- Growth is started at z = 50 on the matter-era growing mode (delta ~ a); a field
  fraction of a few percent before that is ignored.
"""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

# z, fs8, sigma, Omega_m fiducial, label
GOLD = [
    (0.02, 0.428, 0.0465, 0.3, "6dFGS+SnIa"),
    (0.02, 0.398, 0.065, 0.3, "SnIa+IRAS"),
    (0.02, 0.314, 0.048, 0.266, "2MASS"),
    (0.10, 0.370, 0.130, 0.3, "SDSS-veloc"),
    (0.15, 0.490, 0.145, 0.31, "SDSS-MGS"),
    (0.17, 0.510, 0.060, 0.3, "2dFGRS"),
    (0.18, 0.360, 0.090, 0.27, "GAMA"),
    (0.38, 0.440, 0.060, 0.27, "GAMA*"),
    (0.25, 0.3512, 0.0583, 0.25, "SDSS-LRG-200"),
    (0.37, 0.4602, 0.0378, 0.25, "SDSS-LRG-200*"),
    (0.32, 0.384, 0.095, 0.274, "BOSS-LOWZ"),
    (0.59, 0.488, 0.060, 0.307115, "SDSS-CMASS"),
    (0.44, 0.413, 0.080, 0.27, "WiggleZ"),
    (0.60, 0.390, 0.063, 0.27, "WiggleZ*"),
    (0.73, 0.437, 0.072, 0.27, "WiggleZ*"),
    (0.60, 0.550, 0.120, 0.3, "Vipers PDR-2"),
    (0.86, 0.400, 0.110, 0.3, "Vipers PDR-2*"),
    (1.40, 0.482, 0.116, 0.270, "FastSound"),
]
Z = np.array([g[0] for g in GOLD])
FS8 = np.array([g[1] for g in GOLD])
OMF = np.array([g[3] for g in GOLD])
COV = np.diag(np.array([g[2] for g in GOLD]) ** 2)
COV[12:15, 12:15] = 1e-3 * np.array([[6.400, 2.570, 0.000], [2.570, 3.969, 2.540], [0.000, 2.540, 5.184]])
ICOV = np.linalg.inv(COV)
ZG = np.linspace(0.0, 1.5, 600)


def _fid_HD(z, om):
    """E(z) * integral dz/E for flat LambdaCDM with matter density om."""
    E = np.sqrt(om * (1 + ZG) ** 3 + 1 - om)
    dc = np.concatenate([[0.0], cumulative_trapezoid(1 / E, ZG)])
    return np.interp(z, ZG, E * dc)


FID = np.array([_fid_HD(z, om) for z, om in zip(Z, OMF)])


def chi2_from_curves(z_grid, E, shape):
    """z_grid increasing from 0; E = H/H0; shape = f * delta(z)/delta(0) on z_grid."""
    dc = np.concatenate([[0.0], cumulative_trapezoid(1 / E, z_grid)])
    ratio = np.interp(Z, z_grid, E * dc) / FID
    t = ratio * np.interp(Z, z_grid, shape)
    s8 = (t @ ICOV @ FS8) / (t @ ICOV @ t)
    r = FS8 - s8 * t
    return float(r @ ICOV @ r), float(s8)


def growth_shape(N, Om, dlnH):
    """Solve the growth equation on an increasing N grid (ending today).

    Om = Omega_m(N), dlnH = H'/H. Returns f * delta/delta(today) on the same grid."""
    def rhs(n, s):
        return [s[1], -(2 + np.interp(n, N, dlnH)) * s[1] + 1.5 * np.interp(n, N, Om) * s[0]]
    sol = solve_ivp(rhs, [N[0], N[-1]], [1.0, 1.0], t_eval=N, rtol=1e-8, atol=1e-10)
    d, dp = sol.y
    return dp / d[-1]                               # f * delta / delta0 = delta' / delta0


def lcdm_chi2(Om0, Or0=0.0):
    N = np.linspace(-np.log(51.0), 0.0, 1500)
    a = np.exp(N)
    E2 = Om0 * a ** -3 + Or0 * a ** -4 + 1 - Om0 - Or0
    Om = Om0 * a ** -3 / E2
    dlnH = -0.5 * (3 * Om0 * a ** -3 + 4 * Or0 * a ** -4) / E2
    shape = growth_shape(N, Om, dlnH)
    z = 1 / a - 1
    return chi2_from_curves(z[::-1], np.sqrt(E2)[::-1], shape[::-1])


if __name__ == "__main__":
    for om in (0.315, 0.28, 0.25):
        c, s8 = lcdm_chi2(om)
        print(f"LCDM Omega_m = {om}: chi2 = {c:.2f} for {len(Z)} points, best sigma8 = {s8:.3f}")
    # check of the growth solver: f(z=0) against the Omega_m^0.55 approximation
    N = np.linspace(-np.log(51.0), 0.0, 1500); a = np.exp(N); om = 0.315
    E2 = om * a ** -3 + 1 - om
    sh = growth_shape(N, om * a ** -3 / E2, -1.5 * om * a ** -3 / E2)
    print(f"f(z=0) from solver = {sh[-1]:.4f}; Omega_m^0.55 = {om ** 0.55:.4f}")
