"""Planck 2018 plik_lite high-l TT+TE+EE likelihood: own loader and chi2.

Data files (in data/planck2018_plik_lite/): cl_cmb_plik_v22.dat, c_matrix_plik_v22.dat,
blmin.dat, blmax.dat, bweight.dat. They are the Planck Legacy Archive plik_lite v22
files as redistributed in https://github.com/heatherprince/planck-lite-py
(Prince & Dunkley, arXiv:1909.05869; Planck 2018 likelihood paper arXiv:1907.12875).

The file formats, bin counts (215 TT bins for 30 <= l <= 2508, 199 TE and 199 EE bins
for 30 <= l <= 1996), the binning rule and the treatment of the covariance file were
learned by reading planck_lite_py.py in that repository; this module is an
independent implementation and does not import it:
  - spectra enter as C_l (not D_l), in muK^2, binned with the weights in bweight.dat
    over l = blmin + 30 ... blmax + 30;
  - the covariance file is Fortran unformatted, one (613 x 613) real*8 record whose
    upper triangle must be filled from the lower one.

Check (run this file): with the reference spectrum data/Dl_planck2015fit.dat from the
same repository the log-likelihood must equal -291.33481235418026, the value that
repository quotes for 2018 TT+TE+EE (and for cobaya's native plik_lite).

Low-l TT (optional, low_ell=True): the two approximately Gaussian bins for 2 <= l <= 29 of
Prince & Dunkley (arXiv:1909.05869), files in data/planck2018_low_ell/ from the same
repository (CTT_bin_low_ell_2018.dat: bin centre, C_l, sigma; blmin/blmax/bweight_low_ell.dat;
first multipole 2). As in that code they are independent of the high-l bins, with diagonal
variance sigma^2. Reference: -293.95586501795134 for 2018 TT+TE+EE + low-l TT bins.
"""
import os

import numpy as np
from scipy.io import FortranFile
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "planck2018_plik_lite")
BASE = "https://raw.githubusercontent.com/heatherprince/planck-lite-py/master/data/"
FILES = ["cl_cmb_plik_v22.dat", "c_matrix_plik_v22.dat", "blmin.dat", "blmax.dat", "bweight.dat"]
NTT, NTE, NEE, LMIN = 215, 199, 199, 30
NBIN = NTT + NTE + NEE
LMAX_NEEDED = 2508


def download():
    import urllib.request
    os.makedirs(DIR, exist_ok=True)
    for f in FILES:
        p = os.path.join(DIR, f)
        if not os.path.exists(p):
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(BASE + "planck2018_plik_lite/" + f, p)
    p = os.path.join(HERE, "data", "Dl_planck2015fit.dat")
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + "Dl_planck2015fit.dat", p)


LOW_DIR = os.path.join(HERE, "data", "planck2018_low_ell")


class PlikLite:
    def __init__(self, low_ell=False):
        download()
        self.low_ell = low_ell
        if low_ell:
            _, self.d_lo, self.s_lo = np.genfromtxt(os.path.join(LOW_DIR, "CTT_bin_low_ell_2018.dat"), unpack=True)
            lo_min = np.loadtxt(os.path.join(LOW_DIR, "blmin_low_ell.dat")).astype(int)
            lo_max = np.loadtxt(os.path.join(LOW_DIR, "blmax_low_ell.dat")).astype(int)
            w_lo = np.loadtxt(os.path.join(LOW_DIR, "bweight_low_ell.dat"))
            self.B_lo = np.zeros((len(self.d_lo), LMAX_NEEDED + 1))
            for i in range(len(self.d_lo)):
                self.B_lo[i, lo_min[i] + 2: lo_max[i] + 3] = w_lo[lo_min[i]: lo_max[i] + 1]
        _, self.data, self.sig = np.genfromtxt(os.path.join(DIR, "cl_cmb_plik_v22.dat"), unpack=True)
        blmin = np.loadtxt(os.path.join(DIR, "blmin.dat")).astype(int)
        blmax = np.loadtxt(os.path.join(DIR, "blmax.dat")).astype(int)
        w = np.loadtxt(os.path.join(DIR, "bweight.dat"))
        assert len(self.data) == NBIN, len(self.data)
        f = FortranFile(os.path.join(DIR, "c_matrix_plik_v22.dat"), "r")
        cov = f.read_reals(dtype=float).reshape((NBIN, NBIN))
        f.close()
        iu = np.triu_indices(NBIN, 1)
        cov[iu] = cov.T[iu]                              # upper triangle from the lower one
        self.cov = cov
        self.cho = cho_factor(cov)
        # binning matrices acting on C_l for l = 0 .. LMAX_NEEDED
        self.B_tt = np.zeros((NTT, LMAX_NEEDED + 1))
        for i in range(NTT):
            self.B_tt[i, blmin[i] + LMIN: blmax[i] + LMIN + 1] = w[blmin[i]: blmax[i] + 1]
        self.B_pol = self.B_tt[:NTE].copy()               # same bins and weights for TE, EE
        self.icov_data = cho_solve(self.cho, self.data)

    def bin(self, Dl_tt, Dl_te, Dl_ee):
        """Inputs are D_l = l(l+1) C_l / 2 pi in muK^2 for l = 0 .. >= 2508."""
        l = np.arange(LMAX_NEEDED + 1)
        fac = np.where(l > 0, 2 * np.pi / np.maximum(l * (l + 1), 1), 0.0)
        return np.concatenate([self.B_tt @ (Dl_tt[: LMAX_NEEDED + 1] * fac),
                               self.B_pol @ (Dl_te[: LMAX_NEEDED + 1] * fac),
                               self.B_pol @ (Dl_ee[: LMAX_NEEDED + 1] * fac)])

    def bin_low(self, Dl_tt):
        l = np.arange(LMAX_NEEDED + 1)
        fac = np.where(l > 0, 2 * np.pi / np.maximum(l * (l + 1), 1), 0.0)
        return self.B_lo @ (Dl_tt[: LMAX_NEEDED + 1] * fac)

    def chi2(self, Dl_tt, Dl_te, Dl_ee, scale=1.0):
        """chi2 with the model spectra multiplied by `scale`."""
        r = self.data - scale * self.bin(Dl_tt, Dl_te, Dl_ee)
        c = float(r @ cho_solve(self.cho, r))
        if self.low_ell:
            c += float(np.sum(((self.d_lo - scale * self.bin_low(Dl_tt)) / self.s_lo) ** 2))
        return c

    def chi2_best_scale(self, Dl_tt, Dl_te, Dl_ee):
        """Minimise over one overall amplitude of the model spectra (analytic).

        The amplitude stands for the combination A_s exp(-2 tau) / y_cal^2, which is all
        the high-l spectra constrain apart from the small lensing dependence on A_s."""
        m = self.bin(Dl_tt, Dl_te, Dl_ee)
        im = cho_solve(self.cho, m)
        num, den = float(m @ self.icov_data), float(m @ im)
        if self.low_ell:
            ml = self.bin_low(Dl_tt)
            num += float(np.sum(ml * self.d_lo / self.s_lo ** 2))
            den += float(np.sum(ml * ml / self.s_lo ** 2))
        s = num / den
        return self.chi2(Dl_tt, Dl_te, Dl_ee, scale=s), s


if __name__ == "__main__":
    ls, tt, te, ee = np.genfromtxt(os.path.join(HERE, "data", "Dl_planck2015fit.dat"), unpack=True)
    l0 = int(ls[0])
    pad = lambda a: np.concatenate([np.zeros(l0), a])
    for low, ref in ((False, -291.33481235418026), (True, -293.95586501795134)):
        if low and not os.path.exists(os.path.join(LOW_DIR, "CTT_bin_low_ell_2018.dat")):
            print("low-l files not found in data/planck2018_low_ell; skipped")
            continue
        L = PlikLite(low_ell=low)
        ll = -0.5 * L.chi2(pad(tt), pad(te), pad(ee))
        print(f"{'high-l + low-l TT' if low else 'high-l only':18s}: log-likelihood {ll:.10f}, "
              f"expected {ref:.10f}, difference {ll - ref:.1e}")
