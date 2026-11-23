# CLASS with Tian's potential

`class_field.py` needs CLASS built with Tian's potential in its scalar-field module:

    V(phi) = m V0 exp[-(l1+l2) phi/2 - alpha (l1-l2) sin(phi/alpha)/2]

Build on the server (needs a C compiler and internet access to PyPI):

    bash class_tian/build_class_tian.sh

This downloads classy 3.4.2.0 from PyPI, runs `patch_class_tian.py` (which replaces the
six Albrecht-Skordis potential functions in `source/background.c`), and installs it as
`classy`. It replaces any `classy` already installed in that Python; use a virtual
environment if that matters.

Tested in the sandbox (11 Nov 2026): the build succeeds, ΛCDM runs, ω_r = 4.18259e-5
(CAMB: 4.18258e-5), and started from the stored state at z = 1e9 the field fraction of
two histories (one with a 73% episode near z = 3e5) matches the pipeline's own
integration to better than 2e-8 at every redshift checked from 1e8 to 0.
