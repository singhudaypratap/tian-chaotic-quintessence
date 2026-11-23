#!/bin/bash
# Build CLASS (classy 3.4.2.0) with Tian's potential in the scalar-field module.
# Installs into the current Python as the package "classy" (replacing any existing classy).
# Usage:  bash build_class_tian.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/class_tian_build}"
mkdir -p "$WORK" && cd "$WORK"
pip install cython wheel setuptools numpy
URL=https://files.pythonhosted.org/packages/ed/c0/52a9e22d9f5c0ac6ffddab91645c33e29e6e65f2d7b5c7e4acb420000cd9/classy-3.4.2.0.tar.gz
[ -f classy-3.4.2.0.tar.gz ] || curl -sSL -o classy-3.4.2.0.tar.gz "$URL"
rm -rf classy-3.4.2.0 && tar xzf classy-3.4.2.0.tar.gz && cd classy-3.4.2.0
python3 "$HERE/patch_class_tian.py"
pip install --no-build-isolation .
cd /tmp && python3 -c "from classy import Class; c = Class(); c.set({'h': 0.67}); c.compute(); print('classy OK, Omega_r h^2 =', c.Omega_r() * 0.67**2)"
