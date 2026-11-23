"""Symbolic checks for the exponential-quintessence (CLW) system.

1. The paper's Eqs. (7)-(8) and (9)-(10) equal the standard
   Copeland-Liddle-Wands system.
2. Dulac identity: div(B f) = (3/2)(w-1) B with B = 1/(y * Om^(3/2)),
   Om = 1 - x^2 - y^2.
3. The frequency omega is removed from the amplitude equations by
   u -> u exp(i omega N), v -> v exp(i omega N / 2).
"""
import sympy as sp

x, y, lam, w = sp.symbols("x y lambda w", real=True)
g = 1 + w
s32 = sp.sqrt(sp.Rational(3, 2))

# Standard CLW (Copeland, Liddle, Wands 1998), gamma = 1 + w
brk = 2 * x**2 + g * (1 - x**2 - y**2)
fx_clw = -3 * x + lam * s32 * y**2 + sp.Rational(3, 2) * x * brk
fy_clw = -lam * s32 * x * y + sp.Rational(3, 2) * y * brk

# Paper Eqs. (7)-(8)
fx_78 = -sp.Rational(3, 2) * (
    2 * x + (w - 1) * x**3 + x * (w + 1) * (y**2 - 1)
    - sp.sqrt(sp.Rational(2, 3)) * lam * y**2
)
fy_78 = -sp.Rational(3, 2) * y * (
    (w - 1) * x**2 + (w + 1) * (y**2 - 1) + sp.sqrt(sp.Rational(2, 3)) * lam * x
)

# Paper Eqs. (9)-(10)
fx_910 = sp.Rational(3, 2) * (w - 1) * (x - x**3) + y**2 * (
    s32 * lam - sp.Rational(3, 2) * x * (w + 1)
)
fy_910 = sp.Rational(3, 2) * (w + 1) * (y - y**3) - x * y * (
    s32 * lam + sp.Rational(3, 2) * x * (w - 1)
)

print("Eq 7  == CLW x':", sp.simplify(fx_78 - fx_clw) == 0)
print("Eq 8  == CLW y':", sp.simplify(fy_78 - fy_clw) == 0)
print("Eq 9  == CLW x':", sp.simplify(fx_910 - fx_clw) == 0)
print("Eq 10 == CLW y':", sp.simplify(fy_910 - fy_clw) == 0)

# Invariant curves
Om = 1 - x**2 - y**2
dOm = sp.simplify(-2 * x * fx_clw - 2 * y * fy_clw)
print("Om' / Om =", sp.factor(sp.simplify(dOm / Om)), " (so Om = 0 is invariant)")
print("y'  / y  =", sp.simplify(fy_clw / y), " (so y = 0 is invariant)")

# Dulac function
B = 1 / (y * Om ** sp.Rational(3, 2))
div = sp.diff(B * fx_clw, x) + sp.diff(B * fy_clw, y)
ratio = sp.simplify(div / B)
print("div(B f) / B =", sp.factor(ratio))
assert sp.simplify(ratio - sp.Rational(3, 2) * (w - 1)) == 0

# omega is removable from the amplitude equations
N, om, al, Ou, Ov = sp.symbols("N omega alpha Omega_u Omega_v", real=True)
ut = sp.Function("ut")(N)   # rotating-frame amplitudes
vt = sp.Function("vt")(N)
u = ut * sp.exp(-sp.I * om * N)
v = vt * sp.exp(-sp.I * om * N / 2)
vstar = sp.conjugate(vt) * sp.exp(sp.I * om * N / 2)
eq_u = sp.diff(u, N) - ((Ou - sp.I * om) * u + al * v**2)
eq_v = sp.diff(v, N) - ((Ov - sp.I * om / 2) * v - al * u * vstar)
ru = sp.simplify(eq_u * sp.exp(sp.I * om * N))
rv = sp.simplify(eq_v * sp.exp(sp.I * om * N / 2))
print("u-equation in rotating frame: 0 =", ru)
print("v-equation in rotating frame: 0 =", rv)
print("omega absent:", not ru.has(om) and not rv.has(om))
