"""Replace CLASS's scalar-field potential with Tian's (used by class_field.py).

Run from the root of an unpacked CLASS / classy source tree (tested with classy 3.4.2.0
from PyPI). Replaces the Albrecht-Skordis functions V_e_scf ... ddV_p_scf in
source/background.c; V_scf, dV_scf, ddV_scf (their product rule) are kept.
"""
import sys

p = "source/background.c"
s = open(p).read()
if "MODIFIED for the follow-up study" in s:
    sys.exit("already patched")
i = s.index("double V_e_scf(struct background *pba,")
j = s.index("/** Fianlly we can obtain the overall potential")
NEW = "/* ------------------------------------------------------------------------------------\n * MODIFIED for the follow-up study of Tian's oscillating quintessence\n * (Phys. Rev. D 101, 063531 (2020); potential as in Part IV, arXiv:2302.12512, Eq. 19):\n *     V(phi) = m * V0 * exp[-(l1+l2) phi/2 - alpha (l1-l2) sin(phi/alpha)/2]\n * written as V = V_e * V_p with\n *     V_e = m * V0 * exp(-(l1+l2) phi / 2),   V_p = exp(g(phi)),\n *     g   = -alpha (l1-l2) sin(phi/alpha) / 2.\n * scf_parameters = [l1, l2, alpha, m, V0, phi_ini, phi_prime_ini]\n * (m is the shooting parameter, scf_tuning_index = 3; V0 in m_pl^2/Mpc^2, phi in m_pl.)\n * The original Albrecht-Skordis functions were removed.\n * ------------------------------------------------------------------------------------ */\nstatic double tian_l1(struct background *pba)  { return pba->scf_parameters[0]; }\nstatic double tian_l2(struct background *pba)  { return pba->scf_parameters[1]; }\nstatic double tian_al(struct background *pba)  { return pba->scf_parameters[2]; }\n\ndouble V_e_scf(struct background *pba, double phi) {\n  return pba->scf_parameters[3] * pba->scf_parameters[4] * exp(-0.5 * (tian_l1(pba) + tian_l2(pba)) * phi);\n}\ndouble dV_e_scf(struct background *pba, double phi) {\n  return -0.5 * (tian_l1(pba) + tian_l2(pba)) * V_e_scf(pba, phi);\n}\ndouble ddV_e_scf(struct background *pba, double phi) {\n  double c = 0.5 * (tian_l1(pba) + tian_l2(pba));\n  return c * c * V_e_scf(pba, phi);\n}\nstatic double tian_g(struct background *pba, double phi) {\n  return -0.5 * tian_al(pba) * (tian_l1(pba) - tian_l2(pba)) * sin(phi / tian_al(pba));\n}\nstatic double tian_dg(struct background *pba, double phi) {\n  return -0.5 * (tian_l1(pba) - tian_l2(pba)) * cos(phi / tian_al(pba));\n}\nstatic double tian_ddg(struct background *pba, double phi) {\n  return 0.5 * (tian_l1(pba) - tian_l2(pba)) * sin(phi / tian_al(pba)) / tian_al(pba);\n}\ndouble V_p_scf(struct background *pba, double phi) {\n  return exp(tian_g(pba, phi));\n}\ndouble dV_p_scf(struct background *pba, double phi) {\n  return tian_dg(pba, phi) * V_p_scf(pba, phi);\n}\ndouble ddV_p_scf(struct background *pba, double phi) {\n  double d = tian_dg(pba, phi);\n  return (tian_ddg(pba, phi) + d * d) * V_p_scf(pba, phi);\n}\n\n"
open(p, "w").write(s[:i] + NEW + s[j:])
print("patched", p)
