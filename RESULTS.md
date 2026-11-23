# Follow-up to "Nonlinear dynamical regimes of cosmological frequency combs" — verification results

Target paper: Mishra, Trivedi, Ganesan, Nucl. Phys. B 1031 (2026) 117641.
All numbers below come from the scripts in this folder, run on 11 Nov 2026 (Python 3, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0). Seeds are fixed in the scripts.

## 1. Pure exponential potential (the published model)

| Claim | Result | Script |
|---|---|---|
| Paper's Eqs. (7)-(8) and (9)-(10) are the standard Copeland-Liddle-Wands system | Identical, symbolically | `01_symbolic_checks.py` |
| Dulac function B = 1/(y Ω_m^{3/2}), Ω_m = 1 − x² − y² | div(B f) = (3/2)(w − 1) B, symbolically; y = 0 and Ω_m = 0 are invariant | `01_symbolic_checks.py` |
| ω is removable from the amplitude equations | u → u e^{iωN}, v → v e^{iωN/2} removes it exactly | `01_symbolic_checks.py` |
| Full system has no sustained oscillation | 10 parameter sets × 12 random initial conditions: every orbit ends on a known fixed point; residual oscillation ≤ 3e-14. At w = −1 ± 1e-7 the orbit is still drifting after 4000 e-folds (critical slowing down), with no oscillation | `02_exponential_numerics.py` |
| Amplitude system is not chaotic | 168 runs, all bounded. Largest finite-time exponent 0.067, but it falls as 1/T: 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3 at T = 100, 300, 1000, 3000 | `02_exponential_numerics.py`, `02b_lyap_convergence.py` |

Consequence: for w < 1 the published model has no periodic orbit in the physical region, so no limit cycle and no chaos.

Limits of this check:
- Initial amplitudes tested up to 1e3; the paper sweeps to 1e5.
- λ = 1 and 3 only (the paper does not state its λ).
- The figure captions quote "Ω = 0.5, Ω/2 = 0.25", which does not match the paper's own definitions; I integrated the equations as written, not whatever produced the figures.

## 2. Modulated exponential, V = V0 e^{−λκφ}[1 + A sin(νκφ)]

System and Jacobian are in `modsys.py`; results from `03_modulated_cycle_and_chaos.py` and `05_bifurcation_figure.py`.

- **Limit cycle exists.** At (λ, A, ν, w) = (4, 0.3, 2, 0), five random initial conditions reach the same section point to 3e-11. Period 4.18879020 e-folds, equal to 2πλ/(3(1+w)ν) to all printed digits. Non-trivial Floquet multipliers have modulus 0.043, so the cycle is strongly stable.
- **Chaos exists.** At (10, 0.7, 9.085, 0) the largest Lyapunov exponent is 0.123, 0.113, 0.113, 0.110 at T = 300, 750, 1500, 3000 (it converges, unlike section 1). A Jacobian-free two-trajectory estimate gives 0.18. The analytic Jacobian matches finite differences to 8e-11.
- **Route to chaos is period doubling.** At λ = 10, A = 0.7, w = 0: period 1 up to ν = 7.5, period 2 for 7.75–8.75, period 4 at 9.0, chaotic for 9.25–10.0 (exponent up to 0.23; 0.009 at ν = 9.5, possibly a periodic window), and trapping in a de Sitter minimum from ν = 10.25. Potential minima appear at ν = λ√(1−A²)/A = 10.20. See `fig_bifurcation.png`.

## 3. Viability (radiation + matter + field)

`04_viability.py`, 75 parameter cells (λ = 5, 10, 20, 30, 50; A = 0.7, 0.9, 0.97; five values of Aν/(λ√(1−A²)) from 0.8 to 1.5), 24 initial phases per cell with bisection on Ω_φ(today) = 0.69.

| λ | Solutions with Ω_φ(today) = 0.69 | State today | Early Ω_φ (BBN / CMB window) | Verdict |
|---|---|---|---|---|
| 5 | 4 | rolling, w oscillating | 0.23–0.37 / 0.31–0.57 | excluded by a wide margin |
| 10 | 0 | — | — | — |
| 20 | 1 | trapped, w = −1 | 0.074 / 0.053 | fails BBN |
| 30 | 1 | trapped, w = −1 | 0.029 / 0.013 | passes loose CMB bound only |
| 50 | 2 | trapped, w = −1 | 0.007–0.010 / 0.003–0.005 | one passes the strict bound |

- Every solution that passes has the field frozen in a minimum: H(z) differs from ΛCDM by less than 1e-4 for z < 2. There is no late-time oscillation to observe.
- While the field is still rolling (cycle or chaos), Ω_φ today is at most 0.47 (λ = 10), 0.12 (λ = 20), 0.067 (λ = 30), 0.027 (λ = 50). It cannot supply dark energy.
- So within this scan, "oscillating or chaotic today" and "allowed by early-universe bounds" do not overlap.

Bounds used, indicative only: Ω_φ < 0.045 at BBN (Bean, Hansen, Melchiorri 2001); Ω_e < 0.0036 from Planck, which is derived for a specific early-dark-energy parametrisation, plus a looser 0.02. This is a coarse feasibility scan (24 phases), not a likelihood fit; a narrow viable window could have been missed.

## 4. Prior work that limits the novelty claims

- **Dulac / global dynamics of the exponential model:** Alho, Lim, Uggla, "Cosmological global dynamical systems analysis", arXiv:2201.07596, doi 10.1088/1361-6382/ac7769. Their abstract states they construct monotonic functions and a Dulac function for an exponential scalar field with matter. The no-go result must be cited from there, not claimed as new. I have read only the abstract.
- **Period doubling and chaos in oscillating quintessence:** S. X. Tian, Phys. Rev. D 101, 063531 (2020) and Phys. Rev. D 102, 063509 (2020). A sine-modulated exponent, with "oscillating scaling" and "chaotic accelerating" solutions reached by period doubling. Our potential modulates the prefactor instead, but the phenomenon is published.
- **The modulated potential itself:** Dodelson, Kaplinghat, Stewart, Phys. Rev. Lett. 85, 5276 (2000).
- **The claim being corrected:** "Cosmological Frequency Combs", arXiv:2507.01929.

## 5. What is left that is new

1. The specific correction: the published limit-cycle and chaos claims are artefacts of a truncation in which ω is a removable parameter. (The underlying no-go theorem is known.)
2. A Floquet and Lyapunov characterisation of the prefactor-modulated model, including the period formula and the coincidence of the chaotic window with the onset of potential minima. Must be compared against Tian's model before claiming anything.
3. The viability statement in section 3, if it survives a proper likelihood analysis.

## 6. Search for an independent result (11 Nov, second pass)

Candidate tested: the decay of radiation triggers trapping of the field in a potential minimum, fixing the vacuum energy at a time tied to matter-radiation equality.

- **Background fluid controls trapping (holds so far).** `06_trapping_threshold.py`, single fluid, 6 random initial conditions per point, 600 e-folds. In a matter background every orbit is trapped once minima exist (ratio ≥ 1.02) for all three (λ, A) pairs tested. In a radiation background rolling and trapped outcomes coexist: trapped fraction 0.2–0.8 over ratio 1.06–1.40 for A = 0.7; for A = 0.9 everything is trapped from ratio 1.08.
- **Apparent prediction of the "why now" interval (does not hold).** `07_equality_triggered_trapping.py` gave vacuum-matter equality a median 7–8.7 e-folds after matter-radiation equality for several parameter sets (observed: 7.87). `07b_robustness.py` shows this depends on where the run starts: beginning 40 instead of 30 e-folds before today moves the median to 4.0 (λ = 100) and 0.9 (λ = 30), and rescaling the initial x, y by 3 moves it to 7.4–12.6. Integrator tolerance does not matter. The agreement was an artefact of my initial data, not a prediction.
- What the failure indicates: the outcome depends sensitively on initial data, which suggests fractal basins for the final vacuum. I have not measured this.

Related prior work found: "Emergent Dark Energy from Dark Matter", arXiv:1801.09658 (a scalar trapped in one of many local minima after equality, different minima in different patches).

Gaps in Tian, Phys. Rev. D 102, 063509, as reported by an automated reading of the arXiv PDF (not yet checked by me against the text, and I have not checked later citing papers): no Lyapunov exponents, bifurcation diagrams, basin or Floquet analysis; no perturbation or structure-growth calculation (stated as future work); no supernova or CMB fit and no MCMC (chaos is said to complicate global constraints); trapping in minima not discussed.

## 7. Literature pass on the Tian series (11 Nov, third pass)

Read through an automated reader, not line by line by me.

| Part | Reference | Content | Left open |
|---|---|---|---|
| I | Phys. Rev. D 101, 063531; arXiv:1912.13208 | Model, critical points, chaos noted | No data fit: chaos said to make classical statistical methods invalid; initial conditions would be fit parameters; posterior may be multi-peaked. BBN and CMB not used |
| II | Phys. Rev. D 102, 063509; arXiv:2010.03314 | Oscillating scaling and chaotic accelerating solutions, period doubling, H(z) vs BAO for z < 2 | No Lyapunov exponents, basins or Floquet analysis; no growth calculation; no supernova, CMB or MCMC analysis |
| III | Phys. Rev. D 103, 123545; arXiv:2106.14002 | Inflation with the same field | Not relevant to late-time inference |
| IV | Phys. Rev. D 106, 043524; arXiv:2302.12512 | BBN; model survives; radiation era found to be chaotic, not oscillating scaling | Removes Part II's proposed way around the inference problem |

No Part V or later was found. Not checked: papers by other authors citing the series (the INSPIRE citation query did not work through my tools).

Open gap with the best fit: the first statistical confrontation of this chaotic model with CMB, BAO and supernova data, with the chaos handled explicitly in the likelihood.

## 8. Reproduction of Tian's model and first data-facing test (11 Nov, fourth pass)

Code: `tian.py`, `08_tian_reproduction.py`, `09_tian_ensemble.py`, `10_tian_partIV_ensemble.py`.

**Implementation check.** With Tian's initial data and his three quoted fit points (Part II, Fig. 3: λ1, N_eq = 4.25, 81.16; 4.5, 81.94; 4.75, 84.50), my code gives Ω_φ today = 0.678, 0.679, 0.680 and w_φ = −0.86, −0.97, −0.98. These agree to six digits between tolerances 1e-9 and 1e-12. A first version that obtained the matter density by subtraction was wrong (it lost all precision at early times); the radiation-matter mixture is now treated as one fluid with w(N) = (1/3)/(1 + e^{N−N_eq}), the form Tian uses.

**Lyapunov exponents (not in Tian's papers).** Per e-fold, single background fluid, λ2 = 0.05, α = 0.6:

| λ1 | Radiation background | Matter background |
|---|---|---|
| 4.25 | 0 (period 1) | 0 (period 8, a periodic window) |
| 4.5 | 0 (period 1) | 0.16 (chaotic) |
| 4.75 | 0 (period 2) | ≈ 0, still falling at T = 2000 (period 6) |

At λ1 = 4.5 the route in w is period 1 (w ≥ 0.24), period 2 (0.21 to 0.15), chaos from w = 0.12, matching Tian's description. Chaos in the matter era is therefore not generic across his "viable" λ1 values.

Part IV parameters (λ1 = 10, λ2 = 0.01, α = 0.24): exponent 0.23 in radiation, 0.21 in matter. Errors grow by e every 4.4 e-folds. A single history is numerically reproducible at N_eq = 80 but not at N_eq = 100 with tolerance 1e-9 (Ω_φ today 0.038 against 0.867 at tighter tolerance).

**Scalar-field fraction at early times.**

- Part II parameters, Tian's own three fit points: Ω_φ at z = 1100 is 0.89, 0.69, 0.32, and 0.51 to 0.95 during nucleosynthesis. The radiation era is periodic for these parameters, so this statement stands.
- Part IV parameters: **the ensemble statements originally written here are withdrawn** (see section 9). They came from 2400 histories that differed only in N_eq. In the radiation era the system does not depend on N_eq, so those histories followed one trajectory and amounted to about a dozen independent samples.

Limits: one Part IV parameter set, one family of initial conditions (Tian's), 34 matching histories. A scan over (λ1, λ2, α) is needed before saying the model, not this parameter set, is excluded.

## 9. Corrected sampling, parameter scan, and the cosmologist agent's review (11 Nov, fifth pass)

**Sampling error and fix.** Found by the cosmologist agent and confirmed by my own rerun. Histories are now started from states taken 4 e-folds apart along one long radiation-era trajectory (`tian.attractor_states`), then carried through equality (`tian.history_from_state`). The first scan, made with the old sampling, is kept only as `superseded_scan_correlated_sampling.jsonl` and is not used.

**Scan.** `11_tian_parameter_scan.py`: λ2 = 0.01, λ1 = 6, 10, 14, 20, 30, α = 0.08, 0.15, 0.24, 0.4, 0.8, 800 histories per cell (20,000 in total, 2 failed integrations). Tests per history:
- BBN: peak Ω_φ < 0.04 for 3.7e8 < z < 1.03e10 (2e10 K down to 1e9 K, Tian's main nucleosynthesis era);
- recombination window: peak Ω_φ < 0.02 for 50 < z < 5000;
- today: 0.6 < Ω_φ < 0.8 and w_φ < −0.8.

| λ1 | α | Radiation era | BBN | Recomb. | Today | All three |
|---|---|---|---|---|---|---|
| 10 | 0.24 (Part IV) | chaotic, 0.23 | 11.8% | 11.1% | 2.2% | 0 of 800 |
| 14 | 0.24 | chaotic, 0.20 | 26.2% | 20.2% | 2.9% | 3 |
| 14 | 0.80 | chaotic, 0.08 | 41.5% | 40.2% | 0.6% | 4 |
| 20 | 0.24 | chaotic, 0.14 | 43.8% | 41.4% | 1.8% | 7 |
| 20 | 0.80 | chaotic, 0.14 | 50.5% | 48.2% | 0.9% | 7 |
| 30 | 0.15 | chaotic, 0.12 | 52.4% | 45.0% | 0.8% | 4 |
| 30 | 0.40 | chaotic, 0.08 | 67.5% | 67.0% | 1.0% | 8 |

- 33 of 20,000 histories pass all three, in 6 of 25 cells. The other 19 cells have none in 800.
- All seven cells with a regular (non-chaotic) radiation era have zero passes in every test: there Ω_φ never becomes small.
- For Part IV's own parameters the product of the three marginal rates is 3e-4, so zero in 800 is expected; the joint rate there is not measured.
- At α = 0.8 the joint counts are well above the product of the marginals and the passing histories are nearly identical, so those samples are probably not independent.

**Are the passing histories just ΛCDM?** `12_passing_histories.py`. No: the largest difference in H(z) from flat ΛCDM over z < 3 has median 1.25% and range 0.01% to 5.4%; 22 of 33 exceed 1%. w_φ today lies between −0.999 and −0.877, and at α = 0.8 it falls from about −0.5 at z = 2.

**Conclusion at this stage.** The model is not excluded at background level. A fraction of order 1e-3 to 1e-2 of initial conditions on the chaotic attractor survives the three tests in parts of parameter space, and most survivors are distinguishable from ΛCDM at the percent level, which DESI-class data can test. Tian's published parameter sets are disfavoured (Part II strongly, Part IV by a joint rate of order 1e-4 that I have not measured directly).

**Cosmologist agent's review (its own checks, not rerun by me except where stated).**
- BBN: my earlier window was not too wide; with independent sampling 6 to 11% of Part IV histories pass, depending on the window. The earlier "none pass" was the sampling artefact. (My scan gives 11.8% for the same parameters with the 1e9 K window.)
- Tian's quoted BBN cases: N_ini is the N at which T = 1e12 K, so N_eq = N_ini + 18.8 in my convention, and his initial data use the other branch of the arccos (θ0 = 2π − arccos). With that branch the agent reproduces "hidden during BBN, early dark energy at the end" qualitatively, not exactly.
- Recombination, generic today-matching histories for Part IV: θ* off by a median 25%, sound horizon by 27%, growth since recombination a factor 0.89 (range 0.49 to 1.04). θ* cannot be retuned with h because Ω_φ(z) is fixed by the trajectory.
- Growth with a smooth fraction Ω_e: δ ∝ a^p, p = [−1 + sqrt(1 + 24(1 − Ω_e))]/4.
- Bounds it cites: Planck 2015 XIV (arXiv:1502.01590), constant early fraction < 0.0036; Hojjati, Linder, Samsing (arXiv:1304.3724); Poulin et al. (arXiv:1806.10608), transient fraction below roughly 0.4 to 2%. It did not verify the Samsing-Linder-Smith or Karwal-Kamionkowski numbers.
- Its scripts are in `cosmologist_agent/`.

Limits of this pass: 800 histories per cell, so joint rates below about 1e-3 are unresolved; one value of λ2; threshold tests, not likelihoods; no Boltzmann code; BBN by threshold only.

## 10. GPU ensemble script (11 Nov)

`gpu_ensemble.py` advances all histories together with fixed-step RK4 in float64 (PyTorch on CUDA, NumPy fallback). Every history has its own random start and a 300 e-fold radiation-era burn-in, so samples are independent. Tested here only on the NumPy path (no PyTorch in my sandbox): 3000 histories at Part IV parameters give BBN 11.9%, recombination 10.5%, today 1.9%, against 11.8%, 11.1%, 2.2% from the adaptive-solver scan. 9 of 3000 histories diverged and are excluded. Two passed all three tests: joint rate 6.7e-4 (95% interval 1.8e-4 to 2.4e-3), the first direct estimate for Part IV parameters. The CUDA path is untested.

## 11. Data fit and full pipeline (11 Nov)

`gpu_ensemble.py` now also saves, for every history that passes the BBN and recombination tests with 0.45 < Ω_φ(today) < 0.9, its state at z = 50. `fit_survivors.py` recomputes the last 3.9 e-folds with an adaptive integrator and fits DESI DR2 BAO (13 points, covariance from CobayaSampler/bao_data) and Pantheon+ (1580 supernovae after zHD > 0.01 and removing calibrators, STAT+SYS covariance). Free per history: a shift of "today" (equivalent to ω_m), c/(H0 r_d) (analytic), and the supernova magnitude (marginalised). `run_all.sh` runs the Part IV ensemble, the grid, and both fits.

Check of the fit code: flat ΛCDM gives Ω_m = 0.304, c/(H0 r_d) = 29.68, χ² = 1399.8 (BAO 10.9, supernovae 1388.9).

Small test only (3000 histories per parameter set, NumPy path):

| Parameters | Candidates fitted | Best Δχ² vs ΛCDM | Median Δχ² | Within +4 of ΛCDM |
|---|---|---|---|---|
| λ1 = 20, α = 0.24 | 125 | −0.2 | +2215 | 46 |
| λ1 = 10, α = 0.24 (Part IV) | 3 | −0.4 | −0.1 | 3 |

The candidates that fit are the ones closest to ΛCDM (best fits have w_φ ≈ −1 and H(z) within 0.1% of ΛCDM). No improvement over ΛCDM beyond what one extra freedom gives by chance. These numbers are a test of the pipeline, not results.

Not included in the fit: any CMB information, growth data, or a sound-horizon prior.

## 12. Statistics added to the pipeline (11 Nov)

`gpu_ensemble.py`, per parameter set:
- pass rates with Wilson 95% intervals (already present), plus binomial and 20-block standard errors; a block error well above the binomial one signals correlated histories;
- observed joint count against the count expected if the three tests were independent, with interval;
- number of diverged histories;
- finite-time (100 e-fold) Lyapunov exponent in radiation and matter backgrounds, mean and standard error over 2000 histories;
- `--check-step`: rates recomputed with half the integration step.

`fit_survivors.py`, per parameter set:
- χ²/dof and p-value for ΛCDM and for the best history; ΔAIC and ΔBIC (k = 6 against 3);
- 68% interval on Ω_m for the best history;
- number of trials and the chance of the best Δχ² arising among them;
- likelihood ratio to ΛCDM averaged over initial conditions on the attractor, with bootstrap error and effective number of contributing histories (the shift of today and c/(H0 r_d) are profiled, so this is an upper estimate of a Bayes factor);
- likelihood-weighted mean and spread of Ω_m, w0, w(z = 1) and the largest H(z) difference from ΛCDM.

Test at λ1 = 20, α = 0.24, 3000 histories (NumPy path): block and binomial errors agree; joint count is 1.98 times the independent expectation, so the three tests are positively correlated and the product of marginals underestimates the joint rate; halving the step changes the rates by under 1 percentage point, within sampling error; finite-time exponent 0.196 ± 0.002 (radiation), 0.089 ± 0.001 (matter). Fit: best Δχ² = −0.2, ΔAIC = +5.8, ΔBIC = +21.9, averaged likelihood ratio (9.8 ± 1.4)e-3.

Still not in the pipeline: a posterior over (λ1, λ2, α) beyond the grid; CMB and growth likelihoods; a BBN abundance calculation (threshold only); an automatic burn-in adequacy test.

## 13. CMB, growth, BBN, burn-in check and grid posterior (11 Nov)

The cosmologist agent wrote the three physics modules and wired them into copies of the scripts, then was cut off by a usage limit before testing or reporting. I read its code, checked every hard-coded number against the source, ran the tests and merged it.

| Module | What it is | Source, checked by me against the fetched paper | ΛCDM check |
|---|---|---|---|
| `cmb_likelihood.py` | Planck 2018 distance priors: R, l_A, ω_b with correlations. Sound horizon and distance computed with the history's own H(z), using a table of the early field fraction saved by the GPU run | Chen, Huang, Wang, arXiv:1808.05724, Table I and Eqs. 1-10. The paper prints the z* coefficient as 0.0738 but its appendix code uses 0.0783; the code value is used | Fit to the priors alone gives χ² = 0.0 at Ω_m = 0.316, h = 0.674, ω_b = 0.02236; 100θ* = 1.0410 |
| `growth_likelihood.py` | Gold-2017 fσ8 compilation, 18 points, WiggleZ covariance, fiducial-cosmology correction, σ8 as a free amplitude | Nesseris, Pantazis, Perivolaropoulos, arXiv:1703.10538, Table III and Eq. 2.8 | χ² = 13.0 for 18 points at Ω_m = 0.315, σ8 = 0.743 |
| `bbn_abundance.py` | Y_p and D/H from the PRyMordial table as a function of ω_b and extra radiation; the field fraction is mapped to an equivalent number of extra neutrino species | Table from the PRyMordial repository (arXiv:2307.07061); observed Y_p = 0.245 ± 0.003 and D/H = (25.08 ± 0.29)e-6 from the PDG 2025 review | Standard BBN: Y_p = 0.2455, D/H = 25.15e-6, χ² = 0.03; dY_p/dΔN = 0.0126 |

Not built, with the agent's stated reasons: a full CMB spectrum likelihood (CAMB with a tabulated w(a) treats the field as an effective fluid, and a profiled spectrum fit costs minutes per history); a BBN network calculation (the table assumes constant extra radiation through BBN).

`gpu_ensemble.py` now stores, per candidate, 86 bin averages of sqrt(1 − Ω_φ) between z = 1e11 and z = 50. `fit_survivors.py` has `--cmb`, `--growth`, `--bbn`; with the flags off it reproduces the earlier numbers exactly.

Test, 3000 histories at λ1 = 20, α = 0.24 (NumPy path), all data:

| | ΛCDM | Best history |
|---|---|---|
| Total χ² | 1413.86 | +4.2 above ΛCDM |
| CMB priors | 0.60 | 0.36 |
| Growth | 12.90 | 13.06 |
| BBN | 0.03 | 0.03 |
| BAO | 12.31 | 17.7 |

ΔAIC = +8.2, ΔBIC = +19.0. Averaged likelihood ratio (1.1 ± 0.6)e-4, against (9.8 ± 1.4)e-3 with BAO and supernovae only. Effective number of contributing histories 3.5. Cost with all flags: about 2 CPU-seconds per candidate. Part IV parameters gave no candidate in 3000 histories.

`check_burn.py`: pass rates and the distribution of Ω_φ after burn-in lengths of 150, 300, 600, 1200 e-folds. A test with 2500 histories at Part IV parameters and 75, 150, 300 e-folds was inconclusive: shifts of up to 2.9σ between lengths and KS p-values of 0.04 to 0.07. There is a physical reason to take this seriously: where the potential is nearly flat the field moves by about 0.04 rad per e-fold, so excursions with Ω_φ ≈ 1 can last tens of e-folds and the stationary distribution is approached slowly.

`posterior_grid.py`: posterior over grid cells from the averaged likelihood ratios (prior uniform over cells), marginals in λ1 and α, credible sets, and the Bayes factor against ΛCDM. `run_all.sh` can scan λ2 as well (`L2_LIST`).

## 14. Full CMB spectrum stage and BBN network wrapper (11 Nov)

Written by a second cosmologist agent, which finished and reported. I scanned its files for unsafe operations (none: the only network calls download the Planck data files), re-ran its Planck likelihood test, diffed its changes to the shared scripts, merged them, and ran the chain ensemble -> fast fit -> spectrum stage on one candidate here. Its own notes are in `AGENT_STATUS.md`.

| File | Status | Evidence |
|---|---|---|
| `plik_lite.py` | Tested | Own reader of the Planck 2018 plik_lite TT+TE+EE data (613 bins). Reference spectrum gives log-likelihood −291.3348123542, equal to the value quoted in the planck-lite-py repository to 2e-13 (re-run by me; the reference value is in the user's own copy of that repository, and the five data files are byte-identical to that copy) |
| `cmb_spectrum.py` | Partly tested | History -> w(a) table -> CAMB 2.0.5 -> plik_lite χ², with ΛCDM fitted to the same likelihood. Agent: a w = −1 table reproduces ΛCDM spectra to 2e-11; ΛCDM at Planck values gives χ² = 583.9 for 613 bins. Me: the script runs end to end on one candidate with a 6-evaluation fit (96 s). Converged fits and multi-process runs are untested |
| `bbn_network.py` | Untested | Wrapper around PRyMordial's network with the field's density history added to the expansion rate. Written from reading PRyMordial's source; never run against it. `--self-check` must pass on the server before any number is used |
| `gpu_ensemble.py` | Tested, NumPy path | Additionally stores W, the bin-averaged w_φ, per candidate. Rates unchanged |
| `fit_survivors.py` | Tested | Additionally stores the candidate index. Fit results unchanged |
| `run_all.sh` | Syntax only | Two final stages added; the nucleosynthesis stage runs only if `PRYM_DIR` is set |

Open issues:
- In my one-candidate run CAMB printed many "mismatch in integrated times" warnings with the default dark-energy treatment (PPF). The agent saw these only with the alternative fluid treatment. Results for histories that trigger them should not be trusted until understood.
- The unconverged test gave Δχ² = +130 for that candidate; the agent's unconverged test on another candidate gave +9.8. Neither is a result.
- The spectrum χ² is extremely sensitive to the shift of today (the agent measured +560 for a shift of 0.014 e-folds), so convergence of that one parameter decides everything.
- The fast fit's CMB term and CAMB use ω_m(z_eq) relations that differ by about 0.7%, so the fast fit should be used to rank candidates, and its CMB χ² not quoted beside the spectrum result.
- No low-multipole likelihood; τ fixed at 0.054; massless neutrinos in both model and ΛCDM.
- The ensemble must be regenerated with the current `gpu_ensemble.py`; older output lacks W.

## 15. Corrections to the open issues (11 Nov, evening)

**A gap in the survival tests (new finding).** A diagnostic of the CAMB behaviour showed that most candidate histories have a large field episode between the two test windows, at 5000 < z < 3.7e8, where nothing had been tested. In 3000 histories at λ1 = 20, α = 0.24, among those passing all three tests the peak field fraction in that range has median 0.92; only 6% stay below 1%. For the best-ranked candidate the field reached 84% of the energy near z = 3e5. CAMB's background reproduces the history's own field fraction there to a few percent, so this is physics, not a numerical artefact. `gpu_ensemble.py` now records the peak (`gap_max`) for every history and reports its distribution among survivors; it is not used as a cut, because I have no sourced bound for that redshift range.

| Issue | Fix | Evidence |
|---|---|---|
| CAMB warnings | Fortran output is now captured per CAMB call (`GFORTRAN_UNBUFFERED_ALL`, file-descriptor redirect) and the count at the best fit and over the fit is reported; a result is marked unreliable if any occur at the best fit | None occurred in the two converged fits below. The earlier warnings came with PPF and the shift of today as a free parameter |
| Default dark-energy treatment | Now the c_s² = 1 fluid, which is the perturbation content of a canonical scalar field; PPF is evaluated at the best fit as a check | For a history without an early episode the two agree exactly; with an episode they differ by about 250 in χ² |
| Sensitivity to the shift of today | The fit now uses 100θ* and solves the shift with CAMB's background (secant, about 3 calls of 0.5 s). ΛCDM is fitted in θ* as well | Both fits converge smoothly; the ΛCDM fit took 83 evaluations |
| Radiation density, fast fit against CAMB | Not changed, by design. Using CAMB's ω_r inside the distance-prior formulas gives 100θ* = 1.0392 for ΛCDM at Planck values against 1.0410, so the compressed priors must keep their own relation. The spectrum stage no longer inherits the shift from the fast fit, so the 0.7% difference only affects ranking | Documented in `cmb_likelihood.py` |
| No low-multipole data | Two low-ℓ TT bins (2 ≤ ℓ ≤ 29) from the user's planck-lite-py copy added to `plik_lite.py`, on by default | Reference log-likelihoods reproduced: −291.3348123542 (high-ℓ) and −293.9558650180 (with low-ℓ), both to 2e-13 |
| Massless neutrinos | `--mnu` sets one massive neutrino in both model and ΛCDM; default stays 0 so both sides use the same radiation as the ensemble | Not run here |
| Nucleosynthesis wrapper untested | Not fixed: running PRyMordial's code in this sandbox was refused. The self-check must be run on the server | — |

**Converged spectrum fits (Planck 2018 plik_lite TT+TE+EE + low-ℓ TT, 615 bins, plus BAO and supernovae; τ = 0.054 fixed).**

| | Total χ² | CMB part | BAO+SN part | Notes |
|---|---|---|---|---|
| ΛCDM | 1986.88 | 586.99 | 1399.90 | ω_b = 0.02241, ω_c = 0.1195, H0 = 68.09, n_s = 0.9667 |
| Candidate 23 (no episode) | +27.3 | +4.4 | +22.9 | Converged; fluid and PPF agree exactly; reliable. The CMB pulls h to 0.656, where the late expansion fits low-z data badly |
| Candidate 20 (field 84% near z = 3e5) | +200.4 | +192.1 | +8.3 | Not fully converged (120 evaluations); PPF gives +247 more; flagged unreliable |

Both are single candidates at one parameter set, run here as tests. They are not a result for the model.

`run_all.sh` now runs the spectrum stage twice per ensemble: on candidates whose field stays below 1% for 5000 < z < 3.7e8 (`--max-gap 0.01`, where the fluid treatment is safe) and on the best candidates regardless (flagged per history).

What remains for a referee-proof treatment of histories with early episodes: a Boltzmann code that evolves the scalar field itself with Tian's potential (for example CLASS with its scalar-field module and the potential replaced), started from the stored state. Not built.

## 16. Field-level CMB with CLASS: the reliable number (11 Nov, evening)

**Build.** CLASS (classy 3.4.2.0 from PyPI) with its scalar-field potential replaced by Tian's, V = m V0 exp[−(λ1+λ2)φ/2 − α(λ1−λ2) sin(φ/α)/2] (`class_tian/patch_class_tian.py`, `class_tian/build_class_tian.sh`). CLASS solves the Klein-Gordon equation for the field and its perturbations, so no fluid approximation is made. ω_r from CLASS = 4.18259e-5 (CAMB 4.18258e-5).

**Reproduction of the history.** `gpu_ensemble.py` now stores each candidate's state at z = 1e9. `class_field.py` starts CLASS there (φ, φ′ and V0 computed from x, y, θ; CLASS adjusts m so the budget closes today) and compares CLASS's Ω_scf(z) with the pipeline's own integration from the same state. For the two test candidates the largest difference over 1e8 ≥ z ≥ 0 is 1.5e-8 and 2.8e-8, including the episode candidate's 73% field fraction near z = 3e5. CLASS follows the same chaotic history.

**Fits** (CLASS lensed spectra, Planck 2018 plik_lite TT+TE+EE + two low-ℓ TT bins, plus BAO and supernovae; τ = 0.054 fixed; ω_b, n_s, 100θ_s and the amplitude fitted; ΛCDM fitted the same way in CLASS):

| | Δχ² vs ΛCDM | CMB part | BAO+SN part | Converged |
|---|---|---|---|---|
| ΛCDM (CLASS) | 0 (χ² 1987.62) | 587.71 | 1399.91 | yes |
| Candidate 23, no episode, CLASS field | +27.4 | +4.3 | +23.1 | yes |
| Candidate 23, CAMB fluid (section 15) | +27.3 | +4.4 | +22.9 | yes |
| Candidate 20, 84% episode, CLASS field | **+25.6** | +9.9 | +15.7 | yes (after a restart with wider limits) |
| Candidate 20, CAMB fluid (section 15, hit the θ limit) | +200 | +192 | +8.3 | no |
| Candidate 20, CAMB PPF at that point | about +440 | | | — |

- For the episode-free history the two codes agree to 0.1 in χ², which validates both.
- For the episode history the effective-fluid treatments overstated the CMB penalty by a factor of 20 to 40. The field-level result is the one to use; `cmb_spectrum.py` should be trusted only for histories without early episodes.
- The episode candidate's best fit has 100θ_s = 1.0343, against 1.0417 for ΛCDM: with this much early field energy the sound-horizon angle no longer marks the peak positions, so the earlier fit limits (1.035 to 1.050) were too narrow. Both scripts now allow 1.000 to 1.080 for the model and 1.020 to 1.060 for ΛCDM.
- These are two candidates at one parameter set. Both are disfavoured by Δχ² ≈ 26 against ΛCDM; whether every surviving history is disfavoured needs the full run.

Cost: about 9 s per evaluation in CLASS, 100 to 220 evaluations per candidate, so 15 to 35 core-minutes per candidate.

## 17. Precision bug in the background equations, and fix (11 Nov, late)

**Bug.** Ω_bg (radiation + matter) was computed as 1 − x² − y². Once the field dominates, Ω_bg drops below about 1e-16 and the subtraction returns rounding noise, so the time at which the background comes back is set by rounding, not by the equations. This affected every integrator in the project. Symptoms: at λ1 = 14, α = 0.8 the fixed-step pass rates moved with step size (BBN 55.5%, 71.5%, 69.5% at steps 2e-3, 1e-3, 5e-4), DOP853 and Radau disagreed with each other after about 20 e-folds, and some histories went to NaN.

**Fix.** Carry u = ln Ω_bg as a fourth variable, u′ = 3(x² − y² + w e^u − w).
- `gpu_ensemble.py`: after each RK4 step, rescale so that x² + y² + e^u = 1 exactly.
- `tian.py`: new `rhs_u`, `rhs1u`, `rhs_fullu` and `with_u`. The right-hand side is evaluated on the constraint surface, so adaptive trial stages cannot overflow e^u. `history`, `attractor_states` and `history_from_state` use the new equations; `attractor_states` now returns u as a fourth column.
- `class_field.py`: candidate early states now include u (older files without it still load), and Ω_m today is taken from e^u.
- `step_convergence.py`: uses the new equations, and takes the solver and `--adaptive-only` as options.

The old versions are kept as `gpu_ensemble.pre_u.py` and `tian.pre_u.py`. The old 3-variable `rhs1` and `rhs_full` remain, for `08_tian_reproduction.py` and the Lyapunov code.

**Checks after the fix.**

Single trajectories at λ1 = 14, α = 0.8, over 30 e-folds with field-dominated stretches down to ln Ω_bg ≈ −48 (`traj_check2.py`):

| Comparison | Maximum difference |
|---|---|
| RK4, step 5e-4 vs 2.5e-4 | 7e-11 |
| DOP853, rtol 1e-10, vs RK4 | up to 2e-5 |
| DOP853, rtol 1e-12, vs RK4 | up to 5e-7 |
| DOP853, rtol 1e-13, vs RK4 | up to 4e-8 |

The adaptive solvers converge towards fixed-step RK4 as the tolerance tightens. Fixed-step RK4 in the u form is therefore the reference, and DOP853 at rtol 1e-10 is not accurate enough inside field-dominated episodes.

Ensemble rates from the same 100 starting states:

| Cell | Integrator | BBN | Recombination | Today |
|---|---|---|---|---|
| λ1 = 14, α = 0.8 | RK4, step 2e-3 | 47% | 59% | 0% |
| | RK4, step 1e-3 | 48% | 59% | 0% |
| | RK4, step 5e-4 | 48% | 59% | 0% |
| | DOP853, rtol 1e-10 | 37% | 36% | 2% |
| λ1 = 14, α = 0.24 | RK4, step 2e-3 | 32% | 25% | 3% |
| | RK4, step 1e-3 | 28% | 29% | 6% |
| | RK4, step 5e-4 | 33% | 26% | 3% |
| | DOP853, rtol 1e-10 | 32% | 28% | 2% |

- At λ1 = 14, α = 0.8, RK4 is converged in step size, while the adaptive rate still differs.
- At α = 0.24, all integrators agree within sampling noise (±5% at n = 100). Episodes there peak below Ω_φ ≈ 0.94, so Ω_bg never underflows.

**What changes.**
- **Section 9.**
  - The α = 0.8 and α = 0.4 rows, and the λ1 = 30 rows, came from the adaptive solver and are superseded. The new value for λ1 = 14, α = 0.8 is BBN about 48% and recombination about 59%, against 41.5% and 40.2% before.
  - The α ≤ 0.24 rows are expected to stand, and the λ1 = 14, α = 0.24 check above agrees with them.
  - The whole scan is rerun by the grid stage of `run_all.sh` on the server. The counts of histories that pass all three tests will be taken from that rerun.
- **Section 10.** Part IV parameters (α = 0.24) were not affected; the smoke rerun below confirms this.
- **Section 16.** The CLASS number (+25.6) is unaffected, because CLASS integrates the densities directly from the stored z = 1e9 state.
- **Section 15.** The fixed-step "NaN divergences" at large α are gone: the u form cannot leave the physical disc.

## 18. Full run, Part IV parameters (server, 12 Nov)

Ensemble: λ1 = 10, λ2 = 0.01, α = 0.24; 2,000,000 histories on CUDA (u form, step 2e-3); no history diverged.

| Test | Pass rate |
|---|---|
| BBN | 11.91% |
| Recombination window | 10.23% |
| Today | 1.97% |
| All three | 195 histories, rate 9.75e-5 (95% CI 8.5e-5 to 1.1e-4) |

- The product of the three marginal rates is 2.4e-4, so the joint rate falls below it: the tests are anti-correlated.
- Halving the step on 200,000 histories changes every rate by less than 0.1 percentage point. BBN goes 11.88% → 11.96%, recombination 10.17% → 10.19%, today 2.01% → 1.94%.

Fits to BAO + SN + CMB distance priors + growth + BBN: 784 candidates. The best has Δχ² = +0.2 against ΛCDM (w0 = −1.000, max ΔH/H < 1e-3), 6 lie within +4, and the median is +663.

Every candidate has the field reaching Ω_φ = 1.00 for 5000 < z < 3.7e8, between the windows tested.
- The clean CAMB run (peak below 0.01) selected 0 candidates.
- The CAMB best run (top 50) flags all 50 as unreliable; its best Δχ² is +58.7.
- CLASS gives the reliable values. For the top 10 they run from +99 to +190, with one outlier at +1224, nearly all from the CMB. The fits pull n_s to about 0.91, and the background agrees with the stored history to 1e-8 to 3e-8.

Conclusion for Part IV: histories survive the background tests only through a complete field-dominated episode between BBN and recombination. Distance priors miss it; the full CMB spectrum excludes it. The parameter set is excluded.

## 19. Review of the CLASS stage (12 Nov, evening)

Changes to `class_field.py`:
- τ is now fitted, with the Planck 2018 Gaussian prior 0.0544 ± 0.0073, in both the model and the ΛCDM fit. It used to be fixed, which could have biased n_s.
- n_s range widened to 0.85–1.05. The Part IV fits had reached n_s ≈ 0.908, close to the old bound at 0.9.
- Nelder–Mead now restarts from its best point, up to 3 rounds, until it converges (`--rounds`).
- The starting Hubble rate uses u = ln Ω_bg when the stored state has it.
- Each result now records the fitted point (x, including τ), Ω_φ at the CLASS start, and the precision preset used.
- Bug fixed in `--restart`: earlier results were matched by `cand_index` alone, but that number repeats across cells. They are now matched by (λ1, λ2, α, cand_index).

New `class_precision_check.py`:
- Re-evaluates the best fits with CLASS at default and high precision. High precision means a perturbation sampling step of 0.05, tolerance 1e-6, `l_linstep` 20, `l_logstep` 1.03 and accurate lensing.
- Plots the TT, TE and EE residuals against ΛCDM in units of the plik_lite bin errors.

Smoke test, clean candidate (λ1 = 25, α = 0.15): CMB Δχ² is +9.95 at default precision and +9.74 at high, a change of −0.22, so the result is converged.

The Part IV CLASS numbers in section 18 came from the old version (τ fixed, n_s ≥ 0.9). They are rerun with this version.

## 20. Full run 12–13 Nov: results and review

**Reproducibility.** The Part IV rerun gave the identical 195 histories and identical rates.

**Part IV is excluded.** CLASS was fitted with τ free, n_s allowed down to 0.85, and restarts.
- The best of the 50 candidates is Δχ² = +59.3 against ΛCDM (cand 713).
- Every candidate has its early-episode peak at 0.997–1.00.
- The precision check shifts the top-5 values by about +0.5.

**The grid has a viable region.**
- Clean survivors fit the full CMB spectrum (CLASS) plus BAO and SN as well as ΛCDM: Δχ² from −2.4 to +2.
- They lie mainly at α = 0.6–0.8 and λ1 = 8–17, with w0 between −0.90 and −0.98 and peak below 2e-5.
- CAMB and CLASS agree on them to about 0.3.
- With BAO+SN alone, up to 780 candidates per cell beat ΛCDM, the best by Δχ² ≈ −4.8 (w0 ≈ −0.85).
- Against ΛCDM: ΔAIC ≈ +3 and ln B = −9.93 (upper estimate), so ΛCDM is preferred.

**Problems found:**
1. Burn-in is not converged at λ1 = 20, α = 0.24. The today rate goes 0.76% → 1.01% → 1.73% → 2.51% for burns of 150, 300, 600 and 1200 e-folds, with KS p = 0.000 at every step. The grid's 300 e-fold rates are therefore not attractor rates in at least that cell. `burn_scan.py` scans all 54 cells.
2. The CLASS fit stalls on some clean candidates, ending worse than CAMB by 3–14, mainly in BAO+SN, which is computed from the same background (cands 1937, 271, 433, 990). `class_field.py --seed-from` now adds CAMB's best fit as a second start.
3. "Episodes are excluded" is too strong. Cand 1023 (λ1 = 30, α = 0.08) has a peak of 0.80 and a CMB Δχ² of 0.00 in both codes. Only complete domination (peak at least 0.997) is excluded so far. `episode_penalty.py` maps where the line falls.
