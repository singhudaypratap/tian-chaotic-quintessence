# Chaotic oscillating quintessence confronted with cosmological data

Code and results for the paper of the same title (submitted to Nuclear Physics B).

Pipeline:
1. `gpu_ensemble.py`, `burn_scan.py`: evolve ensembles of histories (PyTorch, float64) and apply the BBN, recombination and present-day tests.
2. `fit_survivors.py`: background-stage fits (DESI DR2 BAO, Pantheon+, CMB distance priors, f sigma8, BBN).
3. `class_field.py`: full-spectrum fits with a modified CLASS that evolves the field. `class_tian/`: `patch_class_tian.py` applied to classy 3.4.2.0, built by `build_class_tian.sh`; see `CLASS_TIAN.md`. `class_precision_check.py`, `lensing_check.py`: precision, residual and ACT DR6 lensing checks.
4. `posterior_grid.py`, `episode_weight.py`: Bayes factors and posterior weights.
5. `part4_clean.py`: rebuilds the clean Part IV histories; `fig_paper.py`: figures.

`RESULTS.md` records every run, number and file behind the tables and figures.
Likelihood data (DESI DR2, Pantheon+, Planck 2018 plik_lite, ACT DR6 lensing v1.2) are public and are not redistributed here.
Large ensemble files (*.jsonl) are in the Zenodo archive.
