#!/bin/bash
# Reruns everything that used the CLASS-side replay before the tolerance fix (15 Nov). CPU only.
# Start with:  nohup bash fix2_run.sh > fix2_run.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=${CPU_PROCS:-36}
S=grid_l2_0.01
stamp() { echo "== $(date +%H:%M) $* =="; }
stamp "replay check after the fix (expect mismatched: 0)"
python3 state_consistency.py "fit_${S}_all.json" "$S.jsonl" --top 50 2>/dev/null | grep -E "MISMATCH|mismatched"
python3 state_consistency.py fit_part4_all.json part4.jsonl --top 50 2>/dev/null | grep -E "MISMATCH|mismatched"
stamp "CLASS vs CAMB at identical parameters (the 4 former outliers + 2 controls)"
python3 camb_class_diag.py "fit_${S}_all.json" "$S.jsonl" "cmb_spectrum_${S}_best.json" \
    --cands "10:0.8:1937,12:0.8:271,14:0.8:433,25:0.8:990,10:0.6:414,12:0.8:64" --procs 6 \
    --out camb_class_diag_v3.json 2>&1 | grep -E "cand [0-9]+:|largest|omega_r"
stamp "CLASS refit, grid top 50 (v3)"
python3 class_field.py "fit_${S}_all.json" "$S.jsonl" --top 50 --procs "$P" --maxfev 250 --rounds 3 \
    --seed-from "cmb_spectrum_${S}_best.json" --out "class_field_${S}_v3.json" > "logs/class_${S}_v3.log" 2>&1
grep -E "delta chi2 =|LCDM" "logs/class_${S}_v3.log" | cut -c1-230
stamp "CLASS refit, Part IV top 50 (v3)"
python3 class_field.py fit_part4_all.json part4.jsonl --top 50 --procs "$P" --maxfev 250 --rounds 3 \
    --seed-from cmb_spectrum_part4_best.json --out class_field_part4_v3.json > logs/class_part4_v3.log 2>&1
grep -E "delta chi2 =" logs/class_part4_v3.log | sort -t= -k4 -g | head -n 5 | cut -c1-200
stamp "precision check (grid v3)"
python3 class_precision_check.py "fit_${S}_all.json" "$S.jsonl" "class_field_${S}_v3.json" --top 5 --procs 10 \
    --out "class_precision_${S}_v3.json" 2>&1 | grep -E "precision check|change"
mv -f class_residuals.png "class_residuals_${S}_v3.png" 2>/dev/null
stamp "episode boundary (v3)"
python3 episode_penalty.py class_field_part4_v3.json "class_field_${S}_v3.json" \
    --camb cmb_spectrum_part4_best.json "cmb_spectrum_${S}_best.json" --out episode_penalty_v3.png
stamp "done"
