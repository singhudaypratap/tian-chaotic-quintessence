#!/bin/bash
# Follow-up to the 12-13 Nov full run. Keeps every earlier output; writes new files only.
#   GPU : burn-in scan of all 54 grid cells (burn 300 -> 1000 -> 2000 e-folds) + Lyapunov exponent
#   CPU : CLASS refit of the grid's top 50, with CAMB's best fit as a second start; precision check;
#         CMB penalty against early-episode peak (Part IV + grid)
# Start with:  nohup bash fix_run.sh > fix_run.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
GPU_SHARDS=${GPU_SHARDS:-3}
CPU_PROCS=${CPU_PROCS:-40}
N_SCAN=${N_SCAN:-50000}
BURNS=${BURNS:-300,1000,2000}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
S=grid_l2_0.01
stamp() { echo "== $(date +%H:%M) $* =="; }

stamp "GPU: burn-in scan, $N_SCAN histories per cell, burns $BURNS, $GPU_SHARDS processes"
rm -f burn_scan_*.jsonl
for ((i = 0; i < GPU_SHARDS; i++)); do
  python3 burn_scan.py --shard "$i/$GPU_SHARDS" --n "$N_SCAN" --burns "$BURNS" --out "burn_scan_$i.jsonl" > "logs/burn_scan_$i.log" 2>&1 &
done
GPU_PIDS=$(jobs -p)

(
  stamp "CPU: CLASS refit of the grid top 50 with a second start at CAMB's best fit"
  python3 class_field.py "fit_${S}_all.json" "$S.jsonl" --top 50 --procs "$CPU_PROCS" --maxfev 250 --rounds 3 \
      --seed-from "cmb_spectrum_${S}_best.json" --out "class_field_${S}_v2.json" > "logs/class_${S}_v2.log" 2>&1
  grep -E "delta chi2 =|second start" "logs/class_${S}_v2.log" | cut -c1-220
  stamp "CPU: precision check of the refit"
  python3 class_precision_check.py "fit_${S}_all.json" "$S.jsonl" "class_field_${S}_v2.json" --top 5 --procs 10 \
      --out "class_precision_${S}_v2.json" > "logs/class_check_${S}_v2.log" 2>&1
  mv -f class_residuals.png "class_residuals_${S}_v2.png" 2>/dev/null
  grep -E "precision check|change" "logs/class_check_${S}_v2.log"
  stamp "CPU: CMB penalty against early-episode peak"
  python3 episode_penalty.py class_field_part4.json "class_field_${S}_v2.json" \
      --camb cmb_spectrum_part4_best.json "cmb_spectrum_${S}_best.json" --out episode_penalty.png 2>&1
) > logs/fix_cpu.txt 2>&1 &
CPU_PID=$!

wait $GPU_PIDS
stamp "GPU: burn-in scan done"
cat burn_scan_*.jsonl > burn_scan.jsonl
grep -h "l1=" logs/burn_scan_*.log | sort -t= -k2 -g
echo "cells NOT CONVERGED: $(grep -h 'NOT CONVERGED' logs/burn_scan_*.log | wc -l) of 54"
wait $CPU_PID
echo "--- CPU part ---"; cat logs/fix_cpu.txt
stamp "done"
