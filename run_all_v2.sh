#!/bin/bash
# Full pipeline on the GPU server, with GPU and CPU work overlapped.
# Start with:  nohup bash run_all.sh > run_all.log 2>&1 & disown
# Needs: python3 with numpy, scipy, torch (CUDA), camb, the modified classy (class_tian/);
# the data/ folder next to the scripts (DESI and Pantheon+ download themselves).
#
# Schedule
#   GPU lane : Part IV ensemble -> grid (GPU_SHARDS processes share the GPU) -> burn-in check
#   CPU lane A (starts when Part IV is done, runs while the grid is on the GPU):
#              Part IV fits -> CAMB (clean, best) and CLASS, in parallel
#   CPU lane B (starts when the grid is done): grid fits -> posterior -> CAMB and CLASS, in parallel
#   BBN network last, if PRYM_DIR is set.
# Each stage writes its own log in logs/; run_all.log has the stage headers and the summaries.
set -u
cd "$(dirname "$0")"
N_PART4=${N_PART4:-2000000}       # histories for Tian's Part IV parameters
N_GRID=${N_GRID:-500000}          # histories per grid cell
N_BURN=${N_BURN:-200000}          # histories for the burn-in check
L2_LIST=${L2_LIST:-0.01}          # e.g. L2_LIST="0.01 0.1 0.3" to scan lambda_2 as well
TOP=${TOP:-50}                    # best candidates per file sent to the slow CMB stages
CPU_PROCS=${CPU_PROCS:-${PROCS:-24}}  # CPU workers in total (leave room for other jobs on the box)
GPU_SHARDS=${GPU_SHARDS:-3}       # grid processes sharing the GPU (small kernels are launch-bound)
PRYM_DIR=${PRYM_DIR:-}            # path to a PRyMordial clone (optional)
CLASS_FIELD=${CLASS_FIELD:-1}     # 1: run the field-level CMB stage with CLASS
MAXFEV_CAMB=${MAXFEV_CAMB:-160}
MAXFEV_CLASS=${MAXFEV_CLASS:-250}
ENSEMBLE_ARGS=${ENSEMBLE_ARGS:-}
# Quick end-to-end test on a CPU-only machine (numbers meaningless):
#   N_PART4=3000 N_GRID=200 N_BURN=300 TOP=1 CPU_PROCS=8 GPU_SHARDS=2 MAXFEV_CAMB=6 MAXFEV_CLASS=6 \
#   ENSEMBLE_ARGS="--no-lyap" bash run_all.sh

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # no thread oversubscription in pools
HALF=$(( CPU_PROCS / 2 > 0 ? CPU_PROCS / 2 : 1 ))
mkdir -p logs
rm -f part4.jsonl grid_l2_*.jsonl grid_l2_*_shard*.jsonl          # ensemble files are appended to
FAIL=0
stage() { echo "== $(date +%H:%M) $* ==" ; }
# run NAME CMD... : log to logs/NAME.log with CAMB's repeated warning lines removed (they are counted
# by cmb_spectrum.py itself), and print the summary lines to run_all.log
run() {
  local name=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "mismatch in integrated times" > "logs/$name.log"
  local rc=${PIPESTATUS[0]}
  grep -E "precision check|CMB delta chi2 at the best fit|valid [0-9]|step check|grid cells|summary|reliable only|LCDM:|Bayes factor|delta chi2 =|best:|Traceback|Error" "logs/$name.log" | head -n 40 | sed "s/^/  [$name] /"
  [ "$rc" -ne 0 ] && { echo "  [$name] FAILED (exit $rc), see logs/$name.log"; FAIL=1; }
  return 0
}

cmb_stages() {   # SET: CAMB clean | CAMB best | CLASS, side by side
  local S=$1 FIT="fit_$1_all.json" ENS="$1.jsonl"
  [ -s "$FIT" ] || { echo "  $FIT missing (no candidates passed the three tests): CMB stages skipped for $S"; return 0; }
  python3 plik_lite.py > logs/plik_lite_check.log 2>&1
  run "camb_${S}_clean" python3 cmb_spectrum.py "$FIT" "$ENS" --top "$TOP" --procs "$HALF" --check-other --max-gap 0.01 --maxfev "$MAXFEV_CAMB" --out "cmb_spectrum_${S}_clean.json" &
  local p1=$!
  if [ "$CLASS_FIELD" = "1" ] && python3 -c "from classy import Class" 2>/dev/null; then
    ( python3 class_field.py "$FIT" "$ENS" --top "$TOP" --procs 2 --check-only > "class_field_${S}_check.txt" 2>&1
      run "class_${S}" python3 class_field.py "$FIT" "$ENS" --top "$TOP" --procs "$HALF" --maxfev "$MAXFEV_CLASS" --rounds 3 --out "class_field_${S}.json"
      # precision convergence of the CLASS numbers, and where in l the penalty sits
      [ -s "class_field_${S}.json" ] && run "class_check_${S}" python3 class_precision_check.py "$FIT" "$ENS" "class_field_${S}.json" --top 5 --procs 4 --out "class_precision_${S}.json" \
        && mv -f class_residuals.png "class_residuals_${S}.png" 2>/dev/null; mv -f class_residuals.npz "class_residuals_${S}.npz" 2>/dev/null; true ) &
  else
    echo "  modified CLASS not importable: build it with bash class_tian/build_class_tian.sh"; ( true ) &
  fi
  local p2=$!
  wait $p1
  run "camb_${S}_best" python3 cmb_spectrum.py "$FIT" "$ENS" --top "$TOP" --procs "$HALF" --check-other --maxfev "$MAXFEV_CAMB" --out "cmb_spectrum_${S}_best.json"
  wait $p2
}

cpu_lane_part4() {
  stage "CPU: Part IV fits"
  run fit_part4_bao_sn python3 fit_survivors.py part4.jsonl --procs "$CPU_PROCS" --out fit_part4_bao_sn.json
  run fit_part4_all python3 fit_survivors.py part4.jsonl --cmb --growth --bbn --procs "$CPU_PROCS" --out fit_part4_all.json
  stage "CPU: Part IV CMB spectra (CAMB) and field-level CMB (CLASS)"
  cmb_stages part4
  stage "CPU: Part IV lane done"
}

cpu_lane_grid() {
  for L2 in $L2_LIST; do
    stage "CPU: grid fits, lambda_2 = $L2"
    run "fit_grid_${L2}_bao_sn" python3 fit_survivors.py "grid_l2_${L2}.jsonl" --procs "$CPU_PROCS" --out "fit_grid_l2_${L2}_bao_sn.json"
    run "fit_grid_${L2}_all" python3 fit_survivors.py "grid_l2_${L2}.jsonl" --cmb --growth --bbn --procs "$CPU_PROCS" --out "fit_grid_l2_${L2}_all.json"
  done
  stage "CPU: posterior over the grid"
  run posterior python3 posterior_grid.py fit_grid_l2_*_all.json
  for L2 in $L2_LIST; do
    stage "CPU: grid CMB spectra and CLASS, lambda_2 = $L2"
    cmb_stages "grid_l2_${L2}"
  done
}

# ---------------- GPU lane ----------------
stage "GPU: Part IV parameters, $N_PART4 histories"
run part4_ensemble python3 gpu_ensemble.py --l1 10 --alpha 0.24 --n "$N_PART4" --check-step $ENSEMBLE_ARGS --out part4.jsonl
[ -s part4.jsonl ] || { echo "Part IV ensemble produced nothing; stopping"; exit 1; }

cpu_lane_part4 > logs/lane_part4.txt 2>&1 &      # CPU lane A overlaps the grid
LANE_A=$!

for L2 in $L2_LIST; do
  stage "GPU: grid, lambda_2 = $L2, $N_GRID histories per cell, $GPU_SHARDS processes"
  pids=()
  for ((i = 0; i < GPU_SHARDS; i++)); do
    run "grid_${L2}_shard$i" python3 gpu_ensemble.py --grid --shard "$i/$GPU_SHARDS" --l2 "$L2" --n "$N_GRID" $ENSEMBLE_ARGS --out "grid_l2_${L2}_shard$i.jsonl" &
    pids+=($!)
  done
  wait "${pids[@]}"
  cat grid_l2_${L2}_shard*.jsonl > "grid_l2_${L2}.jsonl"
  echo "  grid cells written: $(wc -l < grid_l2_${L2}.jsonl)"
done

cpu_lane_grid > logs/lane_grid.txt 2>&1 &         # CPU lane B
LANE_B=$!

stage "GPU: burn-in check (diagnostic, nothing depends on it)"
run burn_part4 python3 check_burn.py --l1 10 --alpha 0.24 --n "$N_BURN" --out burn_check_part4.json
run burn_l20 python3 check_burn.py --l1 20 --alpha 0.24 --n "$N_BURN" --out burn_check_l20.json
grep -E "^burn|KS" logs/burn_part4.log logs/burn_l20.log | sed 's/^/  /'

wait $LANE_A; echo "--- CPU lane A (Part IV) ---"; cat logs/lane_part4.txt
wait $LANE_B; echo "--- CPU lane B (grid) ---"; cat logs/lane_grid.txt

if [ -n "$PRYM_DIR" ]; then
  stage "BBN network with the field (PRyMordial), top $TOP candidates"
  HERE_DIR="$(pwd)"
  run bbn_selfcheck python3 bbn_network.py --self-check --prym "$PRYM_DIR"
  run bbn_part4 python3 bbn_network.py "$HERE_DIR/fit_part4_all.json" "$HERE_DIR/part4.jsonl" --prym "$PRYM_DIR" --top "$TOP" --procs "$CPU_PROCS" --out "$HERE_DIR/bbn_network_part4.json"
  for L2 in $L2_LIST; do
    run "bbn_grid_${L2}" python3 bbn_network.py "$HERE_DIR/fit_grid_l2_${L2}_all.json" "$HERE_DIR/grid_l2_${L2}.jsonl" --prym "$PRYM_DIR" --top "$TOP" --procs "$CPU_PROCS" --out "$HERE_DIR/bbn_network_grid_l2_${L2}.json"
  done
else
  stage "BBN network skipped: set PRYM_DIR to a PRyMordial clone"
fi
grep -l "FAILED" logs/lane_*.txt >/dev/null 2>&1 && FAIL=1
stage "done (failures: $FAIL)"
exit $FAIL
