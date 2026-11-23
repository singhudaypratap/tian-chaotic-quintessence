#!/bin/sh
# Referee items, step 1: the 15 clean Part IV histories (0 e-fold pre-history) through the full CMB fit.
# Run with: systemd-run --user --unit=ref1 /bin/sh $HOME/cfc_followup/run_referee.sh
cd "$HOME/cfc_followup" || exit 1
PY="$HOME/miniconda3/bin/python3"
P=8
mkdir -p logs
echo "start $(date)" > logs/referee.log
$PY part4_clean.py > logs/part4_clean.log 2>&1 || { echo "part4_clean failed" >> logs/referee.log; exit 1; }
echo "histories done $(date)" >> logs/referee.log
$PY fit_survivors.py part4_clean0.jsonl --cmb --growth --bbn --procs $P --out fit_part4_clean0_all.json \
    > logs/fit_part4_clean0.log 2>&1 || { echo "fit_survivors failed" >> logs/referee.log; exit 1; }
echo "background fits done $(date)" >> logs/referee.log
$PY class_field.py fit_part4_clean0_all.json part4_clean0.jsonl --top 15 --procs $P --maxfev 250 --rounds 3 \
    --out class_field_part4_clean0.json > logs/class_part4_clean0.log 2>&1 || { echo "class_field failed" >> logs/referee.log; exit 1; }
echo "CLASS fits done $(date)" >> logs/referee.log
