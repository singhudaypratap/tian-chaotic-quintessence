#!/bin/sh
cd "$HOME/cfc_followup"
PY="$HOME/miniconda3/bin/python3"
$PY lensing_check.py fit_grid_l2_0.01_all.json grid_l2_0.01.jsonl class_field_grid_l2_0.01_v3.json --top 10 --procs 6 --out lensing_grid.json > logs/lensing_grid.log 2>&1
$PY lensing_check.py fit_part4_all.json part4.jsonl class_field_part4_v3.json --top 3 --procs 3 --out lensing_part4.json > logs/lensing_part4.log 2>&1
