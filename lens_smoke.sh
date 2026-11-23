#!/bin/sh
cd "$HOME/cfc_followup"
"$HOME/miniconda3/bin/python3" lensing_check.py fit_grid_l2_0.01_all.json grid_l2_0.01.jsonl class_field_grid_l2_0.01_v3.json --top 1 --procs 1 --out lensing_smoke.json > logs/lensing_smoke.log 2>&1
