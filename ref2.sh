#!/bin/sh
cd "$HOME/cfc_followup"
"$HOME/miniconda3/bin/python3" class_field.py fit_part4_clean0_all.json part4_clean0.jsonl --top 15 --procs 8 --maxfev 400 --rounds 4 --restart class_field_part4_clean0.json --out class_field_part4_clean0_r2.json > logs/class_part4_clean0_r2.log 2>&1
