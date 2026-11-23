#!/bin/sh
cd "$HOME/cfc_followup"
for P in 20 40; do
  "$HOME/miniconda3/bin/python3" episode_weight.py fit_grid_phys${P}_all.json grid_phys${P}_l2_0.01.jsonl
done > logs/episode_weight_v2.log 2>&1
