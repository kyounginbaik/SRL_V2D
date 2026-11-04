#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; V=/home/hansol/v2d/outputs/variants_20261030; O=/data1/hansol/ep33_20261102/depthline
$PY /data1/hansol/mink_S1_20261031/code/handle_depth_line.py S1=$V/recon_E14_best_B2A_S1,HGA=$V/recon_E14_best_B2A_S1_HGA,HGB=$V/recon_E14_best_B2A_S1_HGB 33 blue_cup $O/ep33 > $O/ep33.txt 2>&1
