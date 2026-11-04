#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orient_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; V=/home/hansol/v2d/outputs/variants_20261030
ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json
$PY $W/code/ep33_facing_sheet.py $V/recon_E14_best_B2A_S1_HGA $W/orient/sheet_f180_214.jpg 180 214 2 $ANN --label "HGA latest" &
$PY $W/code/ep33_facing_sheet.py $V/recon_E14_best_B2A_S1_HGA $W/orient/sheet_f240_274.jpg 240 274 2 $ANN --label "HGA latest" &
wait; ls -la $W/orient
