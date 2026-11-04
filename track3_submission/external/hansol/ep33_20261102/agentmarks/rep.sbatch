#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=node200
#SBATCH -o /data1/hansol/ep33_20261102/logs/agentrep_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; A=$W/agentmarks; AR=$W/code/spoon_annot_roll2.py
V=/home/hansol/v2d/outputs/variants_20261030
declare -A B=([r6]=$W/orient/fit4/recon_E14_best_B2A_S1_HGAZr6 [c18]=$W/orient/fit5/recon_HGAc18 [v14]=$V/recon_E14_best_B2A_S1_HGA)
for k in r6 c18 v14; do
  [ -d ${B[$k]}/episode_000033 ] || { echo "missing $k"; continue; }
  $PY $AR $A/rep_$k 33 $A/agent_marks.json 0 346 --base ${B[$k]} --face-thr 0.25 --report-only > $A/rep_agent_$k.txt 2>&1 &
  $PY $AR $A/repu_$k 33 $W/orient/ep33_marks_merged_20261103d.json 0 346 --base ${B[$k]} --face-thr 0.25 --report-only > $A/rep_user_$k.txt 2>&1 &
done; wait; echo done
