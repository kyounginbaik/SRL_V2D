#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=node200
#SBATCH -o /data1/hansol/ep33_20261102/logs/agentrep2_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; A=$W/agentmarks; AR=$W/code/spoon_annot_roll2.py
cd $A && $PY mkb.py
$PY $AR $A/rep_h5 33 $A/agent_marks.json 0 346 --base $A/bundle_H5r16 --face-thr 0.25 --report-only > $A/rep_agent_h5.txt 2>&1 &
$PY $AR $A/repu_h5 33 $W/orient/ep33_marks_merged_20261103d.json 0 346 --base $A/bundle_H5r16 --face-thr 0.25 --report-only > $A/rep_user_h5.txt 2>&1 &
wait
