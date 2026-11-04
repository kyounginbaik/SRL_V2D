#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/spoon3537/logs/roll36-%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; A=/data1/hansol/spoon3537/am36
$PY /data1/hansol/spoon3537/code/spoon_annot_roll3.py $A/fit 36 $A/agent_marks36.json 200 350 --face-thr 0.25 --free-end > $A/fit.txt 2>&1
tail -3 $A/fit.txt
$PY /data1/hansol/spoon3537/code/mk_s3.py $A/fit
