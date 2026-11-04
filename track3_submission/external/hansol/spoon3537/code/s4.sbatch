#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -o /data1/hansol/spoon3537/logs/s4-%j.log
# ep36 tgt_s4: jump fix + depth-locked position fit (12.5-17.5 s) -> roll fit (agent marks + balls x3 + IoU, 12.5-17.5 s) -> finalize -> render
set -e
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/spoon3537/work/s4; C=/data1/hansol/spoon3537/code
cd /data1/hansol/spoon3537
$PY $C/s4pos.py $W 8
$PY $C/spoon_annot_roll3.py $W/roll 36 /data1/hansol/spoon3537/am36/agent_marks36.json 250 350 --face-thr 0.25 --ball-w 3 --iou 250,350,0.5 --base $W/bundle > $W/roll.txt 2>&1
tail -2 $W/roll.txt | cut -c1-600
$PY $C/s4fin.py $W
$PY $C/render_s4.py 36 /data1/hansol/spoon3537/work/ff/ep36_tgt_s4.mp4 11 18
