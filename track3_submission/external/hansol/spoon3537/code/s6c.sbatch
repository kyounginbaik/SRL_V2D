#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -o /data1/hansol/spoon3537/logs/s6c-%j.log
set -e
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/spoon3537/work/s6c; C=/data1/hansol/spoon3537/code
cd /data1/hansol/spoon3537
$PY $C/s6c.py prep
$PY $C/spoon_annot_roll3.py $W/roll 36 $W/marks36_s6.json 250 340 --face-thr 0.25 --ball-w 3 --iou 250,340,1 --base $W/bundle > $W/roll.txt 2>&1
tail -1 $W/roll.txt | cut -c1-600
$PY $C/s6c.py final
if [ -d /data1/hansol/spoon3537/targets/tgt_s6/episode_000036 ]; then $PY $C/render_s6c.py 36 /data1/hansol/spoon3537/work/ff/ep36_tgt_s6.mp4 12 18; fi
