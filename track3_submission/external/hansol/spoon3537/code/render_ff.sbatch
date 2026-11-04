#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 24
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/spoon3537/logs/renderff-%j.log
# usage: sbatch render_ff.sbatch EP T0 T1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /data1/hansol/spoon3537/work/ff
cd /data1/hansol/spoon3537 && /home/hansol/v2d/.venvs/v2d-host/bin/python -B code/render_ff.py $1 /data1/hansol/spoon3537/work/ff/ep$1_frontback.mp4 $2 $3
