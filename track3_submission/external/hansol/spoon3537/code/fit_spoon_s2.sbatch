#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /data1/hansol/spoon3537/logs/fit-%j.log
# usage: sbatch fit_spoon.sbatch EP WINDOWS [JUMPS|-]
export TSET=tgt_s2 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /data1/hansol/spoon3537 && /home/hansol/v2d/.venvs/v2d-host/bin/python -B code/fit_spoon.py $1 /data1/hansol/spoon3537/work_s2 $2 ${3:--} 4
