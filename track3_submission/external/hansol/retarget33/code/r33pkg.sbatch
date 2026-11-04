#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/retarget33/logs/%x-%j.log
#SBATCH -J R33-pkg
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1
cd /data1/hansol/retarget33/tmp && /home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/retarget33/code/r33pkg.py "$@"
