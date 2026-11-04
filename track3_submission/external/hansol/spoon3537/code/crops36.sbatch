#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o /data1/hansol/spoon3537/logs/crops36-%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
/home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/spoon3537/code/crops36.py /data1/hansol/spoon3537/am36/crops 200-350 2
