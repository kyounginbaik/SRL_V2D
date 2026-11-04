#!/bin/bash
#SBATCH -p debug
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/consist_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
/home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/ep33_20261102/orient/consist.py
