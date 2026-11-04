#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=K32-rollcmp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/logs/%x-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/code/rollcmp.py "$@" 2>&1 | grep -v Warn
