#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=K8-exp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/logs/%x-%j.log
# usage: STREAMC_EP=EP sbatch export.sbatch BASE PARAMS_JSON OUT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
echo EP=$STREAMC_EP host=$(hostname)
cd $R/code && $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/code/export_plan.py "$@"
