#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K8-cem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/logs/%x-%j.log
# usage: STREAMC_EP=35 sbatch [--nodelist=..] cem.sbatch ARGS-for-cemK.py   (resumable: re-submit the same args)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
echo EP=$STREAMC_EP host=$(hostname) args="$@"
cd $R/code && timeout 2900 $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/code/cemK.py "$@"
