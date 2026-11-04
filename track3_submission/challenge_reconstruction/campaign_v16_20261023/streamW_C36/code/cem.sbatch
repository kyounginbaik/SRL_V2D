#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=W36-cem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/logs/%x-%j.log
# usage: sbatch cem.sbatch ARGS-for-cem36.py   (resumable: re-submit the same args)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd $R/code && timeout 2900 $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code/cem36.py "$@"
