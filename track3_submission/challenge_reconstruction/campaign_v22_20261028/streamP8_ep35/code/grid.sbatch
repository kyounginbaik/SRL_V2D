#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=P8-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35/logs/%x-%j.log
# usage: sbatch --nodelist=node200 grid.sbatch gridP.py-args...
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35/tmp
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
echo host=$(hostname) args="$@"
cd $R/code && timeout 2950 $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35/code/gridP.py "$@"
