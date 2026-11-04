#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8-25res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep25/logs/res-%j.log
# usage: sbatch [--nodelist=..] [--cpus-per-task=N] res.sbatch res25.py args...
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep25/tmp
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
cd $TMPDIR; $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep25/code/res25.py "$@"
