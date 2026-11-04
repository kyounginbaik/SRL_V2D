#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=Q25-py
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25/logs/py-%j.log
# generic CPU job: sbatch run_py.sbatch SCRIPT ARGS...
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25 && $R/.venv/bin/python -u "$@"
