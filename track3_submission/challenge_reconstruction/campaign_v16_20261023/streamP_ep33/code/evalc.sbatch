#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=P-evalc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/evalc-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/code/eval_on_corrected.py "$@"
