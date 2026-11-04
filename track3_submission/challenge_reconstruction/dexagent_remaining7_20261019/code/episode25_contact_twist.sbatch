#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=board25-contacttwist
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/board25-contacttwist-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/episode25_contact_twist.py --workers 8
