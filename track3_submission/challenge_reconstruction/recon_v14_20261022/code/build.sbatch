#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=v14-build
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/logs/build-%j.log
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/code/build_v14.py $1 2>&1 | grep -v -i warn
