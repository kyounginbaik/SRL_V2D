#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=h2hS
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/logs/%x-%j.log
# usage: sbatch -J NAME run_ep.sbatch SCRIPT ARGS...
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /tmp; /home/exaFLOPs/miniforge3/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/code/"$@"
