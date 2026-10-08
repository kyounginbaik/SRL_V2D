#!/bin/bash
#SBATCH --job-name=t3-mesh-smooth-score
#SBATCH --array=0-9%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/smoothing_ablation/%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/exaFLOPs/miniforge3/bin/python /data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/smoothing_ablation/run.py
