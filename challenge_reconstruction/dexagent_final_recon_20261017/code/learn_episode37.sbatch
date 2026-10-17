#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%3
#SBATCH --job-name=learned-pickup37
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/learned37-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
gains=(0.01 0.03 0.1)
gain=${gains[$SLURM_ARRAY_TASK_ID]}
.venv/bin/python code/learn_expanded_episode.py --episode 37 \
 --teacher refinement_20261018/expanded_batch/pickup_joint_margin/episode_000037/right_4 \
 --output "refinement_20261018/expanded_batch/learned/episode_000037/gain_${gain}" --feedback-gain "$gain"
