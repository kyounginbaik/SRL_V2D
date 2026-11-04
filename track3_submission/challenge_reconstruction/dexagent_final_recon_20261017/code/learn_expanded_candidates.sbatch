#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-8%9
#SBATCH --job-name=expanded-learned
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/learned-candidates-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
episodes=(6 33 38)
teachers=(retention_coordinates/episode_000006 cup33_retention/seed_72 dustpan_release/episode_000038/case_3)
gains=(0.01 0.03 0.1)
i=$((SLURM_ARRAY_TASK_ID / 3));j=$((SLURM_ARRAY_TASK_ID % 3))
name=$(printf 'episode_%06d' "${episodes[$i]}")
.venv/bin/python code/learn_expanded_episode.py --episode "${episodes[$i]}" \
 --teacher "refinement_20261018/expanded_batch/${teachers[$i]}" \
 --output "refinement_20261018/expanded_batch/learned/${name}/gain_${gains[$j]}" --feedback-gain "${gains[$j]}"
