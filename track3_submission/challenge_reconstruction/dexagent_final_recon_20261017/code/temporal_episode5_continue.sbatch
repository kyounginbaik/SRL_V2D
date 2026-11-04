#!/usr/bin/env bash
#SBATCH --job-name=ep5-local-continue
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=51-52
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep5-continue-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/temporal_refine.py" \
  --episode 5 --source-run "$task_root/refinement_20261018/temporal_s50/episode_000005" \
  --output "$task_root/refinement_20261018/temporal_continue_s${SLURM_ARRAY_TASK_ID}/episode_000005" \
  --workers "$SLURM_CPUS_PER_TASK" --samples 224 --iterations 30 --seed "$SLURM_ARRAY_TASK_ID"
