#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --job-name=spoon-safe-release
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/bimanual-release-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
case=$((SLURM_ARRAY_TASK_ID % 3));openings=(.006 .010 .008);rotations=(0 0 1)
if (( SLURM_ARRAY_TASK_ID < 3 )); then
 ep=6;start=8.35;source=refinement_20261018/expanded_batch/feasible_hand_margin_policy/episode_000006
else
 ep=33;start=13.9;source=refinement_20261018/expanded_batch/learned_supported_release_v2/episode_000033/start_15.80_open_0.008
fi
name=$(printf 'episode_%06d' "$ep")
.venv/bin/python code/fit_bimanual_release_policy.py --source "$source" --output "refinement_20261018/expanded_batch/bimanual_release/$name/case_$case" --side right --object-name wooden_spoon --start "$start" --opening "${openings[$case]}" --place --rotate "${rotations[$case]}"
