#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-release
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-release-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
task_teacher="$task_root/physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1"
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u "$task_root/code/release_refined_grasp_sweeper_snapshot.py" \
 --source "$task_teacher" --episode-dir "$task_root/physics_collision_v2/episodes/episode_000030" \
 --output "$task_teacher/release_${SWEEPER_OPENING}" --side right --object mini_sweeper \
 --start 11.8 --opening-origin "$SWEEPER_OPENING" --workers 4
