#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=sweeper30-releaseactor
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-releaseactor-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
task_teacher="$task_root/physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1/release_pinch/trial0"
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u "$task_root/code/spoon_actor_grid.py" \
 --episode 30 --source "$task_teacher" --output "$task_teacher/actor_grid" --workers 4
