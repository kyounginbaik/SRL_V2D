#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-passvid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-passvid-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u "$task_root/code/render_selected.py" \
 --episode 30 --episode-dir "$task_root/physics_collision_v2/episodes/episode_000030" \
 --baseline "$task_root/physics_collision_v2/runs/baseline_v1/episode_000030/baseline" \
 --refined "$task_root/physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1" \
 --output "$task_root/physics_collision_v2/review/episode_000030/passing_teacher" --stride 1
