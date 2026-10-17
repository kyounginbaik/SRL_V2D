#!/usr/bin/env bash
#SBATCH --job-name=dex-cups-combine
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/cem_logs/combine-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/combine_hand_references.py" \
 --episode 10 --left "$task_root/refinement_20261017/continuous_references/episode_000010.npz" \
 --right "$task_root/refinement_20261017/cem_v2/episode_000010/reference.npz" \
 --scene "$task_root/refinement_20261017/cem_v2/episode_000010/execution/scene_physics.xml" \
 --output "$task_root/refinement_20261017/hand_combinations/episode_000010/original_left_cem_right"
