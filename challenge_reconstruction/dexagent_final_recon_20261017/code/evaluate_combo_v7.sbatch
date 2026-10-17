#!/usr/bin/env bash
#SBATCH --job-name=dex-combo-grasp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/combo_v7-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
depths=(3 6)
depth=${depths[$SLURM_ARRAY_TASK_ID]}
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/evaluate_grasp_seed.py" --episode 3 --variant kp5000_r200_d5 --reference "$task_root/refinement_20261017/geometry_combo_v7_press$depth/episode_000003/tracking_reference_geometry.npz" --output-root "$task_root/refinement_20261017/seed_physics_combo_v7_press$depth"
