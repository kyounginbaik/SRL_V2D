#!/usr/bin/env bash
#SBATCH --job-name=dex-four-finger
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/v8-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
case "$SLURM_ARRAY_TASK_ID" in
0) version=combo_v8; episode=3; variant=kp5000_r200_d5 ;;
1) version=combo_v8_clamp; episode=3; variant=kp5000_r200_d5 ;;
2) version=v8; episode=5; variant=baseline ;;
3) version=v8; episode=5; variant=kp5000_r200_d5 ;;
esac
printf -v episode_name 'episode_%06d' "$episode"
srun --cpu-bind=cores "$task_root/.venv/bin/python" "$task_root/code/evaluate_grasp_seed.py" --episode "$episode" --variant "$variant" --reference "$task_root/refinement_20261017/geometry_$version/$episode_name/tracking_reference_geometry.npz" --output-root "$task_root/refinement_20261017/seed_physics_$version"
