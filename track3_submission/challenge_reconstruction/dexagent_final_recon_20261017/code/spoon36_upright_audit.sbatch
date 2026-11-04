#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=spoon36-release-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-release-audit-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
task_run="$task_root/refinement_20261018/episode_000036/right_only_upright/variant_01"
"$task_root/.venv/bin/python" "$task_root/code/audit_object_support.py" "$task_run"
"$task_root/.venv/bin/python" "$task_root/code/audit_full_joint_limits.py" --run "$task_run"
