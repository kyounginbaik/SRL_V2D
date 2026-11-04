#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=portable-shared5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/portable-shared5-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
bundle="$task_root/refinement_20261018/shared_policy_handoff/episode_000005/portable"
cd /tmp
"$task_root/.venv/bin/python" "$bundle/code/reproduce_policy_component.py" replay --bundle "$bundle" --output "$task_root/refinement_20261018/shared_policy_handoff/episode_000005/portable_validation"
