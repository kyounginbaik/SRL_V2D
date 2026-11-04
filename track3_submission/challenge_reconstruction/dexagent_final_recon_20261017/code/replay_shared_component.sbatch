#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=portable-shared
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/portable-shared-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
episode=$(printf 'episode_%06d' "$1")
handoff="$task_root/refinement_20261018/shared_policy_handoff/$episode"
cd /tmp
"$task_root/.venv/bin/python" "$handoff/portable/code/reproduce_policy_component.py" \
  replay --bundle "$handoff/portable" --output "$handoff/portable_validation"
