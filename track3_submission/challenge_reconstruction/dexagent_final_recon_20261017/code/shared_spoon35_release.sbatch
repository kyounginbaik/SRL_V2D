#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=spoon35-release-review
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon35-review-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
run_root="$task_root/refinement_20261018"
"$task_root/.venv/bin/python" "$task_root/code/reevaluate_component.py" \
 --baseline-report "$run_root/spoon35_supported_release/both_clear/evaluation/result.json" \
 --output "$run_root/shared_evaluator_v3/episode_000035_release"
"$task_root/.venv/bin/python" "$task_root/code/audit_policy_component.py" \
 --run "$run_root/shared_evaluator_v3/episode_000035_release" \
 --baseline-report "$run_root/expanded_batch/retention/episode_000035/result.json"
