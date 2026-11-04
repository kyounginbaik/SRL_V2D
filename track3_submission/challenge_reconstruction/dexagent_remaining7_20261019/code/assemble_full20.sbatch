#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=assemble-full20
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/assemble-full20-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
TASK_BASE=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
TASK_OUTPUT=${ASSEMBLE_OUTPUT:-/data1/SRL_V2D/challenge_submissions/track3_full_20261019/all_20_episodes}
BEST_EFFORT_25=${BEST_EFFORT_25:-$TASK_BASE/components_best_effort/episode_000025_push_pickup_v1}
TASK_ARGS=()
for TASK_EPISODE in 14 15 16 20 28 30; do
    printf -v TASK_NAME 'episode_%06d' "$TASK_EPISODE"
    TASK_ARGS+=(--component "$TASK_EPISODE=$TASK_BASE/common_components/$TASK_NAME")
    TASK_ARGS+=(--portable "$TASK_EPISODE=$TASK_BASE/portable_archives/${TASK_NAME}_portable.tar.gz")
done
if [ -n "${COMPONENT_25:-}" ]; then
    TASK_ARGS+=(--component "25=$TASK_BASE/common_components/episode_000025" --portable "25=$TASK_BASE/portable_archives/episode_000025_portable.tar.gz")
else
    TASK_ARGS+=(--best-effort "25=$BEST_EFFORT_25")
fi
cd /data1/SRL_V2D
"$TASK_PY" -u "$TASK_BASE/package_code/assemble_twenty_best_effort.py" "${TASK_ARGS[@]}" --output "$TASK_OUTPUT" --archive
