#!/bin/bash
# Frozen-policy tissue-box pilot for the exact online TTSA guided action.
# This is a diagnostic, not an end-to-end training or challenge result.
#SBATCH --job-name=uniret-guide
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2%2
#SBATCH --output=/data1/SRL_V2D/slurm_logs/uniret_guide_%A_%a.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/uniret_guide_%A_%a.err
set -euo pipefail

ROOT=/data1/SRL_V2D/robotic_grounding/flash_chord
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SUFFIX=${SUFFIX:-blend025}
OUTPUT="$ROOT/outputs/uniretargeting_tissue_20261007/online_guided_frozen_seed${SEED}_${SUFFIX}"
mkdir -p "$OUTPUT/cache"
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#GPU_UUIDS[@]} == 1 ]] || { echo 'Expected one allocated GPU'; exit 2; }
CONTAINER=uniret-guide-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
CPUSET=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
trap 'docker stop -t 15 "$CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT
/usr/bin/time -v -o "$OUTPUT/runtime.txt" docker run --rm --name "$CONTAINER" --pids-limit 4096 \
    --runtime=nvidia --cpuset-cpus "$CPUSET" --gpus "device=${GPU_UUIDS[0]}" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -v "$ROOT:/workspace" -v /data1/SRL_V2D/.e2e/tissue_box:/v2d/assets:ro \
    -v "$OUTPUT/cache:/tmp/.cache" -w /workspace flash_chord:latest \
    python scripts/optimize_uniretargeting_reference.py \
    --checkpoint "/workspace/outputs/tissue_ttsa_hand_20261006/fixed_seed$SEED/policy_98304000.safetensors" \
    --output "/workspace/outputs/uniretargeting_tissue_20261007/online_guided_frozen_seed${SEED}_${SUFFIX}" \
    --seed "$SEED" --worlds 2 --search-stride 32 --blend "${BLEND:-0.25}" --online-action
