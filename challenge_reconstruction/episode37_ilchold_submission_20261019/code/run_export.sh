#!/usr/bin/env bash
# Run only inside a Slurm GPU allocation. Does not request/cancel allocations.
set -euo pipefail
RUN_ROOT=$(realpath "${1:?Pass source episode run root}")
OUTPUT=$(realpath -m "${2:?Pass new output directory outside source run}")
CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
: "${SLURM_JOB_GPUS:?Requires an allocated Slurm GPU}"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
OUTPUT_PARENT=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_PARENT"
IMAGE=sha256:22d733893d932297a65adfa788d663a42c0778842d96aa20ae2713c07b69971c
GPU_UUID=$(python3 "$RUN_ROOT/code/allocated_gpu.py")
CONTAINER_NAME="ilchold37-export-${SLURM_JOB_ID}-$$"
docker_command() { sg docker -c "$(printf '%q ' docker "$@")"; }
trap 'docker_command stop "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
GPU_ARGS=(--gpus "device=$GPU_UUID")
GPU_MINOR=${SLURM_JOB_GPUS%%,*}
if [[ "$GPU_MINOR" =~ ^[0-9]+$ ]]; then
  GPU_ARGS+=(--device "/dev/nvidia$GPU_MINOR" --device /dev/nvidiactl --device /dev/nvidia-uvm)
fi
CMD=(run --rm --name "$CONTAINER_NAME" "${GPU_ARGS[@]}"
  --user "$(id -u):$(id -g)" --memory 60g --memory-swap 60g
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4 -e WANDB_MODE=disabled
  -v /data1:/data1:ro -v "$RUN_ROOT:$RUN_ROOT:ro"
  -v "$CODE_DIR:$CODE_DIR:ro" -v "$OUTPUT_PARENT:$OUTPUT_PARENT"
  -v "$RUN_ROOT/flash_chord:/workspace:ro"
  -v "$RUN_ROOT/runtime_assets/meshes:/workspace/src/flash_chord/assets/meshes:ro"
  -v "$RUN_ROOT/runtime_assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro"
  -v "$RUN_ROOT/runtime_assets/policies:/workspace/src/flash_chord/assets/policies:ro"
  -w /workspace "$IMAGE" python "$CODE_DIR/export_ilchold_rollout.py"
  --run-root "$RUN_ROOT" --output "$OUTPUT" --world-count 32 --world-index 28)
docker_command "${CMD[@]}" &
wait "$!"
