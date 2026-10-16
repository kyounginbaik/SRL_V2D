#!/usr/bin/env bash
# Run this copy after prepare.py. One container owns one GPU for all phases.
set -euo pipefail
RUN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PHASE=${1:-full}
SEED=${2:-43}
EPISODE=${3:-episode_000001}
[[ "$EPISODE" =~ ^episode_[0-9]{6}$ ]] || exit 2
case "$PHASE" in preflight|physics-gate|smoke|pilot|full|pack-smoke|pack-pilot) ;; *) echo 'Invalid phase' >&2; exit 2;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
python3 "$RUN_ROOT/cluster/check_source.py"
mkdir -p "$RUN_ROOT/flash_chord/src/flash_chord/assets/.cache" "$RUN_ROOT/warp_cache/${EPISODE}_seed_$SEED"
gpu_args=()
if [[ "$PHASE" != preflight && "$PHASE" != pack-* ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        GPU_DEVICE=$(python3 "$RUN_ROOT/cluster/allocated_gpu.py")
    else
        GPU_DEVICE=${GPU_DEVICE:-0}
    fi
    gpu_args=(--gpus "device=$GPU_DEVICE")
    # Slurm's device cgroup on this cluster also needs the allocated device exposed.
    if [[ -n "${SLURM_JOB_GPUS:-}" && "$SLURM_JOB_GPUS" =~ ^[0-9]+$ ]]; then
        gpu_args+=(--device "/dev/nvidia$SLURM_JOB_GPUS" --device /dev/nvidiactl --device /dev/nvidia-uvm)
    fi
else
    gpu_args=(-e JAX_PLATFORMS=cpu -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
fi
CONTAINER="uniretarget-online-${SLURM_JOB_ID:-local-$$}-s${SEED}"
docker_command() {
    if [[ "${DOCKER_USE_SG:-0}" == 1 ]]; then
        sg docker -c "$(printf '%q ' docker "$@")"
    else
        docker "$@"
    fi
}
trap 'docker_command stop "$CONTAINER" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cmd=(run --rm --name "$CONTAINER" "${gpu_args[@]}"
    --user "$(id -u):$(id -g)" --memory "${CONTAINER_MEMORY:-90g}" --memory-swap "${CONTAINER_MEMORY:-90g}" --shm-size 2g
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4
    -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true
    -e "UNIRETARGET_EPISODE=$EPISODE"
    -e "V2D_SUBMISSION_KIT=$RUN_ROOT/submission_kit"
    -e "UNIRETARGET_RUN_TAG=${UNIRETARGET_RUN_TAG:-wrench-stable-control-20261016}"
    -e "WANDB_MODE=${WANDB_MODE:-disabled}" -e "WANDB_ENTITY=${WANDB_ENTITY:-DexDreamer}"
    -e "WANDB_PROJECT=${WANDB_PROJECT:-v2d-track3-uniretarget}"
    -e WANDB_API_KEY -e "WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}"
    -v "$RUN_ROOT:$RUN_ROOT" -v "$RUN_ROOT/flash_chord:/workspace"
    -v "$RUN_ROOT/warp_cache/${EPISODE}_seed_$SEED:/tmp/.cache/warp" -w /workspace)
if [[ "${WANDB_MODE:-disabled}" == online && -f "${WANDB_NETRC:-$HOME/.netrc}" ]]; then
    cmd+=(-v "${WANDB_NETRC:-$HOME/.netrc}:/tmp/.netrc:ro")
fi
cmd+=("${UNIRETARGET_IMAGE:-flash_chord:uniretarget-online}")
if [[ "$PHASE" == pack-* ]]; then
    cmd+=(python "$RUN_ROOT/cluster/pack_phase.py" "${PHASE#pack-}")
elif [[ "$PHASE" == preflight ]]; then
    if [[ -f "$RUN_ROOT/batch_manifest.json" ]]; then
        cmd+=(python "$RUN_ROOT/cluster/preflight_batch.py")
    else
        cmd+=(python "$RUN_ROOT/cluster/preflight_full.py")
    fi
elif [[ "$PHASE" == physics-gate ]]; then
    cmd+=(python "$RUN_ROOT/cluster/physics_gate_suite.py" --run-root "$RUN_ROOT" --seed "$SEED" --episode "$EPISODE")
else
    cmd+=(python "$RUN_ROOT/$EPISODE/code/gpu_entry.py" --seed "$SEED" --phase "$PHASE")
fi
docker_command "${cmd[@]}" &
wait "$!"
