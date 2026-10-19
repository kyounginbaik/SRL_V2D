#!/usr/bin/env bash
#SBATCH --job-name=ep37-fresh-render
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019/slurm/%j-render.log
set -euo pipefail
BASE=/data1/SRL_V2D/challenge_reconstruction/episode37_ilchold_submission_20261019
SNAPSHOT="$BASE/source/episode_000037"
IMAGE=sha256:22d733893d932297a65adfa788d663a42c0778842d96aa20ae2713c07b69971c
: "${SLURM_JOB_GPUS:?Requires a Slurm GPU allocation}"
GPU_UUID=$(python3 "$SNAPSHOT/code/allocated_gpu.py")
NAME="episode37-fresh-render-${SLURM_JOB_ID}"
mkdir -p "$BASE/rendered_fresh/_cache" "$BASE/slurm"
docker_command() { sg docker -c "$(printf '%q ' docker "$@")"; }
trap 'docker_command stop "$NAME" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
CMD=(run --rm --name "$NAME" --gpus "device=$GPU_UUID"
  --user "$(id -u):$(id -g)" --memory 60g --memory-swap 60g
  -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false
  -e OMP_NUM_THREADS=4 -e WANDB_MODE=disabled -e XDG_CACHE_HOME="$BASE/rendered_fresh/_cache"
  -e PYTHONPATH="$SNAPSHOT/reproduction/packaging_python:/workspace/src"
  -v /data1:/data1:ro -v /data3:/data3:ro -v "$BASE:$BASE"
  -v "$SNAPSHOT/flash_chord:/workspace:ro"
  -v "$SNAPSHOT/runtime_assets/meshes:/workspace/src/flash_chord/assets/meshes:ro"
  -v "$SNAPSHOT/runtime_assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro"
  -v "$SNAPSHOT/runtime_assets/policies:/workspace/src/flash_chord/assets/policies:ro"
  -w /workspace "$IMAGE" python "$BASE/render_tools/render_fresh.py" "$@")
docker_command "${CMD[@]}" &
wait "$!"
