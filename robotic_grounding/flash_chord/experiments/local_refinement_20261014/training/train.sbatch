#!/bin/bash
#SBATCH --job-name=train-local-full-d56
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/uniretarget_local_refinement_20261014/training_local/slurm/%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/uniretarget_local_refinement_20261014/training_local/slurm/%j.err
set -euo pipefail
SEED=${1:-43}
PHASE=${2:-pilot}
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
BASE=/data1/SRL_V2D/challenge_reconstruction/uniretarget_local_refinement_20261014/training_local
case "$(hostname -s)" in node01|node100|node200) ;; *) echo 'Only node01, node100, node200 are configured'; exit 2;; esac
GPU_UUID=$(python3 "$BASE/code/allocated_gpu.py")
CONTAINER="uniretarget-wrist-eigend-${SLURM_JOB_ID}"
trap 'sg docker -c "docker stop $CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT
cmd=(docker run --rm --name "$CONTAINER" --gpus "device=$GPU_UUID"
 --device "/dev/nvidia${SLURM_JOB_GPUS%%,*}" --device /dev/nvidiactl --device /dev/nvidia-uvm
 --user "$(id -u):$(id -g)" --memory 90g --memory-swap 90g
 -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4
 -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true
 -v /home/exaFLOPs/.netrc:/tmp/.netrc:ro
 -e WANDB_API_KEY -e WANDB_MODE
 -v /data1:/data1 -v "$BASE/flash_chord:/workspace"
 -v "$BASE/warp_cache:/tmp/.cache/warp"
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/meshes:/workspace/src/flash_chord/assets/meshes:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/policies:/workspace/src/flash_chord/assets/policies:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/.cache:/workspace/src/flash_chord/assets/.cache
 -w /workspace flash_chord:mixed-ep01-media python "$BASE/code/gpu_entry.py" --seed "$SEED" --phase "$PHASE")
sg docker -c "$(printf '%q ' "${cmd[@]}")" &
wait "$!"
