#!/bin/bash
#SBATCH --job-name=t3-chunk-transfer
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_transfer_%A_%a.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_transfer_%A_%a.err
set -euo pipefail
BASE=/data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
EP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["episode"])' "$BASE/reference_manifest.json" "$SLURM_ARRAY_TASK_ID")
GPU_UUID=$(python "$BASE/code/allocated_gpu.py")
CONTAINER="v2d-t3-${SLURM_JOB_ID}"
trap 'sg docker -c "docker stop $CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT
cmd=(docker run --rm --name "$CONTAINER" --gpus "device=$GPU_UUID" --device "/dev/nvidia${SLURM_JOB_GPUS%%,*}" --device /dev/nvidiactl --device /dev/nvidia-uvm --user "$(id -u):$(id -g)"
 -e V2D_SUBSTEPS=${V2D_SUBSTEPS:-5} -e V2D_CHECK_FINITE=${V2D_FLOAT32:-0} -e V2D_FLOAT32=${V2D_FLOAT32:-0} -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4
 -v /data1:/data1 -v "$BASE/runtime_cache/warp:/tmp/.cache/warp"
 -v /data1/track3_transfer_worktree/robotic_grounding/flash_chord:/workspace
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/meshes:/workspace/src/flash_chord/assets/meshes:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/policies:/workspace/src/flash_chord/assets/policies:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/.cache:/workspace/src/flash_chord/assets/.cache -w /workspace
 85361c8fbb21 python "$BASE/code/run_episode.py" "$EP")
sg docker -c "$(printf '%q ' "${cmd[@]}")" &
DOCKER_LAUNCH_PID=$!
wait "$DOCKER_LAUNCH_PID"
