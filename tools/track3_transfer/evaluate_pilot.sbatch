#!/bin/bash
#SBATCH --job-name=t3-export-pilot
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_export_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_export_%j.err
set -euo pipefail
BASE=/data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
SLURM_ARRAY_TASK_ID=13
EP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["episode"])' "$BASE/reference_manifest.json" "$SLURM_ARRAY_TASK_ID")
GPU_UUID=$(python "$BASE/code/allocated_gpu.py")
CONTAINER="v2d-t3-${SLURM_JOB_ID}"
trap 'sg docker -c "docker stop $CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT
cmd=(docker run --rm --name "$CONTAINER" --gpus "device=$GPU_UUID" --user "$(id -u):$(id -g)"
 -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4
 -v /data1:/data1 -v "$BASE/runtime_cache/warp:/tmp/.cache/warp"
 -v /data1/track3_transfer_worktree/robotic_grounding/flash_chord:/workspace
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/meshes:/workspace/src/flash_chord/assets/meshes:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs:/workspace/src/flash_chord/assets/urdfs:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/policies:/workspace/src/flash_chord/assets/policies:ro
 -v /data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/assets/.cache:/workspace/src/flash_chord/assets/.cache -w /workspace
 85361c8fbb21 python /data1/video_to_data_challenge/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py
 "evaluation.checkpoint=$BASE/episode_000030/chunk_pilot_training/policy_65536.safetensors"
 evaluation.world_count=1 evaluation.reset_mode=explicit
 "evaluation.metrics_output=$BASE/pilot_results/episode_000030.json"
 "evaluation.object_trajectories_output=$BASE/pilot_results/episode_000030.parquet" logging.mode=disabled)
sg docker -c "$(printf '%q ' "${cmd[@]}")"
