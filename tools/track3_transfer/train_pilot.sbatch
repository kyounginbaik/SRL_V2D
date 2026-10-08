#!/bin/bash
#SBATCH --job-name=t3-flashsac-pilot
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_flashsac_pilot_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_flashsac_pilot_%j.err
set -euo pipefail
BASE=/data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
PARQUET="$BASE/episode_000030/reference.parquet"
GPU_UUID=$(python /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/allocated_gpu.py)
sg docker -c "docker run --rm --gpus device=$GPU_UUID --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4 -v /data1:/data1 -v /data1/SRL_V2D/robotic_grounding/flash_chord:/workspace -w /workspace flash_chord:latest python scripts/train_flash_sac.py task.parquet=$PARQUET task.motion_speed=1.0 scene.world_count=64 scene.support=false training.total_environment_steps=262144 training.updates_per_collection=2 training.replay.capacity=131072 training.replay.minimum_size=8192 training.replay.batch_size=512 training.checkpoint.save_interval_environment_steps=131072 logging.mode=disabled output_dir=$BASE/episode_000030/pilot_training"
