#!/bin/bash
#SBATCH --job-name=t3-multi-load
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_multi_load_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_multi_load_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GPU_UUID=$(python /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/allocated_gpu.py)
sg docker -c "docker run --rm --gpus device=$GPU_UUID --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONUNBUFFERED=1 -v /data1:/data1 -v /data1/SRL_V2D/reconstruction/modules:/workspace v2d_task_library_loader python /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/load_episode.py $1"
