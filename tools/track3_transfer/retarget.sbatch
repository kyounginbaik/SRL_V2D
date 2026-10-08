#!/bin/bash
#SBATCH --job-name=t3-sharpa-ik
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_sharpa_ik_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_sharpa_ik_%j.err
set -euo pipefail
BASE=/data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
mkdir -p "$BASE/$1/retargeted"
chmod 777 "$BASE/$1/retargeted"
GPU_UUID=$(python /data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009/code/allocated_gpu.py)
sg docker -c "docker run --rm --gpus device=$GPU_UUID --entrypoint /workspace/isaaclab/_isaac_sim/python.sh -e HOME=/tmp -e PYTHONUNBUFFERED=1 -v /data1:/data1 -v /data1/SRL_V2D/robotic_grounding:/workspace/video_to_data/robotic_grounding -w /workspace/video_to_data/robotic_grounding robotic-grounding:latest scripts/retarget/ego_recon_to_sharpa.py --input_dir $BASE/$1/loaded --output_dir $BASE/$1/retargeted --save --smooth_reference --mano_to_robot_scale 1.0"
