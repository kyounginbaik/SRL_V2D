#!/bin/bash
#SBATCH --job-name=t3-b200-probe
#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/data1/SRL_V2D/slurm_logs/t3_b200_%j.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/t3_b200_%j.err
set -uo pipefail
BASE=/data1/SRL_V2D/challenge_submissions/track3/transfer_hawor_chunk1s_20261009
GPU_UUID=$(python "$BASE/code/allocated_gpu.py")
hostname
id
ls -l /dev/nvidia*
nvidia-smi --query-gpu=index,uuid,name,memory.used,driver_version --format=csv
for USER_SPEC in "$(id -u):$(id -g)" "0:0"; do
 sg docker -c "docker run --rm --gpus device=$GPU_UUID --user $USER_SPEC -e HOME=/tmp --entrypoint bash 85361c8fbb21 -c 'id; ls -l /dev/nvidia*; nvidia-smi; python -c \"import warp as w; w.init(); print(w.get_cuda_devices())\"'"
done
sg docker -c "docker run --rm --gpus device=$GPU_UUID --device /dev/nvidia${SLURM_JOB_GPUS%%,*} --device /dev/nvidiactl --device /dev/nvidia-uvm --user $(id -u):$(id -g) -e HOME=/tmp --entrypoint python 85361c8fbb21 -c 'import warp as w; w.init(); print(w.get_cuda_devices())'"
