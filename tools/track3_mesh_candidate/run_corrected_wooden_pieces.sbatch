#!/bin/bash
#SBATCH --job-name=t3-corrected-pieces
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/pieces_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/pieces_%j.err
set -euo pipefail
code="${V2D_MESH_CODE_DIR:-/data1/SRL_V2D/tools/track3_mesh_candidate}"
weights=/data1/SRL_V2D/reconstruction/data/weights/sam3d
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e V2D_SAM2_ONLY_NAMES=wooden_piece_1,wooden_piece_2 \
  -v /data1:/data1 \
  v2d_sam2:latest python "$code/make_sam2_masks.py"
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME="$weights/hf_home" -e TORCH_HOME="$weights/torch_home" \
  -v /data1:/data1 \
  v2d_sam3d:latest python "$code/run_corrected_wooden_pieces.py"
