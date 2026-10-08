#!/bin/bash
#SBATCH --job-name=t3-public-mvsam3d
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/mvsam3d_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/mvsam3d_%j.err
set -euo pipefail
weights=/data1/SRL_V2D/reconstruction/data/weights/sam3d
repo=/data1/SRL_V2D/challenge_reconstruction/vendor/MV-SAM3D
object="${V2D_MV_OBJECT:-white_pot}"
input="${V2D_MV_INPUT:-/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/mvsam3d/white_pot_5v/input}"
views="${V2D_MV_IMAGE_NAMES:-000000,000150}"
seed="${V2D_MV_SEED:-42}"
pointmaps="${V2D_MV_POINTMAPS:-}"
extra=()
if [[ -n "$pointmaps" ]]; then
  extra+=(--da3_output "$pointmaps")
fi
echo "object=$object input=$input views=$views seed=$seed pointmaps=$pointmaps"
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME="$weights/hf_home" -e TORCH_HOME="$weights/torch_home" \
  -v /data1:/data1 -w "$repo" \
  v2d_sam3d:latest python run_inference_weighted.py \
  --input_path "$input" --mask_prompt "$object" \
  --image_names "$views" --seed "$seed" "${extra[@]}"
