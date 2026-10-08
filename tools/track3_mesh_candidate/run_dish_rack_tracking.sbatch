#!/bin/bash
#SBATCH --job-name=t3-rack-masks
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/rack_masks_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/rack_masks_%j.err
set -euo pipefail
root=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark
out="$root/mvsam3d/dish_rack_tracking"
mkdir -p "$out"
cat > "$out/prompts.json" <<EOF
{"prompts":[{"frame_index":0,"object_id":1,"mask_path":"$root/inputs/plastic_dish_rack/mask_000000.png"}]}
EOF
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /data1:/data1 \
  v2d_sam2:latest python /data1/SRL_V2D/reconstruction/modules/v2d_sam2/lib/video_to_masks.py \
  --video_path /data1/video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000012.mp4 \
  --prompts_path "$out/prompts.json" --masks_dir "$out/masks" \
  --weights_dir /data1/SRL_V2D/reconstruction/data/weights/sam2
