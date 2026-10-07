#!/bin/bash
#SBATCH --job-name=t3-public-meshes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/run_all_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/run_all_%j.err

set -euo pipefail
root=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark
code="${V2D_MESH_CODE_DIR:-/data1/SRL_V2D/tools/track3_mesh_candidate}"
weights=/data1/SRL_V2D/reconstruction/data/weights/sam3d
/home/exaFLOPs/miniforge3/bin/python3 "$code/prepare.py"
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /data1:/data1 \
  v2d_sam2:latest python "$code/make_sam2_masks.py" || true
docker run --rm --gpus all --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME="$weights/hf_home" -e TORCH_HOME="$weights/torch_home" \
  -v /data1:/data1 \
  v2d_sam3d:latest python "$code/run_sam3d_batch.py" || true
/home/exaFLOPs/miniforge3/bin/python3 "$code/score.py"
/home/exaFLOPs/miniforge3/bin/python3 -c \
  "import json; from pathlib import Path; d=json.loads(Path('$root/cd_o_summary.json').read_text()); assert d['coverage']['objects']==10, d['coverage']"
