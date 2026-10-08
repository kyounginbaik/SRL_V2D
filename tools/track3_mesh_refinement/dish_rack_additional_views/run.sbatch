#!/bin/bash
#SBATCH --job-name=t3-rack-newviews
#SBATCH --array=0-5%2
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/dish_rack_additional_views/%A_%a.log
set -euo pipefail
root=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark
frames=(000030 000090 000150)
seeds=(17 42)
frame=${frames[$((SLURM_ARRAY_TASK_ID / 2))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 2))]}
out="$root/dish_rack_additional_views/f${frame}_s${seed}"
mkdir -p "$out"
image="/data1/SRL_V2D/challenge_reconstruction/public/episode_000012/frames/$frame.png"
mask="$root/mvsam3d/dish_rack_tracking/masks/1/$frame.png"
weights=/data1/SRL_V2D/reconstruction/data/weights/sam3d
test -s "$image"
test -s "$mask"
container="v2d-rack-${SLURM_JOB_ID}"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [ ! -s "$out/generated.glb" ]; then
 docker run --rm --name "$container" --gpus "device=${SLURM_JOB_GPUS%%,*}" --network none --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME="$weights/hf_home" -e TORCH_HOME="$weights/torch_home" \
  -v /data1:/data1 v2d_sam3d:latest python -m v2d.sam3d.lib.image_to_mesh \
  --image_path "$image" --mask_path "$mask" --mesh_path "$out/generated.glb" \
  --transform_path "$out/transform.json" --intrinsics_path "$out/intrinsics.json" --weights_dir "$weights" --seed "$seed"
fi
/home/exaFLOPs/miniforge3/bin/python "$root/dish_rack_additional_views/score_one.py" "$out"
