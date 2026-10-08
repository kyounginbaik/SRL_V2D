#!/bin/bash
#SBATCH --job-name=t3-pitcher-steps
#SBATCH --array=0-1%2
#SBATCH --exclude=node200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/pitcher_sampling_steps/%A_%a.log
set -euo pipefail
root=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark
src=/data1/SRL_V2D/challenge_reconstruction/public/episode_000021
weights=/data1/SRL_V2D/reconstruction/data/weights/sam3d
steps=(50 100)
step=${steps[$SLURM_ARRAY_TASK_ID]}
out="$root/pitcher_sampling_steps/steps${step}_seed123_frame120"
mkdir -p "$out"
container="v2d-pitcher-steps-${SLURM_JOB_ID}"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
docker run --rm --name "$container" --gpus "device=${SLURM_JOB_GPUS%%,*}" --network none --shm-size=8g \
 --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME="$weights/hf_home" -e TORCH_HOME="$weights/torch_home" -v /data1:/data1 \
 v2d_sam3d:latest python -m v2d.sam3d.lib.image_to_mesh \
 --image_path "$src/frames/000120.png" --mask_path "$src/masks/1/000120.png" \
 --mesh_path "$out/generated.glb" --transform_path "$out/transform.json" --intrinsics_path "$out/intrinsics.json" \
 --weights_dir "$weights" --seed 123 --stage1_inference_steps "$step"
/home/exaFLOPs/miniforge3/bin/python "$root/pitcher_sampling_steps/score_one.py" "$out"
