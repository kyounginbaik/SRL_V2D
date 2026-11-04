#!/bin/bash
#SBATCH --job-name=v14-fpmetric
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/logs/fp_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/logs/fp_%j.err
# Public variant of eval_stereo_current_20261010/episode_000005/pose.sbatch; $1 = episode id, $2 = object.
set -euo pipefail
WS=/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020
export PATH="$WS/code/bin:$PATH"
export TASK_CPUSET=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
export V2D_DOCKER_GPUS=$(/home/exaFLOPs/miniforge3/bin/python $WS/code/allocated_gpu.py)
export V2D_FOUNDATION_POSE_TARGET_SIZE=1014x760
export V2D_DROID_WEIGHTS=/data1/SRL_V2D/reconstruction/data/weights/droid_slam
export V2D_FOUNDATION_POSE_WEIGHTS=/data1/SRL_V2D/reconstruction/data/weights/foundation_pose
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "job=$SLURM_JOB_ID node=$(hostname) gpus=$SLURM_JOB_GPUS docker_gpus=$V2D_DOCKER_GPUS episode=$1 object=$2 start=$(date -Is)"
cleanup() { sg docker -c '/usr/bin/docker ps -q --filter label=v2d_job='"$SLURM_JOB_ID"' | xargs -r /usr/bin/docker stop -t 5' >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT
sg docker -c "/home/exaFLOPs/miniforge3/bin/python3 /data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022/code/run_pose_metric_eval.py $1 $2 $3 $4 $5"
echo "done=$(date -Is)"
