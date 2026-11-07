#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
R=/scratch2/exaFLOPs/srl_yam_run; O=$R/ablate/oak; L=$O/fc_write; TMP=/tmp/wd_$SLURM_JOB_ID; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,BOARD_JSON=${BJ:-/oak/write/board.json},OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$L:/workspace:ro" -B "$O:/oak" -B "$O/.warp:/tmp/.cache/warp" -B "$O/.assets:/workspace/src/flash_chord/assets/.cache" --pwd /workspace $R/flash_chord_yam.sif python /oak/write/diag.py "$@" 2>&1 | grep -v -E '^Module |UserWarning|warnings.warn'
