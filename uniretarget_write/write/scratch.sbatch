#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000,asus_6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# Usage: sbatch -J name scratch.sbatch OUTNAME [GENS]
set -euo pipefail
R=/scratch2/exaFLOPs/srl_yam_run; O=$R/ablate/oak; L=$O/fc_write; TMP=/tmp/ol_$SLURM_JOB_ID; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
apptainer exec --nv --containall --no-home --home /tmp --env PIB_SEG=472,PIB_PROBE=1,FRIC_DR=${FRIC_DR-0.6:1.4},WORLDS=${WORLDS:-2048},DEBUG=${DEBUG:-0},SEED=${SEED:-0},X0=${X0:-null},PYTHONUNBUFFERED=1,BOARD_JSON=/oak/write/board.json,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$L:/workspace:ro" -B "$O:/oak" -B "$O/.warp:/tmp/.cache/warp" -B "$O/.assets:/workspace/src/flash_chord/assets/.cache" --pwd /workspace $R/flash_chord_yam.sif python /oak/write/scratch.py /oak/runs/write_g6_r2/base-s44/policy_100007936.safetensors /oak/clips/write_wb_pib128/none/reference.parquet /oak/clips/write_wb/none/reference.parquet /oak/scratch/$1 ${2:-30}
