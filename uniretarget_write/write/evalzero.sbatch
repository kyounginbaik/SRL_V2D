#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000,asus_6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# Usage: sbatch -J e eval.sbatch TASK ARM SEED   Scores every saved checkpoint of a run at VOC 0.
T=$1; ARM=$2; SEED=$3; R=/scratch2/exaFLOPs/srl_yam_run; O=$R/ablate/oak; L=$O/fc_write; D=runs/$T/$ARM-s$SEED; TMP=/tmp/abe_$SLURM_JOB_ID; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
for c in $(echo 100007936); do apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,BOARD_JSON=${BJ-/oak/write/board.json},PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$L:/workspace:ro" -B "$O:/oak" -B "$O/.warp:/tmp/.cache/warp" -B "$O/.assets:/workspace/src/flash_chord/assets/.cache" --pwd /workspace $R/flash_chord_yam.sif python /oak/eval_ref.py /oak/$D/policy_$c.safetensors /oak/$D/zero_$c.npz 32 2>&1 | grep -E '^EVAL|^FRAMES|Error|Traceback' | cut -c1-1800; done; echo EVAL_DONE
