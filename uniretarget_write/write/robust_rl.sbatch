#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000,asus_6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
# Usage: sbatch -J name robust_rl.sbatch RUN_DIR(rel to oak) STEP   Stress tests of one checkpoint in Newton.
D=$1; C=$2; R=/scratch2/exaFLOPs/srl_yam_run; O=$R/ablate/oak; L=$O/fc_write; TMP=/tmp/abr_$SLURM_JOB_ID; mkdir -p $TMP $O/$D/robust; trap 'rm -rf $TMP' EXIT
for kf in "none 1" "friction 0.7" "friction 0.5" "friction 1.3" "mass 2" "kp 0.5" "kp 2"; do set -- $kf
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,BOARD_JSON=/oak/write/board.json,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$L:/workspace:ro" -B "$O:/oak" -B "$O/.warp:/tmp/.cache/warp" -B "$O/.assets:/workspace/src/flash_chord/assets/.cache" --pwd /workspace $R/flash_chord_yam.sif python /oak/eval_robust.py /oak/$D/policy_$C.safetensors /oak/$D/robust/${1}_$2.npz 32 $1 $2 2>&1 | grep -E '^PERTURB|^EVAL|Error|Traceback' | cut -c1-300; done; echo ROBUST_DONE
