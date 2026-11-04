#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/diag/parity-%j.log
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/recon_000020; TMP=$H/.cache/tmp_d_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,RECON_EP=$E,RECON_SIDE=left,RECON_OBJ=0,RECON_PARQUET=/search/recon_000020/s42/flash_pass/reference.parquet -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B $R/search:/search -B $R/comparison:/cmp -B $R/pipe/diag:/diag --pwd /workspace $R/flash_chord_yam.sif python /diag/parity.py
