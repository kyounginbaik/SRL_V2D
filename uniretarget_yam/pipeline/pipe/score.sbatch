#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
set -euo pipefail
N=$1; SEED=${2:-42}; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/$EP; TMP=$H/.cache/tmp_s_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",RECON_EP=$E,RECON_SIDE=$SIDE,RECON_OBJ=$IDX -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B $R/comparison:/cmp -B $R/search:/search --pwd /workspace $R/flash_chord_yam.sif python /cmp/score_any.py ${ARM:-${EP}_s$SEED} 16 "$M/out/$EP/${DIRN:-U-seed$SEED}"
echo SCORE_DONE
