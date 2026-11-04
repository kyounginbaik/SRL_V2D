#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J noise noise.sbatch   "N stage side obj parquet" lines on stdin file $1: does a grasp survive action noise of the size the policy produces in training?
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; TMP=$H/.cache/tmp_n_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
while read N ST SIDE IDX P; do EP=$(printf recon_%06d $N); C=/uniretarget_yam/episodes/$EP; echo "=== ep$N stage $ST $P"
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",RECON_HOLDS_UPTO=$ST,RECON_EP=$C,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,RECON_PARQUET=$P -B "$TMP:/tmp" -B "$H:/uniretarget_yam" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B "$R/comparison:/cmp" -B "$R/search:/search" -B /scratch2 --pwd /workspace $R/flash_chord_yam.sif python /cmp/ref_noise.py A $P 2>&1 | grep -E '"policy"|Traceback|Error' | python3 -c "
import sys,json
for l in sys.stdin:
    if not l.startswith('{'): print(l.strip()[:200]); continue
    r=json.loads(l); print(r['policy'],r['seed'],r['kinds'],'cm',round(r['cm'],2),'lift',round(r['lift'],2),'rise',round(r['rise'],1),flush=True)"
done < $1
echo NOISE_DONE
