#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch hq_scene.sbatch "37 25"   The presentation view of the Mink reference in the table-only scene.
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; V=$R/videos/scene_only; mkdir -p $V; TMP=$H/.cache/tmp_hq_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
run() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",$1 -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$R/comparison:/cmp" -B "$V:/vid" -B "$R/search:/search" -B $R/pipe/diag:/d -B /scratch2 --pwd /workspace "$R/flash_chord_yam.sif" "${@:2}"; }
for N in $1; do L=$(printf ep%02d $N); EP=$(printf recon_%06d $N); OBJ=$(python3 -c "import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];print(max(m,key=lambda x:x['move_cm'])['index'])")
 run X=1 python /d/fail_runs.py $N live 2>&1 | grep -viE 'warn' | tail -1 | cut -c1-200
 run PYTHONPATH=/cmp/shim python $M/code/render_hq.py --model $M/episodes/$EP/yam/yam_model.mjb --object $OBJ --run /vid/ep$N/mink --kinematic --title "$L Mink Reference" --output /vid/${L}_Mink_Reference_Table_Only.mp4 2>&1 | grep -E 'video|Traceback|Error' | cut -c1-200
done; echo HQ_DONE
