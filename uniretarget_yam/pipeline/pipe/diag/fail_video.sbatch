#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/diag/fv-%x-%j.log
set -euo pipefail
N=$1; MODE=${2:-}; R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; if [ "$MODE" = live ]; then E=$M/episodes/recon_$(printf %06d $N); V=$R/videos/failed_pilots; else E=$M/episodes/.fail_recon_$(printf %06d $N); V=$R/videos/failed; fi; mkdir -p $V; TMP=$H/.cache/tmp_fv_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
run() { local envs=$1; shift; apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",$envs -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$R/comparison:/cmp" -B "$V:/vid" -B "$R/search:/search" -B $R/pipe/diag:/d --pwd /workspace "$R/flash_chord_yam.sif" "$@"; }
run X=1 python /d/fail_runs.py $N $MODE 2>&1 | grep -viE 'warn' | tail -3 | cut -c1-300
read OBJ ATT FRONT CAUSE LIFT <<<$(python3 -c "
import json;d=json.load(open('$V/ep$N/info.json'));print(d['obj'],int(d['attempt']),d.get('frontier',-1),d.get('cause',-1),d.get('lift_mm',0))")
hq() { run PYTHONPATH=/cmp/shim python $M/code/render_hq.py --model $E/yam/yam_model.mjb --object $OBJ "$@"; }
hq --run /vid/ep$N/mink --kinematic --title "ep$N Mink Reference " --subtitle "Robot at Mink joint targets, object on the reconstructed path" --output /vid/ep${N}_Mink_Reference.mp4
[ "$ATT" = 1 ] && hq --run /vid/ep$N/attempt --cause $CAUSE --title "ep$N Best Grasp Search Attempt (failed)" --subtitle "Open-loop MuJoCo replay; would be stopped at frame $FRONT, peak rise ${LIFT} mm" --output /vid/ep${N}_Best_Grasp_Attempt.mp4
echo FV_DONE
