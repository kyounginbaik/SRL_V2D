#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch scene_video.sbatch "37 8"   Mink reference in the table-only scene, beside the source video.
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; B=/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103; V=$R/videos/scene_only; mkdir -p $V; TMP=$H/.cache/tmp_sv_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
run() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$R/comparison:/cmp" -B "$V:/vid" -B "$R/search:/search" -B $R/pipe/diag:/d -B /scratch2 --pwd "${PD:-/workspace}" "$R/flash_chord_yam.sif" "$@"; }
for N in $1; do L=$(printf ep%02d $N); EP=$(printf recon_%06d $N); Z=$(printf episode_%06d $N)
 run python /d/fail_runs.py $N live 2>&1 | grep -viE 'warn' | tail -2 | cut -c1-240
 PD=$M/code run python $M/code/render_ego.py --episode $M/episodes/$EP --bundle $B/bundles/$Z --run /vid/ep$N/mink --title "$L Mink reference, table only" --output /vid/${L}_Source_vs_Mink_Table_Only.mp4 --source $R/source_videos/ego_cam_a/$Z.mp4 2>&1 | grep -E 'RENDERED|fit_cm|Traceback|Error' | cut -c1-220
done; echo SCENE_DONE
