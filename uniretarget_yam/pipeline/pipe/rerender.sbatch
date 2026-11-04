#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
set -euo pipefail
UM=0; AM=0; N=$1; SEED=${2:-42}; EP=$(printf recon_%06d $N); L=$(printf ep%02d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/$EP; V=$R/videos/$EP; mkdir -p $V; TMP=$H/.cache/tmp_v_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
run() { local envs=$1; shift; apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",RECON_EP=$E,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,OBJ_X=$IDX,$envs -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B "$R/comparison:/cmp" -B "$V:/vid" -B "$R/search:/search" --pwd /workspace "$R/flash_chord_yam.sif" "$@"; }
hq() { run PYTHONPATH=/cmp/shim python $M/code/render_hq.py --model $E/yam/yam_model.mjb "$@"; }
hq --run /vid/run_mA --title "$L Fixed Reference: Low Level Only" --subtitle "FlashSAC on the Mink reference, ${AM}M steps" --output /vid/${L}_Fixed_Reference_Low_Level.mp4
hq --run /vid/run_mU --title "$L UniRetarget: Physics Result" --subtitle "Bilevel from the Mink reference, policy at ${UM}M steps" --output /vid/${L}_UniRetarget_Physics_Result.mp4
hq --run /vid/run_refinement --kinematic --ghost /vid/run_refinement/ghost.npz --ghost_side ${SIDE}_ --title "$L UniRetarget: Reference Refinement" --subtitle "Blue: Mink reference. Solid: the reference bilevel used at ${UM}M steps" --output /vid/${L}_UniRetarget_Reference_Refinement.mp4
echo RERENDER_DONE
