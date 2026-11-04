#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J v38 video.sbatch N [seed]   The three videos, with the best scored checkpoint for the bilevel run.
set -euo pipefail
N=$1; SEED=${2:-42}; EP=$(printf recon_%06d $N); L=$(printf ep%02d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; E=$M/episodes/$EP; V=$R/videos/$EP${VT:-}; mkdir -p $V; TMP=$H/.cache/tmp_v_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$H/episodes/$EP/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
read UCK UM <<<$(python3 -c "
import json;r=[x for x in json.load(open('$R/comparison/results/upper_test_${RN:-$EP}_s$SEED.json'))];b={}
for x in r:b.setdefault(x['steps'],[]).append(x)
k=max(b,key=lambda s:(min(y['passed'] for y in b[s]),-sum(y['cm'] for y in b[s])));print(k,round(k/1e6))")
ACK=0; [ -n "${UONLY:-}" ] || ACK=$(ls $H/out/$EP/A${AT:-}-seed$SEED | grep policy_ | sed 's/policy_//;s/.safetensors//' | sort -n | tail -1); AM=$((ACK/1000000)); echo "U $UCK A $ACK side $SIDE obj $IDX"
run() { local envs=$1; shift; apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",RECON_EP=$E,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,OBJ_X=$IDX,$envs -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B "$R/comparison:/cmp" -B "$V:/vid" -B "$R/search:/search" --pwd /workspace "$R/flash_chord_yam.sif" "$@"; }
[ -n "${UONLY:-}" ] || run CLIP=mA,CKPT=policy_$ACK.safetensors python /cmp/record_clip2.py ${RN:-$EP}A 2 $M/out/$EP/A${AT:-}-seed$SEED
run CLIP=mU,CKPT=policy_$UCK.safetensors python /cmp/record_clip2.py ${RN:-$EP}U 2 $M/out/$EP/U${UT:-}-seed$SEED
run X=1 python /cmp/make_runs.py $([ -n "${UONLY:-}" ] && echo mU || echo mA,mU) mU
hq() { run PYTHONPATH=/cmp/shim python $M/code/render_hq.py --model $E/yam/yam_model.mjb "$@"; }
[ -n "${UONLY:-}" ] || hq --run /vid/run_mA --title "$L Fixed Reference: Low Level Only${VT:+ (VOC)}" --subtitle "FlashSAC on the Mink reference, ${AM}M steps" --output /vid/${L}_Fixed_Reference_Low_Level.mp4
hq --run /vid/run_mU --title "$L UniRetarget: Physics Result${VT:+ (VOC)}" --subtitle "Bilevel from the Mink reference, policy at ${UM}M steps" --output /vid/${L}_UniRetarget_Physics_Result.mp4
hq --run /vid/run_refinement --kinematic --ghost /vid/run_refinement/ghost.npz --ghost_side ${SIDE}_ --title "$L UniRetarget: Reference Refinement${VT:+ (VOC)}" --subtitle "Blue: Mink reference. Solid: the reference bilevel used at ${UM}M steps" --output /vid/${L}_UniRetarget_Reference_Refinement.mp4
[ -n "${VT:-}" ] || SRC=$R/source_videos/ego_cam_a bash $R/pipe/diag/ego.sbatch "$N" || echo EGO_FAILED
echo VIDEO_DONE
