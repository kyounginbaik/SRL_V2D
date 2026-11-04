#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; V=$R/videos/.parts16_recon_000038; E=$M/episodes/.parts16_recon_000038; TMP=$H/.cache/tmp_v_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
hq() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",PYTHONPATH=/cmp/shim -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$R/comparison:/cmp" -B "$V:/vid" -B /scratch2 --pwd /workspace $R/flash_chord_yam.sif python $M/code/render_hq.py --model $E/yam/yam_model.mjb "$@" 2>&1 | grep -E '"video"|Traceback|Error' | cut -c1-160; }
hq --run /vid/run_mA --title "ep38 Fixed Reference: Low Level Only" --output /vid/ep38_Fixed_Reference_Low_Level.mp4
hq --run /vid/run_mU --title "ep38 UniRetarget: Physics Result" --output /vid/ep38_UniRetarget_Physics_Result.mp4
hq --run /vid/run_refinement --kinematic --ghost /vid/run_refinement/ghost.npz --ghost_side right_ --title "ep38 UniRetarget: Reference Refinement" --output /vid/ep38_UniRetarget_Reference_Refinement.mp4
echo RENDER_DONE
