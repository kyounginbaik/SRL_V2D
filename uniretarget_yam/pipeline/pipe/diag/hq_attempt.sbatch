#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; V=$R/videos/scene_only; TMP=$H/.cache/tmp_hq_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4,NVIDIA_DRIVER_CAPABILITIES=all,MUJOCO_GL=egl,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",PYTHONPATH=/cmp/shim -B "$TMP:/tmp" -B "$H:$M" -B "$H/flash_chord:/workspace:ro" -B "$R/comparison:/cmp" -B "$V:/vid" -B /scratch2 --pwd /workspace "$R/flash_chord_yam.sif" python $M/code/render_hq.py --model $M/episodes/recon_000037/yam/yam_model.mjb --object 0 --run /vid/ep37/attempt --title "ep37 Grasp Search (open-loop)" --output /vid/ep37_Grasp_Search_Table_Only.mp4 2>&1 | grep -E 'video|Traceback|Error' | cut -c1-200
echo DONE
