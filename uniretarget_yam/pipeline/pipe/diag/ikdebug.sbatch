#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch ikdebug.sbatch N   Reruns the Mink IK into yam/flash/mink_debug and saves targets and robot keypoints.
N=$1; EP=$(printf recon_%06d $N); W=/scratch2/exaFLOPs/latest_recon/run37; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; E=$H/episodes/$EP; TMP=$H/.cache/tmp_ik_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$E/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
cd $W; apptainer exec --containall --no-home --home /tmp --env OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$W/rg/source/robotic_grounding:$W/rg/lib:$W/pydeps,OMNI_KIT_ACCEPT_EULA=YES,IK_PALM_ORI=${IK_PALM_ORI:-0.2},IK_PALM_POS=${IK_PALM_POS:-0.2},IK_TIP_ORI=${IK_TIP_ORI:--1} -B /scratch2 -B $TMP:/tmp --pwd $W /lustre/exaflops/singularity_exports/robotic-grounding-mink-1.3.0.sif /isaac-sim/python.sh yam_mink.py --bundle ${BUNDLE:-$E/bundle} --flash $E/yam/flash --source source --side $SIDE --object_index $IDX --output $E/yam/flash/mink_debug${TAG:-} 2>&1 | grep -E 'hand_repair|Traceback|Error|wrist cm' | tail -6 | cut -c1-240
ls -la $E/yam/flash/mink_debug${TAG:-}/ik_debug.npz; echo IK_DONE
