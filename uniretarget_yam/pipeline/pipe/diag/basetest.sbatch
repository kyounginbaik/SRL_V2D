#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch basetest.sbatch N UP X   IK in a copy of the episode with the arm bases raised by UP and moved along the table by X.
N=$1; UP=$2; X=$3; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; W=/scratch2/exaFLOPs/latest_recon/run37; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; T=.base_${N}_${UP}_${X}; E=$H/episodes/$T; rm -rf $E; mkdir -p $E; cp -r $H/episodes/$EP/{baseline,bundle,native_model.mjb,center.json,up.json,recon_meta.json,other_bundles.txt} $E/; for b in $H/episodes/$EP/bundle_*; do [ -e $b ] && cp -r $b $E/; done; mkdir -p $E/yam/flash; cp -r $H/episodes/$EP/yam/flash/source $E/yam/flash/; TMP=$H/.cache/tmp_b_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
ln -sfn $H/objects/$EP $H/objects/$T; apptainer exec --containall --no-home --home /tmp --env YAM_BASE_UP=$UP,YAM_BASE_X=$X,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=2 -B "$TMP:/tmp" -B "$H:/uniretarget_yam" -B /scratch2 --pwd /uniretarget_yam/code $R/flash_chord_yam.sif python -c "
import sys;from pathlib import Path;import yam_scene,yam_flash_assets
e=Path('/uniretarget_yam/episodes/$T');yam_scene.build(e,e/'yam');yam_flash_assets.build(e,e/'yam/flash/yam_sharpa_bimanual.urdf');print('scene built')" 2>&1 | tail -2 | cut -c1-200
read IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$E/recon_meta.json'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['index'],m['side'])")
cd $W; apptainer exec --containall --no-home --home /tmp --env OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$W/rg/source/robotic_grounding:$W/rg/lib:$W/pydeps,OMNI_KIT_ACCEPT_EULA=YES -B /scratch2 -B $TMP:/tmp --pwd $W /lustre/exaflops/singularity_exports/robotic-grounding-mink-1.3.0.sif /isaac-sim/python.sh yam_mink.py --bundle $E/bundle --flash $E/yam/flash --source source --side $SIDE --object_index $IDX --output $E/yam/flash/mink 2>&1 | grep -E 'Traceback|Error' | tail -2 | cut -c1-220
ls $E/yam/flash/mink/ik_debug.npz; echo BASE_DONE
