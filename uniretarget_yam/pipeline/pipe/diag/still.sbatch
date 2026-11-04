#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam
apptainer exec --nv --containall --no-home --home /tmp --env MUJOCO_GL=egl -B "$H:/uniretarget_yam" -B /scratch2 --pwd /uniretarget_yam/code $R/flash_chord_yam.sif python $R/pipe/diag/scene_still.py $1
