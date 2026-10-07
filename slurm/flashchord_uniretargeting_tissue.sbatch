#!/bin/bash
# Matched FlashSAC scheduled-VOC tissue-box study: original smooth editor vs
# UniRetargeting visual-penetration-aware smooth editor, seeds 42/123/456.
# Submit only after the frozen-policy pilot and short training smoke pass.
#SBATCH --job-name=fc-uniretarget
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5%3
#SBATCH --output=/data1/SRL_V2D/slurm_logs/fc_uniretarget_%A_%a.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/fc_uniretarget_%A_%a.err
set -euo pipefail

export EDITOR_STUDY=1
export VARIANT_OVERRIDE=smooth
export STUDY=full
export ENGINE=ttsa_flashsac
export FULL_INTERVAL=3276800
export LOG_PROJECT=v2d-tissue-uniretargeting
export SELECT_BEST_TRIAL=1

if (( SLURM_ARRAY_TASK_ID < 3 )); then
    export PENETRATION_AWARE=0
    export FULL_GROUP=uniretargeting_control_20261007
else
    export PENETRATION_AWARE=1
    export PENETRATION_WEIGHT=1.0
    export PENETRATION_STRIDE=8
    export GUIDED_PENETRATION_PROPOSAL=1
    export FULL_GROUP=uniretargeting_visual_20261007
fi

exec bash /data1/SRL_V2D/slurm/flashchord_tissue_bilevel.sbatch
