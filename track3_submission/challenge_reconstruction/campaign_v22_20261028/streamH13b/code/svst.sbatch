#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H13b-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/logs/svst-%j.log
# usage: sbatch svst.sbatch EP ACTOR TABLE SAVE_OUT STAGING_BASE "note" TAG   (save.sbatch body, stageH.sbatch body, verify.sbatch body)
set -euo pipefail
H13=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b
bash $H13/code/save.sbatch $1 $2 $3 $4
OBJ='{}'   # object errors json is used only by eps 15/16 (mkbaseH); not used here
bash $H13/code/stageH.sbatch $1 $4/actor $5 "$6" "$OBJ"
bash $H13/code/verify.sbatch $1 $5/components/$(printf episode_%06d $1) $7
