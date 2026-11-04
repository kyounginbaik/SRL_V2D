#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=D9-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/logs/svst-%j.log
# usage: sbatch savestage.sbatch EP TABLE.npz SAVE_OUT STAGING_BASE "note"   (save.sbatch body, then stage.sbatch body)
set -euo pipefail
D9=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight
bash $D9/code/save.sbatch $1 $2 $3
bash $D9/code/stage.sbatch $1 $3/actor $4 "$5"
