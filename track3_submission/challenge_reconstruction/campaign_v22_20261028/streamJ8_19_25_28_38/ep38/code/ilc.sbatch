#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J8e38-ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/logs/ilc-%A_%a.log
# usage: sbatch --array=0-N ilc.sbatch LISTFILE  (each line: OUT_NAME SOURCE_DIR [ilc_carry_generic args...]); world = J8 ep38 target world
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep38/code/env.sh
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $L; N=$1; S=$2; shift 2
cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $S --out $SD/ilc/$N --side right --object mini_dust_pan --episode-dir $SD/world0/episode_target/episode_000038 "$@" 2>&1 | grep -v Warn
