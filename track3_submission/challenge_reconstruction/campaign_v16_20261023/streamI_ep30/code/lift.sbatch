#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=I30-lift
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/logs/lift-%A_%a.log
# extra right-wrist lift (rackscan ep30_v8/code/extra_lift.py; EPDIR env points it at the Stream-I world)
# usage: WORLD=... sbatch --array=0-(N-1) lift.sbatch SRC_DIR OUT_BASE "HEIGHT START END" ["..." ...]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/code/env.sh
SRC=$1; O=$2; shift 2; A=("$@"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $RS/ep30_v8/code/extra_lift.py $SRC $O/lift$1_$2_$3 $1 $2 $3
