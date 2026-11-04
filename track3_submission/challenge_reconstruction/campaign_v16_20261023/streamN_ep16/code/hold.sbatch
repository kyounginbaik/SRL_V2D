#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=N16-hold
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/hold-%A_%a.log
# hold-until-end re-plan (streamE hold_variant.py: right wrist+fingers frozen from T_HOLD, following the static spoon target)
# usage: WORLD=... sbatch --array=0-(N-1) hold.sbatch SRC_DIR OUT_BASE T_HOLD [T_HOLD ...]   (ramp 0.2 s)
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SRC=$1; O=$2; shift 2; A=("$@"); TH=${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $N/code/hold_variant.py --teacher $SRC --episode-dir $EPDIR --out $O/hold_$TH --side right --object wooden_spoon --t-hold $TH --ramp 0.2
