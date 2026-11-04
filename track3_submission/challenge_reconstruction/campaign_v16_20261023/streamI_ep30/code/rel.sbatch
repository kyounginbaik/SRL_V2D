#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=I30-rel
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/logs/rel-%A_%a.log
# v9 release re-plan (rackscan ep30_v9/code/release_v9.py: wrist freeze at RELEASE, optional seat-lower, slow opening, up-then-lateral retreat)
# usage: WORLD=... sbatch --array=0-(N-1) rel.sbatch SRC_DIR OUT_BASE "REL OPEN LOWER MODE APERTURE" ["..." ...]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/code/env.sh
SRC=$1; O=$2; shift 2; A=("$@"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $RS/ep30_v9/code/release_v9.py $SRC $EPDIR $O/rel_$1_o$2_l$3_$4$5 $1 $2 $3 0.06 0.06 $4 $5
