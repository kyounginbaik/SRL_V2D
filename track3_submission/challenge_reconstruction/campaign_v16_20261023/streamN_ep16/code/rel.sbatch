#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=N16-rel
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/rel-%A_%a.log
# release re-plan (code/release_v9_spoon.py). usage: WORLD=... sbatch --array=.. rel.sbatch SRC OUT_BASE "REL OPEN LOWER UP LAT MODE APERTURE" ...
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SRC=$1; O=$2; shift 2; A=("$@"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -u $N/code/release_v9_spoon.py $SRC $EPDIR $O/rel_$1_o$2_l$3_u$4_$6$7 $1 $2 $3 $4 $5 $6 $7
