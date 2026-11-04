#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=R32-zero
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/zero-%A_%a.log
# usage: sbatch --array=0-N zero_eval.sbatch SPECS_JSON_LIST OUTROOT [PARAMS]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh; cd $R/code
spec=$(python3 -c "import json,sys; print(json.load(open('$1'))[int(sys.argv[1])])" $SLURM_ARRAY_TASK_ID); name=$(basename $spec .json)
[ -f $2/$name/result.json ] || $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/code/zero_eval32.py $spec $2/$name ${3:-}
