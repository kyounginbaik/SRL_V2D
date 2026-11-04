#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=M14-rel
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/logs/rel-%j.log
# cup release re-plan (R7 code/release_refined_grasp.py, the tool that made the pre-scan teacher release_neutral/trial2):
# opens the retained left grasp at START s with the given opening origin and retreats; 4 trials (trial0-3) with full-episode verification.
# usage: WORLD=... sbatch rel.sbatch SRC_DIR OUT_DIR START [ORIGIN=neutral|body|pinch]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/code/env.sh
SRC=$1; OUT=$2; START=$3; ORIGIN=${4:-neutral}
cd /tmp && $PY -u $R7/code/release_refined_grasp.py --source $SRC --episode-dir $EPDIR --output $OUT --side left --object blue_cup --start $START --opening-origin $ORIGIN --workers 4
$PY - <<PYEOF
import json,glob
for p in sorted(glob.glob('$OUT/trial*/result.json')):
    r=json.load(open(p)); m=r['metrics']['official_style_reference_proxy']['objects']
    print('RESULT',p,r['metrics']['reference_proxy_mp_sr'],{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m.items()})
PYEOF
