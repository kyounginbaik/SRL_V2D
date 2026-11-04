#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=I30-pick
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/logs/pickup-%j.log
# physics_pickup_search_v2 CEM in the Stream-I world. usage: sbatch pickup.sbatch SPEC OUT SEED [RESUME_PROGRESS_JSON]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/code/env.sh
SPEC=$1; OUT=$2; SEED=$3; RES=${4:-}
cd /tmp; if [ -n "$RES" ]; then X="--resume $RES"; else X=""; fi
$PY -B -u $R/code/physics_pickup_search_v2.py --spec $SPEC --output $OUT --workers 16 --samples 48 --iterations 12 --seed $SEED $X
$PY -c "import json;r=json.load(open('$OUT/result.json'));m=r['metrics'];print('RESULT $OUT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
