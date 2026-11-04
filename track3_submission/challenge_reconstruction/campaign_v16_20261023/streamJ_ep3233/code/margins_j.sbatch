#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=J-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/margins-%j.log
# env: EP ACTOR OUT [SW=1.0 STD=0.04 INIT= EXTRA=]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${EP:?}" "${ACTOR:?}" "${OUT:?}"; cd /tmp
$PY -B -u $R/refinement_20261021_ep32/code/search_finger_margins_v13.py --episode $EP --actor $ACTOR --output $OUT --tolerance-deg 5 --spoon-weight ${SW:-1.0} --std ${STD:-0.04} --iterations 12 --samples 96 --workers 16 ${INIT:+--init $INIT} ${EXTRA:-}
python3 -c "import json;r=json.load(open('$OUT/evaluation/result.json'));m=r['metrics']['official_style_reference_proxy'];print('DONE mp_sr',m['mp_sr'],{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()})" || true
