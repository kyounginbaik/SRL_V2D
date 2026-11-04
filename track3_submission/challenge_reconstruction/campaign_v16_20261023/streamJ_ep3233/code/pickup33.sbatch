#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=J33-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/pickup33-%j.log
# env: SPEC OUT [SEED]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${SPEC:?}" "${OUT:?}"; cd /tmp
$PY -B -u $R/code/physics_pickup_search_v2.py --spec $SPEC --output $OUT --samples 96 --iterations 12 --workers 16 --seed ${SEED:-44}
python3 -c "import json;r=json.load(open('$OUT/result.json'));print('DONE',{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()})"
