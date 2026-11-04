#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=J-rsc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/rsc-%j.log
# env: SRC EPD OUT T0 T1 SEED [W=1.0 SCALE=0.5 SIDE=right PROT=blue_cup CUPMESH SPOONMESH FSCALE]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
: "${SRC:?}" "${EPD:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${SEED:?}"
cd $R/code
[ -f $OUT/result.json ] || $PY -u $R/refinement_20261021_ep32/code/retention_spooncup.py --source $SRC --episode-dir $EPD --output $OUT --start-s $T0 --stop-s $T1 --side ${SIDE:-right} --protect-object ${PROT:-blue_cup} --cup-weight ${W:-1.0} --seed $SEED --scale ${SCALE:-0.5} --spoon-mesh ${SPOONMESH:-$FR/episode_000032/assets/wooden_spoon.obj} --cup-mesh ${CUPMESH:-$FR/episode_000032/assets/blue_cup.obj} --samples 96 --iterations 12 --workers 16 ${FSCALE:+--finger-scale $FSCALE}
python3 -c "import json;r=json.load(open('$OUT/result.json'));print('DONE',{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()},'cup_err',round(r['best'].get('spoon_in_cup_error_cm',-1),2))"
