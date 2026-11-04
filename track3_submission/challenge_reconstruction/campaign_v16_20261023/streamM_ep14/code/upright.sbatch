#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=M14-up
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/logs/upright-%j.log
# upright retention CEM (rackscan code/retention_upright.py, world-agnostic): one-hand smooth constant offset in [T0,T1] s + tail term
# on the cup's final TAIL_S seconds; rack protected. usage: WORLD=... sbatch upright.sbatch SRC_DIR OUT_DIR T0 T1 SCALE TAIL_WEIGHT SEED [SIDE=left] [TAIL_S=1.5]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/code/env.sh
SRC=$1; OUT=$2; T0=$3; T1=$4; SC=$5; TW=$6; SEED=$7; SIDE=${8:-left}; TS=${9:-1.5}
cd $R/code && $PY -u $RS/code/retention_upright.py --source $SRC --episode-dir $EPDIR --output $OUT --start-s $T0 --stop-s $T1 --side $SIDE --target-object blue_cup --protect-object plastic_dish_rack --seed $SEED --scale $SC --tail-weight $TW --tail-s $TS --samples 96 --iterations 12 --workers 16
$PY -c "import json;r=json.load(open('$OUT/result.json'));m=r['metrics'];print('RESULT $OUT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['official_style_reference_proxy']['objects'].items()}, 'tail', {k:r['best'].get(k) for k in ('tail_position_cm','tail_orientation_deg')})"
