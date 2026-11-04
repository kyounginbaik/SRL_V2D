#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=O28-retup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/logs/retup-%j.log
# constant-offset LEFT-wrist retention CEM with terminal (tail) term (rackscan code/retention_upright.py, world-agnostic), rack protected.
# usage: WORLD=... sbatch ret_up.sbatch SRC_DIR OUT_DIR T0 T1 SCALE SEED [TAIL_S=1.5] [TAIL_W=1.0] [SIDE=left]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/code/env.sh
SRC=$1; OUT=$2; T0=$3; T1=$4; SC=$5; SEED=$6; TAIL=${7:-1.5}; TW=${8:-1.0}; SIDE=${9:-left}
cd $R/code && $PY -u $RS/code/retention_upright.py --source $SRC --episode-dir $EPDIR --output $OUT --start-s $T0 --stop-s $T1 --side $SIDE --target-object white_pot --tail-s $TAIL --protect-object plastic_dish_rack --seed $SEED --scale $SC --tail-weight $TW --samples 96 --iterations 12 --workers 16
$PY -c "import json;r=json.load(open('$OUT/result.json'));m=r['metrics'];print('RESULT $OUT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['official_style_reference_proxy']['objects'].items()}, 'tail', {k:r['best'].get(k) for k in ('tail_position_cm','tail_orientation_deg')})"
