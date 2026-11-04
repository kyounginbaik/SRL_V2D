#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-ret
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/ret-%j.log
# constant-offset retention CEM (rackscan code/retention.py, world-agnostic) over [T0,T1] s, rack protected.
# usage: WORLD=... sbatch ret.sbatch SRC_DIR OUT_DIR T0 T1 SCALE SEED [TOOL=retention|retention_upright] [SAMPLES=96] [ITER=12]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SRC=$1; OUT=$2; T0=$3; T1=$4; SC=$5; SEED=$6; TOOL=${7:-retention}; S=${8:-96}; IT=${9:-12}
cd $R/code && $PY -u $RS/code/$TOOL.py --source $SRC --episode-dir $EPDIR --output $OUT --start-s $T0 --stop-s $T1 --protect-object plastic_dish_rack --seed $SEED --scale $SC --samples $S --iterations $IT --workers 16
$PY -c "import json;r=json.load(open('$OUT/result.json'));m=r['metrics'];print('RESULT $OUT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
