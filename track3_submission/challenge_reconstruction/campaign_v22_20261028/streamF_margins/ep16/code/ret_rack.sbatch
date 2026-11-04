#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=F16-retr
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/logs/F16-retr-%j.log
# rack-weighted retention (streamN code/retention_rack.py, unchanged) in world_T. usage: sbatch ret_rack.sbatch SRC OUT T0 T1 SCALE SEED RACK_W [ITERS]
WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/world_T
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SRC=$1; OUT=$2; T0=$3; T1=$4; SC=$5; SEED=$6; RW=$7; IT=${8:-12}
cd $R/code && PYTHONPATH=$PYTHONPATH:$RS/code $PY -u $N/code/retention_rack.py --source $SRC --episode-dir $EPDIR --output $OUT --start-s $T0 --stop-s $T1 --protect-object plastic_dish_rack --seed $SEED --scale $SC --samples 96 --iterations $IT --workers 16 --rack-weight $RW
$PY -c "import json;r=json.load(open('$OUT/result.json'));m=r['metrics'];print('RESULT $OUT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
