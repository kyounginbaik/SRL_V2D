#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-pick
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/pickup-%A_%a.log
# physics_pickup_search_v2 CEM (full 8.6 s horizon, hold). usage: sbatch --array=.. pickup.sbatch SPECS_JSON OUT_BASE SEED ITER [resume]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SP=$1; O=$2; SEED=$3; IT=$4; RES=${5:-}
spec=$(python3 -c "import json,sys; print(json.load(open('$SP'))[int(sys.argv[1])])" "$SLURM_ARRAY_TASK_ID"); name=$(basename "$spec" .json)
X=""; [ -n "$RES" ] && [ -f $O/$name/progress.json ] && X="--resume $O/$name/progress.json"
[ -n "$RES" ] && [ -f $O/$name/progress.json ] && O=${O}_cont
cd $R/code && $PY -B -u physics_pickup_search_v2.py --spec "$spec" --output $O/$name --samples 64 --iterations $IT --workers 16 --seed $SEED $X
$PY -c "import json;r=json.load(open('$O/$name/result.json'));m=r['metrics']['official_style_reference_proxy'];print('RESULT $O/$name', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['objects'].items()})"
