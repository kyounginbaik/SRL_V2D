#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --job-name=O28-ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/logs/ilc-%j.log
# reference-level carry ILC on the left wrist (refinement_20261021_ep38/code/ilc_carry_generic.py, world-agnostic through --episode-dir)
# usage: WORLD=... sbatch ilc.sbatch SRC_DIR OUT_DIR GAIN_ROT GAIN_POS [ITER=3] [T0=1.5] [T1=8.3] [TCLEAR=9.0]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamO_ep28/code/env.sh
SRC=$1; OUT=$2; GR=$3; GP=$4; IT=${5:-3}; T0=${6:-1.5}; T1=${7:-8.3}; TC=${8:-9.0}
cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $SRC --out $OUT --side left --object white_pot --t0 $T0 --t1 $T1 --t-clear $TC --episode-dir $EPDIR --iterations $IT --gain-rot $GR --gain-pos $GP
for d in $OUT/iter*; do $PY -c "import json;r=json.load(open('$d/result.json'));m=r['metrics'];print('RESULT $d', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1),round(v['max_position_error_m'][0]*100,1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"; done
