#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=I30-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/logs/chain-%j.log
# pickup -> extra lift H at 2.5-3.5 s -> neutral release at 12.3 s (4 trials) -> best trial -> retention T0-12.0 (scale 0.75) -> v9 release re-plans
# usage: WORLD=... sbatch chain.sbatch PICKUP_DIR OUT_BASE LIFT_H RET_T0 [RET_SEED]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30/code/env.sh
set -uo pipefail
P=$1; O=$2; H=$3; T0=$4; SEED=${5:-101}; mkdir -p $O; cd /tmp
[ -f $O/lift$H/result.json ] || $PY -u $RS/ep30_v8/code/extra_lift.py $P $O/lift$H $H 2.5 3.5
[ -f $O/lift$H/release_neutral/summary.json ] || $PY -u $R7/code/release_refined_grasp_sweeper_snapshot.py --source $O/lift$H --episode-dir $EPDIR --output $O/lift$H/release_neutral --side right --object mini_sweeper --start 12.3 --opening-origin neutral --workers 4
T=$($PY $RS/ep30_v8/code/pick_best.py $O/lift$H/release_neutral); echo "BEST_RELEASE $T"; [ -f $T/reference.npz ] || exit 3
cd $R/code; RT=$T/ret_${T0}_12.0_s0.75
[ -f $RT/result.json ] || $PY -u $RS/code/retention.py --source $T --episode-dir $EPDIR --output $RT --start-s $T0 --stop-s 12.0 --protect-object plastic_dish_rack --seed $SEED --scale 0.75 --samples 96 --iterations 12 --workers 16
$PY -c "import json;r=json.load(open('$RT/result.json'));m=r['metrics'];print('RESULT $RT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
cd /tmp; for rel in "11.6 1.2 0" "11.3 1.2 0.005"; do set -- $rel; RD=$RT/rel2_$1_o$2_l$3_body0.01
  [ -f $RD/result.json ] || $PY -u $RS/ep30_v9/code/release_v9.py $RT $EPDIR $RD $1 $2 $3 0.06 0.06 body 0.01
done
echo CHAIN_DONE
