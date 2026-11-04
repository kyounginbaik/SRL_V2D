#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=G30-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep30_chain-%j.log
# usage: sbatch ep30_chain.sbatch PICKUP_DIR OUT_BASE   (extra lift -> neutral release -> best trial -> retention 3.2-12.0 -> v9 release re-plan)
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
RS=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021
EPDIR=$RS/ep30_v8/episodes/episode_000030
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code RACKW=$RS/ep30_v8
P=$1; O=$2; mkdir -p $O; cd /tmp
[ -f $O/lift0.06/result.json ] || $PY -u $RS/ep30_v8/code/extra_lift.py $P $O/lift0.06 0.06 2.5 3.5
[ -f $O/lift0.06/release_neutral/summary.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/release_refined_grasp_sweeper_snapshot.py --source $O/lift0.06 --episode-dir $EPDIR --output $O/lift0.06/release_neutral --side right --object mini_sweeper --start 12.3 --opening-origin neutral --workers 4
T=$($PY $RS/ep30_v8/code/pick_best.py $O/lift0.06/release_neutral); echo "BEST_RELEASE $T"; [ -f $T/reference.npz ] || exit 3
cd $R/code
for spec in "3.2 12.0 0.75 101" "2.0 12.0 0.75 103"; do set -- $spec; RT=$T/ret_$1_$2_s$3
  [ -f $RT/result.json ] || $PY -u $RS/code/retention.py --source $T --episode-dir $EPDIR --output $RT --start-s $1 --stop-s $2 --protect-object plastic_dish_rack --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
  $PY -c "import json;r=json.load(open('$RT/result.json'));m=r['metrics'];print('RESULT $RT', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
  for rel in "11.6 1.2 0" "11.8 1.2 0"; do set -- $rel; RD=$RT/rel2_$1_o$2_l$3_body0.01
    [ -f $RD/result.json ] || $PY -u $RS/ep30_v9/code/release_v9.py $RT $EPDIR $RD $1 $2 $3 0.06 0.06 body 0.01
  done
done
echo CHAIN_DONE
