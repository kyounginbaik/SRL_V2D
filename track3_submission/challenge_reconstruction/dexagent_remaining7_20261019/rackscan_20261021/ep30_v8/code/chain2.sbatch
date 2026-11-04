#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=ep30v8-c2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/logs/chain2-%j.log
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EPDIR=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/episodes/episode_000030 RACKW=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /data1/SRL_V2D
: "${BASE_RUN:?}"
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/release_refined_grasp_sweeper_snapshot.py --source $BASE_RUN --episode-dir $EPDIR --output $BASE_RUN/release_neutral --side right --object mini_sweeper --start 12.3 --opening-origin neutral --workers 8
T=$($PY /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/code/pick_best.py $BASE_RUN/release_neutral); echo "BEST_RELEASE $T"
[ -f $T/reference.npz ] || exit 3
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
for spec in "3.2 12.0 0.75 101" "4.0 11.5 1.0 102"; do set -- $spec; O=$T/ret_$1_$2_s$3
  [ -f $O/result.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention.py --source $T --episode-dir $EPDIR --output $O --start-s $1 --stop-s $2 --protect-object plastic_dish_rack --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
  $PY -c "import json;r=json.load(open('$O/result.json'));m=r['metrics'];print('RESULT $O', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
  $PY -c "import json,sys;sys.exit(0 if json.load(open('$O/result.json'))['metrics']['reference_proxy_mp_sr']==1 else 3)" && $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/code/actor_grid.py --episode 30 --source $O --output $O/actor_grid --workers 16
done
echo CHAIN2_DONE
