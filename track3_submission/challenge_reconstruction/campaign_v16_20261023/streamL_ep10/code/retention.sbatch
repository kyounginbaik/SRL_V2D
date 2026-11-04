#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=L10-retain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/logs/retain-%j.log
# usage: sbatch --export=ALL,SRC=...,OUT=...,T0=..,T1=..,SCALE=..,SEED=..,SIDE=..,PROTECT=..,ITER=12 retention.sbatch
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SRC:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${SCALE:?}" "${SEED:?}" "${SIDE:?}" "${PROTECT:?}"; ITER=${ITER:-12}
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10
PY=$R/.venv/bin/python; EPD=$S/world_v1/episode_000010
cd $R/code; export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
$PY -u $R/refinement_20261021_ep10/code/retention10.py --source $SRC --episode-dir $EPD --output $OUT --start-s $T0 --stop-s $T1 --protect-object $PROTECT --seed $SEED --scale $SCALE --samples 96 --iterations $ITER --workers 16 --side $SIDE
$PY $S/code/insertion_recon.py $OUT/execution/trajectory.npz
$PY - <<PYEOF
import json; r=json.load(open('$OUT/result.json')); print('RESULT', r['metrics']['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()})
PYEOF
