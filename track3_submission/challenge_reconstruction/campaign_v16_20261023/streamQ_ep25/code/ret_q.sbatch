#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=Q25-ret
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25/logs/ret-%j.log
# retention_upright CEM (unchanged rackscan tool) in the Stream-Q corrected world. env: WORLD SRC OUT T0 T1 SIDE TGT PROT SEED [SCALE TAILW TAILS SAMPLES ITERS]
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${WORLD:?}" "${SRC:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${SIDE:?}" "${TGT:?}" "${PROT:?}" "${SEED:?}" "${SCALE:=0.75}" "${TAILW:=1.0}" "${TAILS:=1.5}" "${SAMPLES:=96}" "${ITERS:=12}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention_upright.py --source $SRC --episode-dir $WORLD/episodes/episode_000025 --output $OUT --start-s $T0 --stop-s $T1 --side $SIDE --target-object $TGT --protect-object $PROT --seed $SEED --scale $SCALE --tail-s $TAILS --tail-weight $TAILW --samples $SAMPLES --iterations $ITERS --workers 16
$PY - <<PYEOF
import json; r=json.load(open('$OUT/result.json')); m=r['metrics']['official_style_reference_proxy']
print('RESULT', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],3),round(v['mean_orientation_error_degrees'][0],2),round(v['max_position_error_m'][0]*100,1)) for k,v in m['objects'].items()})
PYEOF
