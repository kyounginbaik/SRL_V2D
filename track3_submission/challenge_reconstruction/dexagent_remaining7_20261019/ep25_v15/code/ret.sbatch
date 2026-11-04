#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=ep25-ret
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/logs/ret-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SRC:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${SIDE:?}" "${TGT:?}" "${PROT:?}" "${SEED:?}" "${SCALE:=0.75}" "${TAILW:=0.3}" "${TAILS:=1.5}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention_upright.py --source $SRC --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025 --output $OUT --start-s $T0 --stop-s $T1 --side $SIDE --target-object $TGT --protect-object $PROT --seed $SEED --scale $SCALE --tail-s $TAILS --tail-weight $TAILW --samples 96 --iterations 12 --workers 16
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python - <<PYEOF
import json; r=json.load(open('$OUT/result.json')); m=r['metrics']['official_style_reference_proxy']
print('RESULT', m['mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()})
PYEOF
