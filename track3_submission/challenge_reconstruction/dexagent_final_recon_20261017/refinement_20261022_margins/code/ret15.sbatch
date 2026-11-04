#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=ret15old
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/logs/ret15-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
: "${T0:?}" "${T1:?}" "${SCALE:?}" "${SEED:?}" "${OUT:?}"
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_window32_44 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015 --output $OUT --start-s $T0 --stop-s $T1 --protect-object plastic_dish_rack --seed $SEED --scale $SCALE --samples 96 --iterations 12 --workers 16
$PY -B - <<PYEOF
import json,sys; r=json.load(open('$OUT/result.json')); m=r['metrics']['official_style_reference_proxy']['objects']; print('RESULT', r['metrics']['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m.items()})
PYEOF
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/spoon_actor_grid.py --episode 15 --source $OUT --output $OUT/actor_grid --workers 8 || true
