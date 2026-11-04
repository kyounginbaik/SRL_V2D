#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=ep10-retain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/logs/retain-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SRC:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${SCALE:?}" "${SEED:?}" "${SIDE:?}" "${PROTECT:?}"
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/code/retention10.py --source $SRC --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/episode_dir/episode_000010 --output $OUT --start-s $T0 --stop-s $T1 --protect-object $PROTECT --seed $SEED --scale $SCALE --samples 96 --iterations 12 --workers 16 --side $SIDE
$PY /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/code/insertion_of.py $OUT/execution/trajectory.npz
$PY - <<PYEOF
import json,sys; r=json.load(open('$OUT/result.json')); print('RESULT', r['metrics']['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()}); sys.exit(0 if r['metrics']['reference_proxy_mp_sr']==1 else 3)
PYEOF
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/code/actor_grid10.py --episode 10 --source $OUT --output $OUT/actor_grid --workers 8
