#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --job-name=rack-retarget
#SBATCH --array=1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/pipeline-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
EPS=(14 15 16 28 30)
TEACHERS=(/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/cup_transfer/release_neutral/trial2 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_window32_44 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000016/retention_after38 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/pot_transfer/pinch_roll180 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1/release_neutral/trial1)
E=${EPS[$SLURM_ARRAY_TASK_ID]}; T=${TEACHERS[$SLURM_ARRAY_TASK_ID]}; N=$(printf episode_%06d $E)
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
$PY -u prepare_variant.py --episode $E --source-root /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/collision_recon --workspace /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021 | tail -3
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retarget_teacher.py $E $T /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/$N | tee /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/$N.summary.json 2>/dev/null || $PY -c "pass"
$PY - <<PYEOF
import json,sys; r=json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/'+'$N'+'/result.json')); sys.exit(0 if r['metrics']['reference_proxy_mp_sr']==1 else 3)
PYEOF
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/actor_grid.py --episode $E --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/$N --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/$N/actor_grid --workers 4
