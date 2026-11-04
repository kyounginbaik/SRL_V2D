#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=E-ret15
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/ret15-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15
G=("1.9 3.6 0.5 201" "1.9 3.6 0.75 202" "2.1 4.0 0.5 203" "2.3 4.6 0.5 204" "1.6 3.4 0.5 205" "2.0 4.8 0.75 206"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
OUT=$W/retention/ret_$1_$2_s$3
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/code/retention.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_window32_44 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/episodes/episode_000015 --output $OUT --start-s $1 --stop-s $2 --protect-object plastic_dish_rack --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
$PY - <<PYEOF
import json; r=json.load(open('$OUT/result.json')); print('RESULT', r['metrics']['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()})
PYEOF
