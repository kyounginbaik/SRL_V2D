#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --job-name=v9-rack14
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/logs/pipeline-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
$PY -u prepare_variant.py --episode 14 --source-root /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/collision_recon --workspace /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9 | tail -2
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/code/retarget_teacher.py 14 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000014/up_3.5_6.3_s0.75_w2.0 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/retarget/episode_000014
cp /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000014/up_3.5_6.3_s0.75_w2.0/reference.npz /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/retarget/episode_000014/reference.npz
$PY - <<PYEOF
import json,sys; r=json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/retarget/episode_000014/result.json')); print('TEACHER', r['metrics']['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in r['metrics']['official_style_reference_proxy']['objects'].items()}); sys.exit(0 if r['metrics']['reference_proxy_mp_sr']==1 else 3)
PYEOF
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/code/actor_grid.py --episode 14 --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/retarget/episode_000014 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep14_v9/retarget/episode_000014/actor_grid --workers 8
