#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=ep15v8-prep
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/logs/pipeline-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
$PY -u prepare_variant.py --episode 15 --source-root /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/collision_recon --workspace /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8 | tail -2
$PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/code/retarget_teacher.py 15 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_window32_44 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/retarget/episode_000015 || true
$PY -c "import json,shutil;r=json.load(open('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/retarget/episode_000015/result.json'));shutil.copy(r['reference'],'/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/retarget/episode_000015/reference.npz')"
for spec in "2.8 4.6 0.5 101" "3.0 5.5 0.5 102" "3.2 4.4 0.75 103" "3.0 8.0 0.75 104" "3.3 6.0 1.0 105"; do set -- $spec
  EP=15 SRC=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/retarget/episode_000015 OUT=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/search/episode_000015/ret_$1_$2_s$3 T0=$1 T1=$2 SCALE=$3 SEED=$4 sbatch --export=ALL --job-name=v8r15_$1 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/code/retention.sbatch; done
