#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=ep30v8-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/logs/chain-%j.log
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EPDIR=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/episodes/episode_000030 RACKW=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /data1/SRL_V2D
SEED=${SEED:-44}; P=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/search/pickup_s$SEED
[ -f $P/result.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/physics_pickup_search_v2.py --spec /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/search/pickup_seed/spec.json --output $P --workers 16 --samples 48 --iterations 12 --seed $SEED
for h in 0.08 0.06 0.10; do [ -f ${P}_lift$h/result.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/code/extra_lift.py $P ${P}_lift$h $h 2.5 3.5; done
for h in 0.08 0.06 0.10; do $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/release_refined_grasp_sweeper_snapshot.py --source ${P}_lift$h --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/episodes/episode_000030 --output ${P}_lift$h/release_neutral --side right --object mini_sweeper --start 12.3 --opening-origin neutral --workers 8 || echo "release $h failed"; done
echo CHAIN_DONE
