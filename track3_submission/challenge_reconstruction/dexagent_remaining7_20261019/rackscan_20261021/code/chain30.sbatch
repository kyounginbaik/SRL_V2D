#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=rack30-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/chain30-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code
cd /data1/SRL_V2D
[ -f /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup/result.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/physics_pickup_search_v2.py --spec /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup_seed/spec.json --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup --workers 16 --samples 48 --iterations 12 --seed 44
for h in 0.06 0.08 0.04; do [ -f /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup_lift$h/result.json ] || $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/extra_lift.py /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup_lift$h $h 2.5 3.5; done
for h in 0.06 0.08 0.04; do $PY -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/release_refined_grasp_sweeper_snapshot.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup_lift$h --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/episodes/episode_000030 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/pickup_lift$h/release_neutral --side right --object mini_sweeper --start 12.3 --opening-origin neutral --workers 8 || echo "release $h failed"; done
