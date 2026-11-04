#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=spooncup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/rsc-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${SRC:?}" "${EPD:?}" "${OUT:?}" "${T0:?}" "${T1:?}" "${W:?}" "${SEED:?}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/retention_spooncup.py --source $SRC --episode-dir $EPD --output $OUT --start-s $T0 --stop-s $T1 --cup-weight $W --seed $SEED --scale ${SCALE:-0.5} --spoon-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_0000$EP/wooden_spoon.obj --cup-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_0000$EP/blue_cup.obj --samples 96 --iterations 12 --workers 16
