#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=robust33
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/logs/robust-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${T0:?}" "${T1:?}" "${NOISE:?}" "${OUT:?}" "${SEED:?}"
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/code/robust_spooncup.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/tail/c1_12.4_16.5 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000033 --output $OUT --start-s $T0 --stop-s $T1 --finger-noise $NOISE --seed $SEED --spoon-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_000033/wooden_spoon.obj --cup-mesh /data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes/meshes/episode_000033/blue_cup.obj
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/actor_grid_fine_legacy.py --episode 33 --source $OUT --output $OUT/actor_grid_fine --workers 16
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/actor_grid_legacy.py --episode 33 --source $OUT --output $OUT/actor_grid --workers 16
