#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=spooncup-tail
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/tail-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${SRC:?}" "${EPD:?}" "${ROOT:?}" "${WINS:?}" "${SEED:?}"
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
ROOT=$ROOT SRC=$SRC EPD=$EPD EP=$EP WINS="$WINS" SEED=$SEED bash /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/rsc_chain.sbatch
LAST=$(ls -d $ROOT/c* | sort -V | tail -1)
for d in $SRC $LAST; do [ -f $d/actor_grid/attempts.json ] || /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/actor_grid_legacy.py --episode $EP --source $d --output $d/actor_grid --workers 8; done
