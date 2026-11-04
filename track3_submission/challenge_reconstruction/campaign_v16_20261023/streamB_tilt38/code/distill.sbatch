#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=sB-distill
#SBATCH --nodelist=node01
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/distill-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${TEACHER:?}" "${EPDIR:?}" "${OUT:?}"
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/actor_grid_sB.py --episode $EP --teacher "$TEACHER" --episode-dir "$EPDIR" --output "$OUT" --workers 8 ${GRID:+--grid "$GRID"}
