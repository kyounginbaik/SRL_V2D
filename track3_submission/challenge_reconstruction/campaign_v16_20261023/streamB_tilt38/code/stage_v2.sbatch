#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=sB-stage
#SBATCH --nodelist=node01
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${DISTILLED:?}"
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/stage_sB_component.py --episode $EP --distilled "$DISTILLED" --base /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep$EP/staging_v2 ${LINEAGE:-}
